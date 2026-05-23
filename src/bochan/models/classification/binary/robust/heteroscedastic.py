
from __future__ import annotations

import copy
from typing import Callable, Optional, Sequence, List

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean

from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel,
)
from bochan.posteriors.bernoulli import SimpleBernoulliPosterior
from bochan.models.components.heteroscedastic import (
    HeteroscedasticNoiseModelMixin,
    align_like,
    check_categorical_columns_unchanged,
    clone_input_transform,
    concat_optional_train_yvar,
    ensure_2d_col,
    expand_observation_noise_tensor,
    fit_noise_model_mixed,
    fit_noise_model_single,
    fit_variational_classifier_mll,
    make_normalize_only_transform,
    predict_noise_var_from_log_noise_model,
    prepare_conditioning_data,
    prepare_train_yvar,
)


__all__ = [
    "HeteroscedasticBinaryClassificationGPModel",
    "HeteroscedasticBinaryClassificationMixedGPModel",
]


def _prepare_binary_targets(train_Y: Tensor, ref: Tensor) -> Tensor:
    """2値分類 target を [N] の float tensor にそろえる。"""
    y = train_Y
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    return y.reshape(-1).to(dtype=ref.dtype, device=ref.device)


def _estimate_classification_log_var_from_residuals(
    base_model,
    train_X: Tensor,
    train_Y: Tensor,
    *,
    min_noise: float,
) -> Tensor:
    """
    base classifier の p(x) と y の残差二乗から log variance target を作る。

    Notes:
        これは厳密な Bernoulli likelihood noise ではなく、
        局所的なラベル揺らぎ・誤分類傾向を表す pragmatic correction。
    """
    with torch.no_grad():
        post = base_model.posterior(train_X, observation_noise=False)
        p = post.mean
        if p.ndim > 1 and p.shape[-1] == 1:
            p = p.squeeze(-1)

        y = _prepare_binary_targets(train_Y, ref=p)
        noise = (y - p).pow(2).clamp_min(float(min_noise))
        return ensure_2d_col(noise.log())


class HeteroscedasticBinaryClassificationPosteriorMixin(HeteroscedasticNoiseModelMixin):
    """
    classification posterior に heteroscedastic correction を加える mixin。

    observation_noise=True のときだけ、SimpleBernoulliPosterior.variance に
    noise_model 由来の分散を加える。
    """

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> SimpleBernoulliPosterior:
        base_post = super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=False,
            posterior_transform=None,
            **kwargs,
        )

        mean = base_post.mean
        var = base_post.variance

        obs_noise = None
        if torch.is_tensor(observation_noise):
            obs_noise = expand_observation_noise_tensor(observation_noise, X)
            obs_noise = align_like(obs_noise, mean)
        elif observation_noise:
            obs_noise = self.predict_noise_var(X, ref_like=mean)

        if obs_noise is not None:
            var = var + align_like(obs_noise, var)

        posterior = SimpleBernoulliPosterior(mean=mean, variance=var)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior


class HeteroscedasticBinaryClassificationGPModel(
    HeteroscedasticBinaryClassificationPosteriorMixin,
    BinaryClassificationGPModel,
):
    """
    2値分類用の heteroscedastic GP。

    Args:
        train_X: raw-space の学習入力。
        train_Y: 0/1 の学習ラベル。
        train_Yvar: 既知の追加ノイズ分散。指定時は residual 推定より優先。
        likelihood: final classifier に渡す likelihood。
        input_transform: final classifier に渡す input transform。
            noise_model 学習では Normalize のみを抽出して使う。
        mean_module: final classifier に渡す mean module。
        covar_module: final classifier に渡す covariance module。
        num_inducing_points: 分類 SVGP の inducing point 数。
        inducing_points: 分類 SVGP の inducing points。
        learn_inducing_locations: inducing point を学習するか。
        aux_lr: 補助分類器の学習率。
        aux_num_epochs: 補助分類器の学習 epoch 数。
        aux_batch_size: 補助分類器の batch size。
        aux_shuffle: 補助分類器の DataLoader shuffle。
        min_noise: 追加ノイズ分散の下限。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
    ) -> None:
        train_X_raw = train_X.detach().clone()
        y_bin = _prepare_binary_targets(train_Y, ref=train_X_raw)

        self.train_inputs_raw = (train_X_raw,)
        self.min_noise = float(min_noise)

        noise_tf = make_normalize_only_transform(
            input_transform=input_transform,
            train_X=train_X_raw,
        )

        provided_yvar = prepare_train_yvar(
            train_Yvar=train_Yvar,
            ref_X=train_X_raw,
            min_noise=self.min_noise,
        )

        if provided_yvar is None:
            base_model = BinaryClassificationGPModel(
                train_X=train_X_raw,
                train_Y=y_bin,
                train_Yvar=None,
                likelihood=BernoulliLikelihood(),
                input_transform=clone_input_transform(noise_tf),
                mean_module=copy.deepcopy(mean_module),
                covar_module=copy.deepcopy(covar_module),
                num_inducing_points=num_inducing_points,
                inducing_points=inducing_points,
                learn_inducing_locations=learn_inducing_locations,
            )
            fit_variational_classifier_mll(
                base_model,
                lr=aux_lr,
                num_epochs=aux_num_epochs,
                batch_size=aux_batch_size,
                shuffle=aux_shuffle,
            )
            train_log_var = _estimate_classification_log_var_from_residuals(
                base_model=base_model,
                train_X=train_X_raw,
                train_Y=y_bin,
                min_noise=self.min_noise,
            )
        else:
            train_log_var = provided_yvar.log()

        noise_model = fit_noise_model_single(
            train_X=train_X_raw,
            train_Y_log_var=train_log_var,
            input_transform=clone_input_transform(noise_tf),
        )

        predicted_noise_var = predict_noise_var_from_log_noise_model(
            noise_model=noise_model,
            X=train_X_raw,
            ref_like=ensure_2d_col(y_bin),
            min_noise=self.min_noise,
        )

        super().__init__(
            train_X=train_X_raw,
            train_Y=y_bin,
            train_Yvar=predicted_noise_var.to(train_X_raw),
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

        self.train_inputs_raw = (train_X_raw,)
        self.train_Yvar = predicted_noise_var.to(train_X_raw)
        self.noise_model = noise_model
        self.noise_input_transform = noise_tf

        self._constructor_kwargs = {
            "likelihood": None,
            "input_transform": clone_input_transform(input_transform),
            "mean_module": copy.deepcopy(mean_module),
            "covar_module": copy.deepcopy(covar_module),
            "num_inducing_points": int(num_inducing_points),
            "inducing_points": None,
            "learn_inducing_locations": bool(learn_inducing_locations),
            "aux_lr": float(aux_lr),
            "aux_num_epochs": int(aux_num_epochs),
            "aux_batch_size": aux_batch_size,
            "aux_shuffle": bool(aux_shuffle),
            "min_noise": self.min_noise,
        }

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> "HeteroscedasticBinaryClassificationGPModel":
        if kwargs:
            raise NotImplementedError(
                f"Unsupported kwargs for {self.__class__.__name__}: {sorted(kwargs)}"
            )

        X_new, Y_new, Yvar_new = prepare_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.train_inputs_raw[0].shape[-1],
        )

        train_X_old = self.train_inputs_raw[0]
        train_Y_old = _prepare_binary_targets(self.train_targets, ref=train_X_old)

        X_full = torch.cat([train_X_old, X_new.to(train_X_old)], dim=0)
        Y_full = torch.cat([train_Y_old, Y_new.to(train_Y_old)], dim=0)

        Yvar_full = concat_optional_train_yvar(
            old_Y=train_Y_old,
            old_Yvar=getattr(self, "train_Yvar", None),
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            **self._constructor_kwargs,
        )
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


class HeteroscedasticBinaryClassificationMixedGPModel(
    HeteroscedasticBinaryClassificationPosteriorMixin,
    BinaryClassificationMixedGPModel,
):
    """
    mixed 入力対応の heteroscedastic binary classification GP。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        cont_kernel_factory: Optional[
            Callable[[torch.Size, int, Optional[List[int]]], Kernel]
        ] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be non-empty.")

        train_X_raw = train_X.detach().clone()
        y_bin = _prepare_binary_targets(train_Y, ref=train_X_raw)
        self.cat_dims = [int(i) for i in cat_dims]
        self.train_inputs_raw = (train_X_raw,)
        self.min_noise = float(min_noise)

        noise_tf = make_normalize_only_transform(
            input_transform=input_transform,
            train_X=train_X_raw,
            cat_dims=self.cat_dims,
        )

        if noise_tf is not None:
            noise_tf.eval()
            with torch.no_grad():
                X_check = noise_tf(train_X_raw)
            check_categorical_columns_unchanged(
                X=train_X_raw,
                X_tf=X_check,
                cat_dims=self.cat_dims,
                name=f"{self.__class__.__name__}.noise_input_transform",
            )

        provided_yvar = prepare_train_yvar(
            train_Yvar=train_Yvar,
            ref_X=train_X_raw,
            min_noise=self.min_noise,
        )

        if provided_yvar is None:
            base_model = BinaryClassificationMixedGPModel(
                train_X=train_X_raw,
                train_Y=y_bin,
                cat_dims=list(self.cat_dims),
                train_Yvar=None,
                likelihood=BernoulliLikelihood(),
                input_transform=clone_input_transform(noise_tf),
                mean_module=copy.deepcopy(mean_module),
                covar_module=copy.deepcopy(covar_module),
                cont_kernel_factory=cont_kernel_factory,
                num_inducing_points=num_inducing_points,
                inducing_points=inducing_points,
                learn_inducing_locations=learn_inducing_locations,
            )
            fit_variational_classifier_mll(
                base_model,
                lr=aux_lr,
                num_epochs=aux_num_epochs,
                batch_size=aux_batch_size,
                shuffle=aux_shuffle,
            )
            train_log_var = _estimate_classification_log_var_from_residuals(
                base_model=base_model,
                train_X=train_X_raw,
                train_Y=y_bin,
                min_noise=self.min_noise,
            )
        else:
            train_log_var = provided_yvar.log()

        noise_model = fit_noise_model_mixed(
            train_X=train_X_raw,
            train_Y_log_var=train_log_var,
            cat_dims=list(self.cat_dims),
            input_transform=clone_input_transform(noise_tf),
        )

        predicted_noise_var = predict_noise_var_from_log_noise_model(
            noise_model=noise_model,
            X=train_X_raw,
            ref_like=ensure_2d_col(y_bin),
            min_noise=self.min_noise,
        )

        super().__init__(
            train_X=train_X_raw,
            train_Y=y_bin,
            cat_dims=list(self.cat_dims),
            train_Yvar=predicted_noise_var.to(train_X_raw),
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

        self.train_inputs_raw = (train_X_raw,)
        self.train_Yvar = predicted_noise_var.to(train_X_raw)
        self.noise_model = noise_model
        self.noise_input_transform = noise_tf

        self._constructor_kwargs = {
            "cat_dims": list(self.cat_dims),
            "likelihood": None,
            "input_transform": clone_input_transform(input_transform),
            "mean_module": copy.deepcopy(mean_module),
            "covar_module": copy.deepcopy(covar_module),
            "cont_kernel_factory": cont_kernel_factory,
            "num_inducing_points": int(num_inducing_points),
            "inducing_points": None,
            "learn_inducing_locations": bool(learn_inducing_locations),
            "aux_lr": float(aux_lr),
            "aux_num_epochs": int(aux_num_epochs),
            "aux_batch_size": aux_batch_size,
            "aux_shuffle": bool(aux_shuffle),
            "min_noise": self.min_noise,
        }

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> "HeteroscedasticBinaryClassificationMixedGPModel":
        if kwargs:
            raise NotImplementedError(
                f"Unsupported kwargs for {self.__class__.__name__}: {sorted(kwargs)}"
            )

        X_new, Y_new, Yvar_new = prepare_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.train_inputs_raw[0].shape[-1],
        )

        train_X_old = self.train_inputs_raw[0]
        train_Y_old = _prepare_binary_targets(self.train_targets, ref=train_X_old)

        X_full = torch.cat([train_X_old, X_new.to(train_X_old)], dim=0)
        Y_full = torch.cat([train_Y_old, Y_new.to(train_Y_old)], dim=0)

        Yvar_full = concat_optional_train_yvar(
            old_Y=train_Y_old,
            old_Yvar=getattr(self, "train_Yvar", None),
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            **self._constructor_kwargs,
        )
        new_model.eval()
        new_model.likelihood.eval()
        return new_model
