
from __future__ import annotations

import copy
from typing import Optional, Sequence

import gpytorch
import torch
from torch import Tensor

from botorch.models.transforms.input import InputTransform

from bochan.fit.ordinal import fit_ordinal_gp
from bochan.models.ordinal.base.models import (
    OrdinalGPModel,
    OrdinalMixedGPModel,
)
from bochan.models.components.heteroscedastic import (
    HeteroscedasticLatentPosteriorMixin,
    clone_input_transform,
    ensure_2d_col,
    fit_noise_model_mixed,
    fit_noise_model_single,
    make_normalize_only_transform,
    predict_noise_var_from_log_noise_model,
    prepare_train_yvar,
)


__all__ = [
    "HeteroscedasticOrdinalGPModel",
    "HeteroscedasticOrdinalMixedGPModel",
]


def _flatten_ordinal_targets(train_Y: Tensor, ref_X: Tensor) -> Tensor:
    """ordinal target を [N] の long tensor にそろえる。"""
    y = torch.as_tensor(train_Y, device=ref_X.device)
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    return y.reshape(-1).long()


def _compute_ordinal_log_var_from_expected_score(
    base_model,
    train_X: Tensor,
    train_Y: Tensor,
    *,
    min_noise: float,
) -> Tensor:
    """
    ordinal class probability の期待スコアと実ラベルの残差から log variance target を作る。

    Notes:
        厳密な ordinal likelihood の観測ノイズではなく、
        順序ラベルの局所的な揺らぎ・モデル誤差を表す pragmatic correction。
    """
    with torch.no_grad():
        probs = base_model.class_probs(train_X)
        utilities = torch.arange(
            probs.shape[-1],
            device=probs.device,
            dtype=probs.dtype,
        )
        expected_score = (probs * utilities.view(1, -1)).sum(dim=-1)

        y = _flatten_ordinal_targets(train_Y, train_X).to(
            device=expected_score.device,
            dtype=expected_score.dtype,
        )
        residual_sq = (expected_score - y).pow(2).clamp_min(float(min_noise))
        return residual_sq.unsqueeze(-1).log()


class HeteroscedasticOrdinalGPModel(
    HeteroscedasticLatentPosteriorMixin,
    OrdinalGPModel,
):
    """
    入力依存ノイズを持つ ordinal GP。

    Args:
        train_X: raw-space の学習入力。
        train_Y: ordinal label。0, 1, ..., K-1 を想定。
        num_classes: クラス数。
        train_Yvar: 既知の追加ノイズ分散。指定時は residual 推定より優先。
        input_transform: final ordinal model に渡す input transform。
            noise_model 学習では Normalize のみを抽出して使う。
        aux_lr: 補助 ordinal model の学習率。
        aux_num_epochs: 補助 ordinal model の学習 epoch 数。
        aux_batch_size: 補助 ordinal model の batch size。
        min_noise: noise variance の下限。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 200,
        aux_batch_size: Optional[int] = None,
        min_noise: float = 1e-6,
        input_transform: Optional[InputTransform] = None,
        train_Yvar: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        train_X_raw = train_X.detach().clone()
        train_Y_ord = _flatten_ordinal_targets(train_Y, train_X_raw)

        self.train_inputs_raw = (train_X_raw,)
        self.min_noise = float(min_noise)

        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.verbose = bool(verbose)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.aux_lr = float(aux_lr)
        self.aux_num_epochs = int(aux_num_epochs)
        self.aux_batch_size = aux_batch_size

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
            base_model = OrdinalGPModel(
                train_X=train_X_raw,
                train_Y=train_Y_ord,
                num_classes=num_classes,
                inducing_points_num=inducing_points_num,
                inducing_points=inducing_points,
                learn_inducing_locations=learn_inducing_locations,
                mean_module=copy.deepcopy(mean_module),
                covar_module=copy.deepcopy(covar_module),
                input_transform=clone_input_transform(noise_tf),
                eps=eps,
                init_gap=init_gap,
                fix_first_cutpoint=fix_first_cutpoint,
                conditioning_steps=conditioning_steps,
                conditioning_lr=conditioning_lr,
                conditioning_batch_size=conditioning_batch_size,
            )
            fit_ordinal_gp(
                base_model,
                num_epochs=self.aux_num_epochs,
                lr=self.aux_lr,
                batch_size=self.aux_batch_size,
                verbose=False,
            )
            train_log_var = _compute_ordinal_log_var_from_expected_score(
                base_model=base_model,
                train_X=train_X_raw,
                train_Y=train_Y_ord,
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
            ref_like=ensure_2d_col(train_Y_ord.to(train_X_raw)),
            min_noise=self.min_noise,
        )

        super().__init__(
            train_X=train_X_raw,
            train_Y=train_Y_ord,
            num_classes=num_classes,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )

        self.noise_model = noise_model
        self.noise_input_transform = noise_tf
        self.train_Yvar = predicted_noise_var.to(train_X_raw)
        self.train_inputs_raw = (train_X_raw,)
        self.train_targets = train_Y_ord

        self._constructor_kwargs = {
            "num_classes": int(num_classes),
            "inducing_points_num": int(inducing_points_num),
            "learn_inducing_locations": bool(learn_inducing_locations),
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "use_predictive_log_likelihood": self.use_predictive_log_likelihood,
            "fix_first_cutpoint": bool(fix_first_cutpoint),
            "init_gap": float(init_gap),
            "eps": float(eps),
            "verbose": self.verbose,
            "conditioning_steps": self.conditioning_steps,
            "conditioning_lr": self.conditioning_lr,
            "conditioning_batch_size": self.conditioning_batch_size,
            "aux_lr": self.aux_lr,
            "aux_num_epochs": self.aux_num_epochs,
            "aux_batch_size": self.aux_batch_size,
            "min_noise": self.min_noise,
            "input_transform": clone_input_transform(input_transform),
            "inducing_points": None,
            "mean_module": copy.deepcopy(mean_module),
            "covar_module": copy.deepcopy(covar_module),
        }

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "HeteroscedasticOrdinalGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for HeteroscedasticOrdinalGPModel.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            train_Yvar=None,
            **self._constructor_kwargs,
        )

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            if refit_lr is None:
                refit_lr = self.lr
            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            if refit_bs is None:
                refit_bs = self.batch_size

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


class HeteroscedasticOrdinalMixedGPModel(
    HeteroscedasticLatentPosteriorMixin,
    OrdinalMixedGPModel,
):
    """
    mixed 入力対応の heteroscedastic ordinal GP。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int] = (),
        category_counts: Optional[dict[int, int]] = None,
        category_values: Optional[dict[int, Sequence[int | float]]] = None,
        cont_kernel: str = "matern52",
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 200,
        aux_batch_size: Optional[int] = None,
        min_noise: float = 1e-6,
        input_transform: Optional[InputTransform] = None,
        train_Yvar: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be non-empty.")

        train_X_raw = train_X.detach().clone()
        train_Y_ord = _flatten_ordinal_targets(train_Y, train_X_raw)
        cat_dims = [int(i) for i in cat_dims]

        self.train_inputs_raw = (train_X_raw,)
        self.cat_dims = list(cat_dims)
        self.min_noise = float(min_noise)

        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.verbose = bool(verbose)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.aux_lr = float(aux_lr)
        self.aux_num_epochs = int(aux_num_epochs)
        self.aux_batch_size = aux_batch_size

        noise_tf = make_normalize_only_transform(
            input_transform=input_transform,
            train_X=train_X_raw,
            cat_dims=cat_dims,
        )

        provided_yvar = prepare_train_yvar(
            train_Yvar=train_Yvar,
            ref_X=train_X_raw,
            min_noise=self.min_noise,
        )

        if provided_yvar is None:
            base_model = OrdinalMixedGPModel(
                train_X=train_X_raw,
                train_Y=train_Y_ord,
                cat_dims=cat_dims,
                num_classes=num_classes,
                category_counts=copy.deepcopy(category_counts),
                category_values=copy.deepcopy(category_values),
                cont_kernel=cont_kernel,
                inducing_points_num=inducing_points_num,
                inducing_points=inducing_points,
                learn_inducing_locations=learn_inducing_locations,
                mean_module=copy.deepcopy(mean_module),
                covar_module=copy.deepcopy(covar_module),
                input_transform=clone_input_transform(noise_tf),
                eps=eps,
                init_gap=init_gap,
                fix_first_cutpoint=fix_first_cutpoint,
                conditioning_steps=conditioning_steps,
                conditioning_lr=conditioning_lr,
                conditioning_batch_size=conditioning_batch_size,
            )
            fit_ordinal_gp(
                base_model,
                num_epochs=self.aux_num_epochs,
                lr=self.aux_lr,
                batch_size=self.aux_batch_size,
                verbose=False,
            )
            train_log_var = _compute_ordinal_log_var_from_expected_score(
                base_model=base_model,
                train_X=train_X_raw,
                train_Y=train_Y_ord,
                min_noise=self.min_noise,
            )
        else:
            train_log_var = provided_yvar.log()

        noise_model = fit_noise_model_mixed(
            train_X=train_X_raw,
            train_Y_log_var=train_log_var,
            cat_dims=cat_dims,
            input_transform=clone_input_transform(noise_tf),
        )

        predicted_noise_var = predict_noise_var_from_log_noise_model(
            noise_model=noise_model,
            X=train_X_raw,
            ref_like=ensure_2d_col(train_Y_ord.to(train_X_raw)),
            min_noise=self.min_noise,
        )

        super().__init__(
            train_X=train_X_raw,
            train_Y=train_Y_ord,
            cat_dims=cat_dims,
            num_classes=num_classes,
            category_counts=copy.deepcopy(category_counts),
            category_values=copy.deepcopy(category_values),
            cont_kernel=cont_kernel,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )

        self.noise_model = noise_model
        self.noise_input_transform = noise_tf
        self.train_Yvar = predicted_noise_var.to(train_X_raw)
        self.train_inputs_raw = (train_X_raw,)
        self.train_targets = train_Y_ord
        self.cat_dims = list(cat_dims)
        self.cont_kernel = str(cont_kernel)

        self._constructor_kwargs = {
            "num_classes": int(num_classes),
            "cat_dims": list(cat_dims),
            "category_counts": copy.deepcopy(category_counts),
            "category_values": copy.deepcopy(category_values),
            "cont_kernel": str(cont_kernel),
            "inducing_points_num": int(inducing_points_num),
            "learn_inducing_locations": bool(learn_inducing_locations),
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "use_predictive_log_likelihood": self.use_predictive_log_likelihood,
            "fix_first_cutpoint": bool(fix_first_cutpoint),
            "init_gap": float(init_gap),
            "eps": float(eps),
            "verbose": self.verbose,
            "conditioning_steps": self.conditioning_steps,
            "conditioning_lr": self.conditioning_lr,
            "conditioning_batch_size": self.conditioning_batch_size,
            "aux_lr": self.aux_lr,
            "aux_num_epochs": self.aux_num_epochs,
            "aux_batch_size": self.aux_batch_size,
            "min_noise": self.min_noise,
            "input_transform": clone_input_transform(input_transform),
            "inducing_points": None,
            "mean_module": copy.deepcopy(mean_module),
            "covar_module": copy.deepcopy(covar_module),
        }

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "HeteroscedasticOrdinalMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for HeteroscedasticOrdinalMixedGPModel.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            train_Yvar=None,
            **self._constructor_kwargs,
        )

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            if refit_lr is None:
                refit_lr = self.lr
            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            if refit_bs is None:
                refit_bs = self.batch_size

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model
