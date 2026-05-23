from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices

from bochan.models.components.layers.feature_extractor import (
    LargeFeatureExtractor,
    SkipLargeFeatureExtractor,
)
from bochan.posteriors.bernoulli import SimpleBernoulliPosterior


# ============================================================
# Helpers
# ============================================================


def _prepare_binary_targets(train_Y: Tensor, train_X: Tensor) -> Tensor:
    """2値分類ラベルを shape=(n,) にそろえる。"""
    if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
        train_Y = train_Y.squeeze(-1)
    return train_Y.to(device=train_X.device, dtype=train_X.dtype).contiguous()


def _clone_tensor_tuple(inputs: Union[Tensor, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """Tensor または Tensor tuple を detach + clone して保持する。"""
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    return tuple(x.detach().clone() for x in inputs)


def _to_device_dtype_transform(
    input_transform: Optional[InputTransform],
    X: Tensor,
) -> Optional[InputTransform]:
    """input_transform を X の device / dtype に合わせる。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        input_transform = input_transform.to(X)
    if hasattr(input_transform, "eval"):
        input_transform.eval()
    return input_transform


def _expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """InputPerturbation 後の q 展開に合わせて raw X を repeat する。"""
    if X.shape == X_tf.shape:
        return X
    if X.ndim < 2 or X_tf.ndim < 2:
        return X
    if X.shape[-1] != X_tf.shape[-1]:
        return X
    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]
        if q_like == q:
            return X
        if q > 0 and q_like % q == 0:
            return X.repeat_interleave(q_like // q, dim=-2)
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)
    return X


def _check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """mixed model で input_transform がカテゴリ列を変更していないか確認する。"""
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = _expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            "input_transform must not modify categorical columns. "
            "For mixed models, transform only continuous columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def _make_train_X_tf(
    train_X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
    name: str = "input_transform",
) -> Tensor:
    """学習用の transformed X を作る。InputPerturbation の q 展開は許さない。"""
    if input_transform is None:
        return train_X

    if hasattr(input_transform, "train"):
        input_transform.train()
    train_X_tf = input_transform(train_X)
    if isinstance(train_X_tf, tuple):
        train_X_tf = train_X_tf[0]
    if hasattr(input_transform, "eval"):
        input_transform.eval()

    if train_X_tf.shape[-2] != train_X.shape[-2]:
        raise RuntimeError(
            f"{name} expanded training inputs. "
            f"train_X.shape={tuple(train_X.shape)}, "
            f"train_X_tf.shape={tuple(train_X_tf.shape)}. "
            "For InputPerturbation, use transform_on_train=False."
        )

    _check_categorical_columns_unchanged(
        X=train_X,
        X_tf=train_X_tf,
        cat_dims=cat_dims,
    )
    return train_X_tf


def _apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """評価用の input_transform を適用する。InputPerturbation の q 展開は許す。"""
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    if isinstance(X_tf, tuple):
        X_tf = X_tf[0]
    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _select_inducing_points(
    X: Tensor,
    num_inducing_points: int,
    inducing_points: Optional[Tensor] = None,
) -> Tensor:
    """入力空間の候補から誘導点を選択する。

    batched train_X が渡された場合でも、候補点方向を安全に flatten して
    inducing points を選ぶ。返り値は通常の [m, d]。
    """
    if inducing_points is not None:
        return inducing_points.to(X)

    X2d = X.reshape(-1, X.shape[-1])
    n = X2d.shape[-2]
    m = min(int(num_inducing_points), n)
    perm = torch.randperm(n, device=X.device)[:m]
    return X2d[perm].clone()


def _default_feature_extractor(
    input_dim: int,
    model_type: str = "DEFAULT",
) -> nn.Module:
    """既定の特徴抽出器を返す。"""
    if model_type.lower() == "skip":
        return SkipLargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dims=[input_dim * 8, input_dim * 4, input_dim * 2],
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
            use_global_skip=True,
        )

    return LargeFeatureExtractor(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dims=[input_dim * 8, input_dim * 4, input_dim * 2],
        activation="leaky_relu",
        dropout=0.0,
        use_bn=False,
    )


# ============================================================
# Inner latent models
# ============================================================


class _DeepKernelLatentBinarySVGP(ApproximateGP):
    """連続入力用の deep feature extractor + latent SVGP。"""

    def __init__(
        self,
        inducing_points: Tensor,
        feature_extractor: Optional[nn.Module] = None,
        train_inputs: Optional[Tensor] = None,
        train_targets: Optional[Tensor] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        # 誘導点は feature space ではなく、inner model の入力空間に置く。
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.deepkernel = feature_extractor
        with torch.no_grad():
            sample_feat = self.transform_features(inducing_points[:1])
        latent_dim = sample_feat.shape[-1]

        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=latent_dim,
                lengthscale_constraint=GreaterThan(1e-4),
            )
        )

        self.train_inputs = None if train_inputs is None else (train_inputs,)
        self.train_inputs_raw = _clone_tensor_tuple(train_inputs) if train_inputs is not None else None
        self.train_targets = train_targets

    def transform_features(self, X: Tensor) -> Tensor:
        """特徴抽出器を通した入力を返す。"""
        return X if self.deepkernel is None else self.deepkernel(X)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """latent f の分布を返す。"""
        Z = self.transform_features(X)
        mean_x = self.mean_module(Z)
        covar_x = self.covar_module(Z)
        return MultivariateNormal(mean_x, covar_x)


class _DeepKernelLatentBinaryMixedSVGP(ApproximateGP):
    """mixed 入力用の deep kernel latent SVGP。"""

    def __init__(
        self,
        inducing_points: Tensor,
        cat_dims: Sequence[int],
        feature_extractor: Optional[nn.Module] = None,
        train_inputs: Optional[Tensor] = None,
        train_targets: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        d = inducing_points.shape[-1]
        self.cat_dims = list(normalize_indices(indices=cat_dims, d=d))
        self.ord_dims = sorted(set(range(d)) - set(self.cat_dims))

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        if len(self.ord_dims) > 0:
            self.deepkernel = feature_extractor
            self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)
        else:
            self.deepkernel = nn.Identity()
            self.scale_to_bounds = nn.Identity()

        self.mean_module = ConstantMean()

        if len(self.ord_dims) == 0:
            self.covar_module = ScaleKernel(
                CategoricalKernel(
                    ard_num_dims=len(self.cat_dims),
                    active_dims=self.cat_dims,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        else:
            cont_kernel = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=len(self.ord_dims),
                active_dims=self.ord_dims,
            )
            cat_kernel = CategoricalKernel(
                ard_num_dims=len(self.cat_dims),
                active_dims=self.cat_dims,
                lengthscale_constraint=GreaterThan(1e-6),
            )
            sum_kernel = ScaleKernel(cont_kernel + ScaleKernel(cat_kernel))
            prod_kernel = ScaleKernel(cont_kernel * cat_kernel)
            self.covar_module = sum_kernel + prod_kernel

        self.train_inputs = None if train_inputs is None else (train_inputs,)
        self.train_inputs_raw = _clone_tensor_tuple(train_inputs) if train_inputs is not None else None
        self.train_targets = train_targets

    def _combine_cont_and_cat(self, X: Tensor) -> Tensor:
        """連続列だけ特徴抽出器に通し、カテゴリ列はそのまま戻す。"""
        if len(self.ord_dims) == 0:
            return X

        cont_x = X[..., self.ord_dims]
        cat_x = X[..., self.cat_dims]

        projected_cont_x = self.deepkernel(cont_x)
        projected_cont_x = self.scale_to_bounds(projected_cont_x)

        out = torch.empty_like(X)
        out[..., self.ord_dims] = projected_cont_x
        out[..., self.cat_dims] = cat_x
        return out

    def forward(self, X: Tensor) -> MultivariateNormal:
        """latent f の分布を返す。"""
        mixed_x = self._combine_cont_and_cat(X)
        mean_x = self.mean_module(mixed_x)
        covar_x = self.covar_module(mixed_x)
        return MultivariateNormal(mean_x, covar_x)


# ============================================================
# Base wrapper
# ============================================================


class _BaseDeepKernelBinaryClassificationModel(ApproximateGPyTorchModel):
    """DeepKernel 2値分類モデルの共通 wrapper。"""

    def __init__(
        self,
        latent_model: ApproximateGP,
        likelihood: BernoulliLikelihood,
        input_transform: Optional[InputTransform],
        train_X: Tensor,
        train_Y: Tensor,
    ) -> None:
        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )
        self._model_dtype = train_X.dtype
        self._model_device = train_X.device
        self.train_inputs = (train_X,)
        self.train_inputs_raw = _clone_tensor_tuple(train_X)
        self.train_targets = train_Y
        self._train_targets = train_Y
        self.input_transform = input_transform
        self.to(train_X)

    @staticmethod
    def _unwrap_inputs(inputs) -> Tensor:
        if isinstance(inputs, tuple):
            return inputs[0]
        return inputs

    def _to_model_dtype_device(self, X: Tensor) -> Tensor:
        return X.to(device=self._model_device, dtype=self._model_dtype)

    def _set_transformed_inputs(self) -> None:
        """BoTorch の eval 時 transformed input 自動更新を無効化する。"""
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """評価時の input_transform を適用する。"""
        return _apply_input_transform_for_eval(
            X,
            getattr(self, "input_transform", None),
            cat_dims=getattr(self, "cat_dims", None),
        )

    def _apply_input_transform(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> Tensor:
        if apply_input_transform and self.input_transform is not None:
            return self.transform_inputs(X)
        return X

    def forward(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> MultivariateNormal:
        """latent SVGP distribution を返す。"""
        X = self._to_model_dtype_device(self._unwrap_inputs(X))
        X = self._apply_input_transform(X, apply_input_transform=apply_input_transform)
        return self.model(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ):
        """p(y=1|x) の posterior を返す。"""
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )

        _ = observation_noise
        self.eval()
        self.likelihood.eval()

        X = self._to_model_dtype_device(self._unwrap_inputs(X))
        X_tf = self._apply_input_transform(X, apply_input_transform=True)

        latent_dist = self.model(X_tf)
        pred_dist = self.likelihood(latent_dist)

        p = pred_dist.mean
        var = pred_dist.variance

        if p.ndim == X_tf.ndim - 1:
            p = p.unsqueeze(-1)
            var = var.unsqueeze(-1)

        posterior = SimpleBernoulliPosterior(mean=p, variance=var)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def probability_posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> SimpleBernoulliPosterior:
        """classification acquisition 用に probability posterior を明示名で返す。"""
        return self.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def predict_proba(self, X: Tensor) -> Tensor:
        """p(y=1|x) を返す簡易関数。"""
        return self.posterior(X).mean

    def set_train_data(
        self,
        inputs: Optional[Union[Tensor, tuple[Tensor, ...]]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        """raw 入力を受け取り、inner model 側には transformed X を反映する。"""
        _ = strict
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            raw_X = self._to_model_dtype_device(inputs[0])
            self.train_inputs = (raw_X,)
            self.train_inputs_raw = _clone_tensor_tuple(raw_X)

            X_tf = _make_train_X_tf(
                raw_X,
                self.input_transform,
                cat_dims=getattr(self, "cat_dims", None),
                name=f"{self.__class__.__name__}.input_transform",
            )
            self.model.train_inputs = (X_tf,)
            self.model.train_inputs_raw = _clone_tensor_tuple(X_tf)

        if targets is not None:
            targets = _prepare_binary_targets(targets, self.train_inputs[0])
            self.train_targets = targets
            self._train_targets = targets
            self.model.train_targets = targets

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def make_mll(self, beta: float = 1.0) -> VariationalELBO:
        """この wrapper 用の VariationalELBO を返す。

        MLL の model は inner latent SVGP なので、num_data も inner model の
        train_inputs に合わせる。wrapper 側の train_inputs は raw-space 保持用。
        """
        inner_train_X = self.model.train_inputs[0]
        inner_train_Y = self.model.train_targets

        if inner_train_X.shape[-2] != inner_train_Y.shape[0]:
            raise RuntimeError(
                "inner train_inputs and train_targets have inconsistent data sizes. "
                f"inner_train_X.shape={tuple(inner_train_X.shape)}, "
                f"inner_train_Y.shape={tuple(inner_train_Y.shape)}. "
                "For InputPerturbation, use transform_on_train=False."
            )

        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=inner_train_X.shape[-2],
            beta=beta,
        )

    def latent_posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        apply_input_transform: bool = True,
        **kwargs,
    ):
        """latent f の posterior を返す。"""
        self.eval()
        X = self._to_model_dtype_device(self._unwrap_inputs(X))
        latent_dist = self.forward(
            X,
            apply_input_transform=apply_input_transform,
        )
        posterior = GPyTorchPosterior(latent_dist)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def posterior_latent(self, X, **kwargs):
        return self.latent_posterior(X, **kwargs)

    def posterior_f(self, X, **kwargs):
        return self.latent_posterior(X, **kwargs)


class DeepKernelBinaryClassificationGPModel(_BaseDeepKernelBinaryClassificationModel):
    """連続入力向け DeepKernel binary classification model。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 64,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        model_type: str = "DEFAULT",
    ) -> None:
        train_Y = _prepare_binary_targets(train_Y, train_X)
        input_transform = _to_device_dtype_transform(input_transform, train_X)
        train_X_tf = _make_train_X_tf(
            train_X,
            input_transform,
            name="DeepKernelClassificationGPModel.input_transform",
        )

        if feature_extractor is None:
            feature_extractor = _default_feature_extractor(
                input_dim=train_X.shape[-1],
                model_type=model_type,
            )
        feature_extractor = feature_extractor.to(train_X)

        # 誘導点は feature space ではなく、inner model の入力空間から選ぶ。
        inducing_points = _select_inducing_points(
            train_X_tf,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )

        latent_model = _DeepKernelLatentBinarySVGP(
            inducing_points=inducing_points,
            feature_extractor=feature_extractor,
            train_inputs=train_X_tf,
            train_targets=train_Y,
            mean_module=mean_module,
            covar_module=covar_module,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            input_transform=input_transform,
            train_X=train_X,
            train_Y=train_Y,
        )


class DeepKernelBinaryClassificationMixedGPModel(_BaseDeepKernelBinaryClassificationModel):
    """混合入力（連続 + カテゴリ）向け DeepKernel binary classification model。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 64,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        model_type: str = "DEFAULT",
    ) -> None:
        _ = mean_module, covar_module
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        train_Y = _prepare_binary_targets(train_Y, train_X)

        d = train_X.shape[-1]
        cat_dims = list(normalize_indices(indices=cat_dims, d=d))
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims

        input_transform = _to_device_dtype_transform(input_transform, train_X)
        train_X_tf = _make_train_X_tf(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name="DeepKernelClassificationMixedGPModel.input_transform",
        )

        if feature_extractor is None:
            if len(ord_dims) > 0:
                feature_extractor = _default_feature_extractor(
                    input_dim=len(ord_dims),
                    model_type=model_type,
                )
            else:
                feature_extractor = nn.Identity()
        feature_extractor = feature_extractor.to(train_X)

        inducing_points = _select_inducing_points(
            train_X_tf,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )

        latent_model = _DeepKernelLatentBinaryMixedSVGP(
            inducing_points=inducing_points,
            cat_dims=cat_dims,
            feature_extractor=feature_extractor,
            train_inputs=train_X_tf,
            train_targets=train_Y,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            input_transform=input_transform,
            train_X=train_X,
            train_Y=train_Y,
        )
        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims


__all__ = [
    "DeepKernelBinaryClassificationGPModel",
    "DeepKernelBinaryClassificationMixedGPModel",
]
