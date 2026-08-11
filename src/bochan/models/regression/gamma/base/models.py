from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ProductKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

from bochan.models.regression.gamma._components import (
    GammaLink,
    GammaLogLikelihood,
    GammaPosterior,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    build_default_gamma_covar_module,
    check_categorical_columns_unchanged,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    prepare_positive_targets,
    select_inducing_points,
    to_device_dtype_transform,
)


def clone_outcome_transform(outcome_transform: OutcomeTransform | None) -> OutcomeTransform | None:
    """OutcomeTransform を安全に複製する。"""
    return None if outcome_transform is None else copy.deepcopy(outcome_transform)


def to_device_dtype_outcome_transform(
    outcome_transform: OutcomeTransform | None,
    ref: Tensor,
) -> OutcomeTransform | None:
    """OutcomeTransform を ref と同じ device / dtype に移す。"""
    if outcome_transform is not None and hasattr(outcome_transform, "to"):
        outcome_transform = outcome_transform.to(device=ref.device, dtype=ref.dtype)
    return outcome_transform


def apply_outcome_transform_for_training(
    train_Y: Tensor,
    train_X: Tensor,
    outcome_transform: OutcomeTransform | None,
    *,
    min_mean: float,
    name: str,
) -> tuple[Tensor, Tensor, OutcomeTransform | None]:
    """Gamma 用に正値を保つ形で outcome_transform を適用する。

    Returns:
        raw_train_Y:
            元スケールの正値 target。condition_on_observations 用に保持する。
        transformed_train_Y:
            likelihood / latent model に渡す model-scale target。
        outcome_transform:
            device / dtype を揃えた transform。
    """
    raw_train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
    outcome_transform = to_device_dtype_outcome_transform(outcome_transform, train_X)

    if outcome_transform is None:
        return raw_train_Y, raw_train_Y, None

    Y_for_transform = raw_train_Y.unsqueeze(-1) if raw_train_Y.ndim == 1 else raw_train_Y
    transformed_Y, _ = outcome_transform(Y_for_transform, X=train_X)
    transformed_train_Y = prepare_positive_targets(
        transformed_Y,
        train_X,
        min_value=min_mean,
    )

    if transformed_train_Y.shape[-1:] == torch.Size([1]) and transformed_train_Y.ndim > 1:
        transformed_train_Y = transformed_train_Y.squeeze(-1)

    if transformed_train_Y.shape != raw_train_Y.shape:
        try:
            transformed_train_Y = transformed_train_Y.reshape_as(raw_train_Y)
        except RuntimeError as err:
            raise RuntimeError(
                f"{name} produced an insupported target shape. "
                f"raw_train_Y.shape={tuple(raw_train_Y.shape)}, "
                f"transformed_train_Y.shape={tuple(transformed_train_Y.shape)}."
            ) from err

    return raw_train_Y, transformed_train_Y.contiguous(), outcome_transform


def _make_cat_kernel(cat_dims: Sequence[int], batch_shape: torch.Size) -> ScaleKernel:
    return ScaleKernel(
        CategoricalKernel(
            active_dims=tuple(cat_dims),
            ard_num_dims=len(cat_dims),
            batch_shape=batch_shape,
        ),
        batch_shape=batch_shape,
    )


def _make_cont_kernel(cont_dims: Sequence[int], batch_shape: torch.Size) -> Kernel:
    return get_covar_module_with_dim_scaled_prior(
        batch_shape=batch_shape,
        ard_num_dims=len(cont_dims),
        active_dims=tuple(cont_dims),
    )


def build_mixed_gamma_kernel(
    d: int,
    cat_dims: Sequence[int],
    batch_shape: torch.Size | None = None,
) -> Kernel:
    """Gamma mixed model 用の continuous + categorical kernel を作る。"""
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    cat_dims = normalize_dims(cat_dims, d)
    cont_dims = get_cont_dims(d, cat_dims)

    if len(cat_dims) == 0:
        return _make_cont_kernel(cont_dims, batch_shape=batch_shape)
    if len(cont_dims) == 0:
        return _make_cat_kernel(cat_dims, batch_shape=batch_shape)

    cont_1 = _make_cont_kernel(cont_dims, batch_shape=batch_shape)
    cont_2 = _make_cont_kernel(cont_dims, batch_shape=batch_shape)
    cat_1 = _make_cat_kernel(cat_dims, batch_shape=batch_shape)
    cat_2 = _make_cat_kernel(cat_dims, batch_shape=batch_shape)

    return cont_1 + cat_1 + ProductKernel(cont_2, cat_2)


class _LatentGammaSVGP(ApproximateGP):
    """Gamma 回帰用の latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        inducing_points: Tensor | None = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
    ) -> None:
        inducing_points = select_inducing_points(
            train_X,
            num_inducing_points=num_inducing,
            inducing_points=inducing_points,
        )
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
        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or build_default_gamma_covar_module(train_X)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class _LatentMixedGammaSVGP(ApproximateGP):
    """mixed 入力 Gamma 回帰用 latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        inducing_points: Tensor | None = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
    ) -> None:
        d = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, d)
        self.cont_dims = get_cont_dims(d, self.cat_dims)
        self._ignore_X_dims_scaling_check = self.cat_dims

        inducing_points = select_inducing_points(
            train_X,
            num_inducing_points=num_inducing,
            inducing_points=inducing_points,
        )
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
        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or build_mixed_gamma_kernel(
            d=d,
            cat_dims=self.cat_dims,
            batch_shape=torch.Size(),
        )
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class _BaseGammaGPModel(ApproximateGPyTorchModel):
    """
    Gamma 回帰 wrapper の共通基底。

    Notes:
        - wrapper は raw-space X を受け取る。
        - latent model は input_transform 後の X を保持する。
        - likelihood / latent model は outcome_transform 後の Y で学習する。
        - posterior(X) は、outcome_transform があれば元スケールに戻した posterior を返す。
        - latent_posterior(X) は GPyTorchPosterior を返す。
    """

    def __init__(
        self,
        *,
        latent_model: ApproximateGP,
        likelihood: GammaLogLikelihood,
        train_X: Tensor,
        train_Y: Tensor,
        input_transform: InputTransform | None,
        outcome_transform: OutcomeTransform | None = None,
        train_Y_raw: Tensor | None = None,
        cat_dims: Sequence[int] | None = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        link: GammaLink = "softplus",
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
    ) -> None:
        super().__init__(model=latent_model, likelihood=likelihood, num_outputs=1)
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform
        self.cat_dims = None if cat_dims is None else list(cat_dims)

        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self.train_targets_raw = prepare_positive_targets(
            train_Y if train_Y_raw is None else train_Y_raw,
            train_X,
            min_value=min_mean,
        )
        self.train_targets = prepare_positive_targets(train_Y, train_X, min_value=min_mean)

        self.num_inducing = int(num_inducing)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)

        self.to(train_X)

    def _set_transformed_inputs(self) -> None:
        """BoTorch の eval 時自動 transform を無効化する。"""
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """raw-space X を latent model の入力空間へ写像する。"""
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
        )

    def transform_outcomes(self, Y: Tensor, X: Tensor | None = None) -> Tensor:
        """raw target を model-scale target に変換する。"""
        Y = prepare_positive_targets(Y, self.train_inputs_raw[0] if X is None else X, min_value=self.min_mean)
        if self.outcome_transform is None:
            return Y
        Y_for_transform = Y.unsqueeze(-1) if Y.ndim == 1 else Y
        Y_tf, _ = self.outcome_transform(Y_for_transform, X=X)
        return prepare_positive_targets(Y_tf, self.train_inputs_raw[0] if X is None else X, min_value=self.min_mean)

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if output_indices is not None:
            raise NotImplementedError(f"{self.__class__.__name__} does not support output_indices.")
        if isinstance(X, tuple):
            X = X[0]

        self.eval()
        X_tf = self.transform_inputs(X)
        latent_dist = self.model(X_tf)
        posterior = GPyTorchPosterior(latent_dist)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = True,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> Posterior:
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support tensor observation_noise."
            )
        latent_post = self.latent_posterior(
            X,
            output_indices=output_indices,
            posterior_transform=None,
            **kwargs,
        )
        posterior: Posterior = GammaPosterior(
            latent_posterior=latent_post,
            likelihood=self.likelihood,
            add_observation_noise=bool(observation_noise),
        )
        if self.outcome_transform is not None:
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def predict_mean(self, X: Tensor) -> Tensor:
        """Gamma mean μ の予測値を返す。"""
        return self.posterior(X, observation_noise=True).mean

    def predict_concentration(self) -> Tensor:
        """Gamma concentration κ を返す。"""
        return self.likelihood.concentration

    def predict_rate_parameter(self, X: Tensor) -> Tensor:
        """Gamma distribution の rate parameter κ / μ を返す。"""
        mean = self.predict_mean(X).clamp_min(self.min_mean)
        concentration = self.predict_concentration().to(device=mean.device, dtype=mean.dtype)
        return concentration / mean

    def make_mll(self) -> VariationalELBO:
        """VariationalELBO を作る。"""
        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.train_inputs_raw[0].shape[-2],
        )


class GammaGPModel(_BaseGammaGPModel):
    """連続入力用 Gamma SVGP 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        likelihood: GammaLogLikelihood | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        num_inducing: int = 128,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        outcome_transform = clone_outcome_transform(outcome_transform)
        raw_train_Y, train_Y, outcome_transform = apply_outcome_transform_for_training(
            train_Y=train_Y,
            train_X=train_X,
            outcome_transform=outcome_transform,
            min_mean=min_mean,
            name="GammaGPModel.outcome_transform",
        )
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            name="GammaGPModel.input_transform",
        )

        likelihood = likelihood or GammaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            exp_clip=exp_clip,
            min_mean=min_mean,
            min_concentration=min_concentration,
        )
        latent_model = _LatentGammaSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            inducing_points=inducing_points,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            train_Y_raw=raw_train_Y,
            cat_dims=None,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.min_concentration = float(min_concentration)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> GammaGPModel:
        if kwargs.get("noise") is not None:
            raise NotImplementedError("GammaGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_positive_targets(Y, X, min_value=self.min_mean)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets_raw, Y], dim=0)
        return self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            outcome_transform=clone_outcome_transform(self.outcome_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing=self.num_inducing,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            link=self.link,
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            exp_clip=self.exp_clip,
            min_mean=self.min_mean,
            min_concentration=self.min_concentration,
        )


class GammaMixedGPModel(_BaseGammaGPModel):
    """連続 + カテゴリ mixed 入力用 Gamma SVGP 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        likelihood: GammaLogLikelihood | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        num_inducing: int = 128,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        d = train_X.shape[-1]
        cat_dims = normalize_dims(cat_dims, d)
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be non-empty for GammaMixedGPModel.")
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        outcome_transform = clone_outcome_transform(outcome_transform)
        raw_train_Y, train_Y, outcome_transform = apply_outcome_transform_for_training(
            train_Y=train_Y,
            train_X=train_X,
            outcome_transform=outcome_transform,
            min_mean=min_mean,
            name="GammaMixedGPModel.outcome_transform",
        )

        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name="GammaMixedGPModel.input_transform",
        )
        check_categorical_columns_unchanged(train_X, train_X_tf, cat_dims=cat_dims)

        likelihood = likelihood or GammaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            exp_clip=exp_clip,
            min_mean=min_mean,
            min_concentration=min_concentration,
        )
        latent_model = _LatentMixedGammaSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            cat_dims=cat_dims,
            inducing_points=inducing_points,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            train_Y_raw=raw_train_Y,
            cat_dims=cat_dims,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )
        self.cat_dims = list(cat_dims)
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.min_concentration = float(min_concentration)


__all__ = [
    "GammaLogLikelihood",
    "GammaPosterior",
    "GammaGPModel",
    "GammaMixedGPModel",
    "apply_outcome_transform_for_training",
    "build_mixed_gamma_kernel",
    "clone_outcome_transform",
]
