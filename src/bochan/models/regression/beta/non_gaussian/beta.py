from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ProductKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from bochan.models.components.beta import (
    BetaMeanLink,
    BetaLogLikelihood,
    BetaPosterior,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    build_default_beta_covar_module,
    check_categorical_columns_unchanged,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    prepare_beta_targets,
    select_inducing_points,
    to_device_dtype_transform,
)


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


def build_mixed_beta_kernel(d: int, cat_dims: Sequence[int], batch_shape: torch.Size = torch.Size()) -> Kernel:
    """Beta mixed model 用の continuous + categorical kernel を作る。"""
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


class _LatentBetaSVGP(ApproximateGP):
    """Beta 回帰用の latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        inducing_points = select_inducing_points(train_X, num_inducing_points=num_inducing_points, inducing_points=inducing_points)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.shape[-2])
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or build_default_beta_covar_module(train_X)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class _LatentMixedBetaSVGP(ApproximateGP):
    """mixed 入力 Beta 回帰用 latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        d = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, d)
        self.cont_dims = get_cont_dims(d, self.cat_dims)
        self._ignore_X_dims_scaling_check = self.cat_dims
        inducing_points = select_inducing_points(train_X, num_inducing_points=num_inducing_points, inducing_points=inducing_points)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.shape[-2])
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or build_mixed_beta_kernel(d=d, cat_dims=self.cat_dims, batch_shape=torch.Size())
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class _BaseBetaGPModel(ApproximateGPyTorchModel):
    """Beta 回帰 wrapper の共通基底。"""

    def __init__(
        self,
        *,
        latent_model: ApproximateGP,
        likelihood: BetaLogLikelihood,
        train_X: Tensor,
        train_Y: Tensor,
        input_transform: Optional[InputTransform],
        cat_dims: Optional[Sequence[int]] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
        link: BetaMeanLink = "sigmoid",
        eps: float = 1e-6,
    ) -> None:
        super().__init__(model=latent_model, likelihood=likelihood, num_outputs=1)
        self.input_transform = input_transform
        self.cat_dims = None if cat_dims is None else list(cat_dims)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self.train_targets = prepare_beta_targets(train_Y, train_X, eps=eps, clip=True)
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.eps = float(eps)
        self.to(train_X)

    def _set_transformed_inputs(self) -> None:
        """BoTorch の eval 時自動 transform を無効化する。"""
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """raw-space X を latent model の入力空間へ写像する。"""
        return apply_input_transform_for_eval(X, self.input_transform, cat_dims=self.cat_dims)

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
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
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = True,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> BetaPosterior:
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(f"{self.__class__.__name__} does not support tensor observation_noise.")
        latent_post = self.latent_posterior(X, output_indices=output_indices, posterior_transform=None, **kwargs)
        posterior = BetaPosterior(
            latent_posterior=latent_post,
            likelihood=self.likelihood,
            add_observation_noise=bool(observation_noise),
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def predict_mean(self, X: Tensor) -> Tensor:
        """Beta mean μ の予測値を返す。"""
        return self.posterior(X, observation_noise=True).mean

    def predict_concentration(self) -> Tensor:
        """Beta concentration φ を返す。"""
        return self.likelihood.concentration

    def predict_beta_params(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Beta distribution の alpha, beta parameter を返す。"""
        mu = self.predict_mean(X).clamp(min=self.eps, max=1.0 - self.eps)
        phi = self.predict_concentration().to(device=mu.device, dtype=mu.dtype)
        alpha = (mu * phi).clamp_min(self.eps)
        beta = ((1.0 - mu) * phi).clamp_min(self.eps)
        return alpha, beta

    def make_mll(self) -> VariationalELBO:
        """VariationalELBO を作る。"""
        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.train_inputs_raw[0].shape[-2],
        )


class BetaGPModel(_BaseBetaGPModel):
    """連続入力用 Beta SVGP 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        likelihood: Optional[BetaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(train_X, input_transform, name="BetaGPModel.input_transform")
        likelihood = likelihood or BetaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
        )
        latent_model = _LatentBetaSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
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
            cat_dims=None,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            eps=eps,
        )
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.min_concentration = float(min_concentration)
        self.clip_targets = bool(clip_targets)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "BetaGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("BetaGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_beta_targets(Y, X, eps=self.eps, clip=self.clip_targets)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)
        return self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            link=self.link,
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            eps=self.eps,
            min_concentration=self.min_concentration,
            clip_targets=self.clip_targets,
        )


class BetaMixedGPModel(_BaseBetaGPModel):
    """連続 + カテゴリ mixed 入力用 Beta SVGP 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        likelihood: Optional[BetaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        d = train_X.shape[-1]
        cat_dims = normalize_dims(cat_dims, d)
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be non-empty for BetaMixedGPModel.")
        train_Y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name="BetaMixedGPModel.input_transform",
        )
        check_categorical_columns_unchanged(train_X, train_X_tf, cat_dims=cat_dims)
        likelihood = likelihood or BetaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
        )
        latent_model = _LatentMixedBetaSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            cat_dims=cat_dims,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
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
            cat_dims=cat_dims,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            eps=eps,
        )
        self.cat_dims = list(cat_dims)
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.min_concentration = float(min_concentration)
        self.clip_targets = bool(clip_targets)


__all__ = [
    "BetaLogLikelihood",
    "BetaPosterior",
    "BetaGPModel",
    "BetaMixedGPModel",
    "build_mixed_beta_kernel",
]
