from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma as TorchGamma

from botorch.acquisition.objective import GenericMCObjective, PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import normalize_indices

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)


LinkFunction = Literal["softplus", "exp"]


def suggest_exp_clip(
    train_y: Tensor,
    *,
    margin: float = 1.2,
    quantile: float | None = 0.98,
    min_clip: float = 2.0,
    max_clip: float = 8.0,
    eps: float = 1e-8,
    return_details: bool = False,
) -> float | dict[str, Any]:
    y = train_y.detach().reshape(-1).to(dtype=torch.float64)

    if y.numel() == 0:
        raise ValueError("train_y is empty.")
    if (y < 0).any():
        raise ValueError("train_y must be non-negative.")

    if quantile is None:
        y_ref = float(y.max().item())
        ref_name = "max"
    else:
        if not (0.0 < quantile <= 1.0):
            raise ValueError("quantile must be in (0, 1].")
        y_ref = float(torch.quantile(y, quantile).item())
        ref_name = f"q{quantile:.3f}"

    target_mean_cap = max(y_ref * margin, eps)
    exp_clip = math.log(target_mean_cap)
    exp_clip = max(min_clip, min(max_clip, exp_clip))+1

    if not return_details:
        return float(exp_clip)

    return {
        "suggested_exp_clip": float(exp_clip),
        "implied_mean_cap": float(math.exp(exp_clip)),
        "reference_value": float(y_ref),
        "reference_name": ref_name,
        "margin": float(margin),
        "y_mean": float(y.mean().item()),
        "y_std": float(y.std(unbiased=False).item()),
        "y_max": float(y.max().item()),
        "y_min": float(y.min().item()),
    }

def _inv_softplus(x: Tensor) -> Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def _positive_link(
    x: Tensor,
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
    min_value: float = 1e-8,
) -> Tensor:
    if link_function == "softplus":
        return F.softplus(x) + min_value
    if link_function == "exp":
        return torch.exp(x.clamp(max=exp_clip)) + min_value
    raise ValueError("link_function must be 'softplus' or 'exp'.")


def _validate_gamma_targets(
    train_Y: Tensor,
    validate_train_Y: bool = True,
) -> Tensor:
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("train_Y must have shape [n] or [n, 1].")

    if validate_train_Y and (train_Y <= 0).any():
        raise ValueError(
            "Gamma targets must be strictly positive. "
            "If zeros exist, consider shifting the target or using another likelihood."
        )

    return train_Y


def _apply_input_transform_for_init(
    train_X: Tensor,
    input_transform: InputTransform | None,
) -> Tensor:
    if input_transform is None:
        return train_X

    input_transform = input_transform.to(train_X)
    was_training = input_transform.training
    input_transform.train()
    with torch.no_grad():
        transformed_X = input_transform(train_X)
    input_transform.train(was_training)
    return transformed_X


def _default_cont_kernel_factory(
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: list[int] | None = None,
) -> Kernel:
    return get_covar_module_with_dim_scaled_prior(
        batch_shape=batch_shape,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
    )


def _make_mixed_covar_module(
    train_X: Tensor,
    cat_dims: list[int],
    cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
) -> tuple[Kernel, list[int], list[int]]:
    if len(cat_dims) == 0:
        raise ValueError("cat_dims must not be empty for GammaMixedGPModel.")

    if cont_kernel_factory is None:
        cont_kernel_factory = _default_cont_kernel_factory

    d = train_X.shape[-1]
    input_batch_shape = train_X.shape[:-2]

    cat_dims = normalize_indices(indices=cat_dims, d=d)
    ord_dims = sorted(set(range(d)) - set(cat_dims))

    if len(ord_dims) == 0:
        covar_module = ScaleKernel(
            CategoricalKernel(
                batch_shape=input_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=GreaterThan(1e-6),
            )
        )
        return covar_module, cat_dims, ord_dims

    sum_kernel = ScaleKernel(
        cont_kernel_factory(
            batch_shape=input_batch_shape,
            ard_num_dims=len(ord_dims),
            active_dims=ord_dims,
        )
        + ScaleKernel(
            CategoricalKernel(
                batch_shape=input_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=GreaterThan(1e-6),
            )
        )
    )

    prod_kernel = ScaleKernel(
        cont_kernel_factory(
            batch_shape=input_batch_shape,
            ard_num_dims=len(ord_dims),
            active_dims=ord_dims,
        )
        * CategoricalKernel(
            batch_shape=input_batch_shape,
            ard_num_dims=len(cat_dims),
            active_dims=cat_dims,
            lengthscale_constraint=GreaterThan(1e-6),
        )
    )
    covar_module = sum_kernel + prod_kernel
    return covar_module, cat_dims, ord_dims


def _make_inducing_points(
    train_X: Tensor,
    inducing_points: Tensor | int | None,
) -> Tensor:
    n = train_X.shape[-2]

    if isinstance(inducing_points, Tensor):
        return inducing_points.to(train_X)

    if inducing_points is None:
        num_inducing = min(max(16, n // 4), n, 128)
    else:
        num_inducing = int(inducing_points)
        if num_inducing <= 0:
            raise ValueError("inducing_points as int must be positive.")
        num_inducing = min(num_inducing, n)

    if num_inducing == n:
        return train_X.clone()

    idx = torch.linspace(
        0,
        n - 1,
        steps=num_inducing,
        device=train_X.device,
        dtype=train_X.dtype,
    ).round().long().unique()
    return train_X.index_select(dim=-2, index=idx)


class GammaLikelihood(_OneDimensionalLikelihood):
    def __init__(
        self,
        concentration: float | Tensor = 10.0,
        learn_concentration: bool = True,
        min_concentration: float = 1e-4,
        min_mean: float = 1e-8,
        link_function: LinkFunction = "softplus",
        exp_clip: float = 8.0,
    ) -> None:
        super().__init__()
        concentration_t = torch.as_tensor(concentration, dtype=torch.get_default_dtype())
        if concentration_t.numel() != 1:
            raise ValueError("concentration must be a scalar.")

        self.min_concentration = float(min_concentration)
        self.min_mean = float(min_mean)
        self.learn_concentration = bool(learn_concentration)
        self.link_function = link_function
        self.exp_clip = float(exp_clip)

        if self.learn_concentration:
            raw = _inv_softplus(concentration_t.clamp_min(self.min_concentration))
            self.raw_concentration = nn.Parameter(raw.reshape(()).clone().detach())
        else:
            self.register_buffer(
                "_fixed_concentration",
                concentration_t.clamp_min(self.min_concentration).reshape(()),
            )
            self.raw_concentration = None

    @property
    def concentration(self) -> Tensor:
        if self.learn_concentration:
            return F.softplus(self.raw_concentration) + self.min_concentration
        return self._fixed_concentration

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> TorchGamma:
        mean = _positive_link(
            function_samples,
            link_function=self.link_function,
            exp_clip=self.exp_clip,
            min_value=self.min_mean,
        )
        concentration = self.concentration.to(mean).expand_as(mean)
        rate = concentration / mean
        return TorchGamma(concentration=concentration, rate=rate)


class _GammaLatentGP(ApproximateGP):
    def __init__(
        self,
        train_X: Tensor,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
    ) -> None:
        if train_X.ndim < 2:
            raise ValueError("train_X must have shape [..., n, d].")

        input_batch_shape = train_X.shape[:-2]
        d = train_X.shape[-1]

        if covar_module is None:
            covar_module = _default_cont_kernel_factory(
                batch_shape=input_batch_shape,
                ard_num_dims=d,
                active_dims=None,
            )

        inducing_points = _make_inducing_points(train_X, inducing_points)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2],
            batch_shape=input_batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )
        super().__init__(variational_strategy=variational_strategy)

        self.mean_module = (
            ConstantMean(batch_shape=input_batch_shape).to(train_X)
            if mean_module is None
            else mean_module.to(train_X)
        )
        self.covar_module = covar_module.to(train_X)

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class _BaseGammaGPModel(ApproximateGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: Kernel | None = None,
        likelihood: GammaLikelihood | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
        link_function: LinkFunction = "softplus",
        exp_clip: float | None = None,
        auto_exp_clip_quantile: float | None = 0.98,
        auto_exp_clip_margin: float = 1.2,
        min_exp_clip: float = 2.0,
        max_exp_clip: float = 8.0,
    ) -> None:
        train_Y = _validate_gamma_targets(train_Y=train_Y, validate_train_Y=validate_train_Y)
        if train_X.shape[-2] != train_Y.shape[-2]:
            raise ValueError("train_X and train_Y must have the same number of rows.")

        transformed_X = _apply_input_transform_for_init(train_X, input_transform)

        latent_model = _GammaLatentGP(
            train_X=transformed_X,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )

        self.link_function = link_function

        if link_function == "exp":
            if exp_clip is None:
                exp_clip_info = suggest_exp_clip(
                    train_Y,
                    quantile=auto_exp_clip_quantile,
                    margin=auto_exp_clip_margin,
                    min_clip=min_exp_clip,
                    max_clip=max_exp_clip,
                    return_details=True,
                )
                self.exp_clip = float(exp_clip_info["suggested_exp_clip"])
                self.exp_clip_info = exp_clip_info
            else:
                self.exp_clip = float(exp_clip)
                self.exp_clip_info = {
                    "suggested_exp_clip": self.exp_clip,
                    "reference_name": "user_specified",
                }
        else:
            self.exp_clip = 8.0
            self.exp_clip_info = {
                "suggested_exp_clip": self.exp_clip,
                "reference_name": "unused_for_softplus",
            }

        if likelihood is None:
            likelihood = GammaLikelihood(
                link_function=link_function,
                exp_clip=self.exp_clip,
            )
        else:
            if hasattr(likelihood, "link_function"):
                link_function = likelihood.link_function
            if hasattr(likelihood, "exp_clip"):
                exp_clip = float(likelihood.exp_clip)
    
        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        if input_transform is not None:
            self.input_transform = input_transform.to(train_X)

        self.train_X_original = train_X
        self.train_inputs = (train_X,)
        self.train_targets = train_Y.squeeze(-1)

        self.model.train_inputs = (transformed_X,)
        self.model.train_inputs_raw = (train_X,)
        self.model.train_inputs_transformed = (transformed_X,)
        self.model.train_targets = train_Y.squeeze(-1).to(dtype=transformed_X.dtype)
        self.to(train_X)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior:
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )

        if observation_noise:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior returns only the latent GP posterior. "
                "Use predict_mean_var(), sample_observations(), or apply the chosen link in the MC objective."
            )

        self.eval()
        X = self.transform_inputs(X)
        dist = self.model(X)
        posterior = GPyTorchPosterior(distribution=dist)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    @torch.no_grad()
    def sample_mean_parameter(self, X: Tensor, num_mc_samples: int = 512) -> Tensor:
        posterior = self.posterior(X)
        latent_samples = posterior.rsample(sample_shape=torch.Size([num_mc_samples]))
        return _positive_link(
            latent_samples,
            link_function=self.link_function,
            exp_clip=self.exp_clip,
            min_value=self.likelihood.min_mean,
        )

    @torch.no_grad()
    def predict_mean_var(self, X: Tensor, num_mc_samples: int = 512) -> tuple[Tensor, Tensor]:
        mean_samples = self.sample_mean_parameter(X=X, num_mc_samples=num_mc_samples)
        alpha = self.likelihood.concentration.to(mean_samples)
        obs_var_samples = mean_samples.square() / alpha
        mean_y = mean_samples.mean(dim=0)
        var_y = obs_var_samples.mean(dim=0) + mean_samples.var(dim=0, unbiased=False)
        return mean_y, var_y

    @torch.no_grad()
    def sample_observations(self, X: Tensor, num_mc_samples: int = 256) -> Tensor:
        mean_samples = self.sample_mean_parameter(X=X, num_mc_samples=num_mc_samples)
        alpha = self.likelihood.concentration.to(mean_samples).expand_as(mean_samples)
        rate = alpha / mean_samples
        dist = TorchGamma(concentration=alpha, rate=rate)
        return dist.sample()


class GammaGPModel(_BaseGammaGPModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: GammaLikelihood | None = None,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
        link_function: LinkFunction = "softplus",
        exp_clip: float | None = None,
    ) -> None:
        if covar_module is None:
            covar_module = _default_cont_kernel_factory(
                batch_shape=train_X.shape[:-2],
                ard_num_dims=train_X.shape[-1],
                active_dims=None,
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            mean_module=mean_module,
            input_transform=input_transform,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
            validate_train_Y=validate_train_Y,
            link_function=link_function,
            exp_clip=exp_clip,
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        likelihood: GammaLikelihood | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "likelihood": likelihood,
            **kwargs,
        }


class GammaMixedGPModel(_BaseGammaGPModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        likelihood: GammaLikelihood | None = None,
        cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
        link_function: LinkFunction = "softplus",
        exp_clip: float | None = None,
    ) -> None:
        covar_module, cat_dims, ord_dims = _make_mixed_covar_module(
            train_X=train_X,
            cat_dims=cat_dims,
            cont_kernel_factory=cont_kernel_factory,
        )

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            mean_module=mean_module,
            input_transform=input_transform,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
            validate_train_Y=validate_train_Y,
            link_function=link_function,
            exp_clip=exp_clip,
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        categorical_features: list[int],
        likelihood: GammaLikelihood | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "cat_dims": categorical_features,
            "likelihood": likelihood,
            **kwargs,
        }


def get_gamma_mean_mc_objective(
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
    min_mean: float = 1e-8,
) -> GenericMCObjective:
    return GenericMCObjective(
        lambda samples, X=None: _positive_link(
            samples[..., 0],
            link_function=link_function,
            exp_clip=exp_clip,
            min_value=min_mean,
        )
    )