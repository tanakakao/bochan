from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import NegativeBinomial as TorchNegativeBinomial

from botorch.acquisition.objective import GenericMCObjective, PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import normalize_indices

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import NegativeBinomialLikelihood
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)


LinkFunction = Literal["softplus", "exp"]


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


def _validate_negative_binomial_targets(
    train_Y: Tensor,
    validate_train_Y: bool = True,
) -> Tensor:
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("train_Y must have shape [n] or [n, 1].")

    if validate_train_Y:
        if (train_Y < 0).any():
            raise ValueError("Negative binomial targets must be non-negative.")
        if not torch.allclose(train_Y, train_Y.round()):
            raise ValueError("Negative binomial targets must be integer-valued counts.")

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
        raise ValueError("cat_dims must not be empty for NegativeBinomialMixedGPModel.")

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


class NegativeBinomialExpLikelihood(_OneDimensionalLikelihood):
    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        probs_constraint: Interval | None = None,
        num_failures_param: bool = False,
        exp_clip: float = 8.0,
        min_value: float = 1e-8,
    ) -> None:
        super().__init__()

        if probs_constraint is None:
            probs_constraint = Interval(0.0, 1.0)

        self.raw_probs = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.register_constraint("raw_probs", probs_constraint)

        self.num_failures_param = bool(num_failures_param)
        self.exp_clip = float(exp_clip)
        self.min_value = float(min_value)

    @property
    def probs(self) -> Tensor:
        return self.raw_probs_constraint.transform(self.raw_probs)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_probs)
        self.initialize(raw_probs=self.raw_probs_constraint.inverse_transform(value))

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> TorchNegativeBinomial:
        probs = torch.clamp(self.probs.to(function_samples), 1e-6, 1 - 1e-6)
        positive = _positive_link(
            function_samples,
            link_function="exp",
            exp_clip=self.exp_clip,
            min_value=self.min_value,
        )

        if self.num_failures_param:
            total_count = positive
        else:
            total_count = positive * (1 - probs) / probs

        return TorchNegativeBinomial(total_count=total_count, probs=probs)



def _mean_from_latent_samples(
    latent_samples: Tensor,
    likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood,
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
) -> Tensor:
    probs = torch.clamp(likelihood.probs.to(latent_samples), 1e-6, 1 - 1e-6)
    positive = _positive_link(
        latent_samples,
        link_function=link_function,
        exp_clip=exp_clip,
        min_value=1e-8,
    )

    if likelihood.num_failures_param:
        mean_samples = positive * probs / (1 - probs)
    else:
        mean_samples = positive
    return mean_samples


class _NegativeBinomialLatentGP(ApproximateGP):
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


class _BaseNegativeBinomialGPModel(ApproximateGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: Kernel | None = None,
        likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
        link_function: LinkFunction = "softplus",
        exp_clip: float = 8.0,
    ) -> None:
        train_Y = _validate_negative_binomial_targets(
            train_Y=train_Y,
            validate_train_Y=validate_train_Y,
        )
        if train_X.shape[-2] != train_Y.shape[-2]:
            raise ValueError("train_X and train_Y must have the same number of rows.")

        transformed_X = _apply_input_transform_for_init(train_X, input_transform)

        latent_model = _NegativeBinomialLatentGP(
            train_X=transformed_X,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )

        self.link_function = link_function
        self.exp_clip = float(exp_clip)
        if likelihood is None:
            if link_function == "softplus":
                likelihood = NegativeBinomialLikelihood()
            elif link_function == "exp":
                likelihood = NegativeBinomialExpLikelihood(
                    exp_clip=self.exp_clip,
                    num_failures_param=False,
                )
            else:
                raise ValueError("link_function must be 'softplus' or 'exp'.")

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
                f"{self.__class__.__name__}.posterior returns only the latent GP "
                "posterior. Use predict_mean(), predict_count_moments(), or "
                "apply get_negative_binomial_mean_mc_objective(...) in the MC objective."
            )

        self.eval()
        X = self.transform_inputs(X)
        dist = self.model(X)
        posterior = GPyTorchPosterior(distribution=dist)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    @torch.no_grad()
    def sample_mean(
        self,
        X: Tensor,
        num_mc_samples: int = 512,
    ) -> Tensor:
        posterior = self.posterior(X)
        latent_samples = posterior.rsample(sample_shape=torch.Size([num_mc_samples]))
        mean_samples = _mean_from_latent_samples(
            latent_samples,
            self.likelihood,
            link_function=self.link_function,
            exp_clip=self.exp_clip,
        )
        return mean_samples

    @torch.no_grad()
    def predict_mean(
        self,
        X: Tensor,
        num_mc_samples: int = 512,
    ) -> tuple[Tensor, Tensor]:
        mean_samples = self.sample_mean(X=X, num_mc_samples=num_mc_samples)
        mean_mean = mean_samples.mean(dim=0)
        var_mean = mean_samples.var(dim=0, unbiased=False)
        return mean_mean, var_mean

    @torch.no_grad()
    def predict_count_moments(
        self,
        X: Tensor,
        num_mc_samples: int = 512,
    ) -> tuple[Tensor, Tensor]:
        mean_samples = self.sample_mean(X=X, num_mc_samples=num_mc_samples)
        probs = torch.clamp(self.likelihood.probs.to(mean_samples), 1e-6, 1 - 1e-6)

        mean_count = mean_samples.mean(dim=0)
        cond_var_mean = (mean_samples / (1 - probs)).mean(dim=0)
        total_var = cond_var_mean + mean_samples.var(dim=0, unbiased=False)
        return mean_count, total_var


class NegativeBinomialGPModel(_BaseNegativeBinomialGPModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood | None = None,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
        link_function: LinkFunction = "softplus",
        exp_clip: float = 8.0,
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
        likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "likelihood": likelihood,
            **kwargs,
        }


class NegativeBinomialMixedGPModel(_BaseNegativeBinomialGPModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood | None = None,
        cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
        link_function: LinkFunction = "softplus",
        exp_clip: float = 8.0,
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
        likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "cat_dims": categorical_features,
            "likelihood": likelihood,
            **kwargs,
        }



def get_negative_binomial_mean_mc_objective(
    likelihood: NegativeBinomialLikelihood | NegativeBinomialExpLikelihood | None = None,
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
) -> GenericMCObjective:
    if likelihood is None:
        return GenericMCObjective(
            lambda samples, X=None: _positive_link(
                samples[..., 0],
                link_function=link_function,
                exp_clip=exp_clip,
                min_value=1e-8,
            )
        )

    def _obj(samples: Tensor, X: Tensor | None = None) -> Tensor:
        probs = torch.clamp(likelihood.probs.to(samples), 1e-6, 1 - 1e-6)
        positive = _positive_link(
            samples[..., 0],
            link_function=link_function,
            exp_clip=exp_clip,
            min_value=1e-8,
        )
        if likelihood.num_failures_param:
            return positive * probs / (1 - probs)
        return positive

    return GenericMCObjective(_obj)


__all__ = [
    "LinkFunction",
    "NegativeBinomialExpLikelihood",
    "NegativeBinomialGPModel",
    "NegativeBinomialMixedGPModel",
    "get_negative_binomial_mean_mc_objective",
]
