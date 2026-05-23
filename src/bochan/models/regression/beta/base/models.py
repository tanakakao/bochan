from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch
from torch import Tensor

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
from gpytorch.likelihoods import BetaLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


def transform_unit_interval_targets(
    train_Y: Tensor,
    method: str = "smithson_verkuilen",
    eps: float = 1e-4,
) -> Tensor:
    """
    Transform unit-interval targets so that endpoints 0 / 1 move into (0, 1).

    Args:
        train_Y: shape [n] or [n, 1].
        method:
            - "smithson_verkuilen": ((y * (n - 1)) + 0.5) / n
            - "clip": clamp to [eps, 1 - eps]
        eps: used only for method="clip"

    Returns:
        Tensor with same shape as input, mapped into (0, 1).
    """
    y = train_Y.clone()
    if method == "smithson_verkuilen":
        n = y.shape[-2] if y.ndim >= 2 else y.shape[0]
        return ((y * (n - 1)) + 0.5) / n
    if method == "clip":
        return y.clamp(min=eps, max=1.0 - eps)
    raise ValueError("method must be 'smithson_verkuilen' or 'clip'.")


def _validate_beta_targets(
    train_Y: Tensor,
    validate_train_Y: bool = True,
) -> Tensor:
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("train_Y must have shape [n] or [n, 1].")

    if validate_train_Y:
        if (train_Y <= 0).any() or (train_Y >= 1).any():
            raise ValueError(
                "Beta targets must lie strictly inside (0, 1). "
                "If your data include 0 or 1, first apply transform_unit_interval_targets(...)."
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
        raise ValueError("cat_dims must not be empty for BetaMixedGPModel.")

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


class _BetaLatentGP(ApproximateGP):
    """
    Latent GP for Beta regression.

    The observation model is handled by GPyTorch's BetaLikelihood:
        y | f ~ Beta(sigmoid(f) * s, (1 - sigmoid(f)) * s)
    where s > 0 is a learned scale parameter.
    """

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


class _BaseBetaGPModel(ApproximateGPyTorchModel):
    """
    BoTorch-compatible Beta GP base model.

    Design:
        - posterior() returns the latent Gaussian posterior.
        - predict_mean_var() returns predictive moments for observed y in (0, 1).
        - For BO, use the MC objective that transforms latent samples via sigmoid.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: Kernel | None = None,
        likelihood: BetaLikelihood | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
    ) -> None:
        train_Y = _validate_beta_targets(
            train_Y=train_Y,
            validate_train_Y=validate_train_Y,
        )

        if train_X.shape[-2] != train_Y.shape[-2]:
            raise ValueError("train_X and train_Y must have the same number of rows.")

        transformed_X = _apply_input_transform_for_init(train_X, input_transform)

        latent_model = _BetaLatentGP(
            train_X=transformed_X,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )

        super().__init__(
            model=latent_model,
            likelihood=BetaLikelihood() if likelihood is None else likelihood,
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
                "posterior. Use predict_mean_var(), sample_mean_parameter(), or "
                "apply sigmoid in the MC objective."
            )

        self.eval()
        X = self.transform_inputs(X)
        dist = self.model(X)
        posterior = GPyTorchPosterior(distribution=dist)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    @torch.no_grad()
    def sample_mean_parameter(
        self,
        X: Tensor,
        num_mc_samples: int = 512,
    ) -> Tensor:
        """
        Return samples of the Beta conditional mean m = sigmoid(f).

        Returns:
            shape [num_mc_samples, ..., q, 1]
        """
        posterior = self.posterior(X)
        latent_samples = posterior.rsample(sample_shape=torch.Size([num_mc_samples]))
        mean_samples = torch.sigmoid(latent_samples)
        return mean_samples

    @torch.no_grad()
    def predict_mean_var(
        self,
        X: Tensor,
        num_mc_samples: int = 512,
    ) -> tuple[Tensor, Tensor]:
        """
        Return predictive mean and variance of observed y.

        For Beta(alpha=ms, beta=(1-m)s):
            E[Y | f]   = m
            Var[Y | f] = m(1-m)/(s+1)
        and by total variance:
            Var[Y] = E[Var(Y|f)] + Var(E[Y|f]).
        """
        mean_samples = self.sample_mean_parameter(
            X=X,
            num_mc_samples=num_mc_samples,
        )

        scale = self.likelihood.scale.to(mean_samples)
        obs_var_samples = mean_samples * (1.0 - mean_samples) / (scale + 1.0)

        mean_y = mean_samples.mean(dim=0)
        var_y = obs_var_samples.mean(dim=0) + mean_samples.var(dim=0, unbiased=False)
        return mean_y, var_y


class BetaGPModel(_BaseBetaGPModel):
    """
    Continuous-input Beta GP for responses in (0, 1).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: BetaLikelihood | None = None,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
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
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        likelihood: BetaLikelihood | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "likelihood": likelihood,
            **kwargs,
        }


class BetaMixedGPModel(_BaseBetaGPModel):
    """
    Mixed-input (continuous + categorical) Beta GP.

    Uses the same kernel design as MixedSingleTaskGP:
        K_cont_1 + K_cat_1 + K_cont_2 * K_cat_2
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        likelihood: BetaLikelihood | None = None,
        cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
        mean_module: Mean | None = None,
        input_transform: InputTransform | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
        validate_train_Y: bool = True,
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
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        categorical_features: list[int],
        likelihood: BetaLikelihood | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "cat_dims": categorical_features,
            "likelihood": likelihood,
            **kwargs,
        }


def get_beta_mean_mc_objective() -> GenericMCObjective:
    """
    Transform latent samples to the Beta conditional mean m = sigmoid(f).
    """
    return GenericMCObjective(lambda samples, X=None: torch.sigmoid(samples[..., 0]))