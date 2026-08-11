"""Correlated variational multi-task Poisson regression models."""

from __future__ import annotations

import copy
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

from bochan.models.classification.binary.base.multitask import _TaskProductKernel
from bochan.models.regression.count.poisson._components import PoissonLink, PoissonLogLikelihood

from .models import (
    clone_input_transform,
    to_device_dtype_transform,
)


class PoissonMultiTaskPosterior(Posterior):
    """Poisson response posterior that preserves the joint task covariance."""

    def __init__(
        self,
        latent_posterior: GPyTorchPosterior,
        likelihood: PoissonLogLikelihood,
        *,
        q: int,
        num_tasks: int,
        add_observation_noise: bool = True,
    ) -> None:
        """Initialize a response-scale multi-task posterior.

        Args:
            latent_posterior: Joint latent posterior over interleaved point-task pairs.
            likelihood: Fitted Poisson likelihood.
            q: Number of candidate points.
            num_tasks: Number of tasks.
            add_observation_noise: Whether to include Poisson observation variance.
        """
        super().__init__()
        self.latent_posterior = latent_posterior
        self.likelihood = likelihood
        self.q = int(q)
        self.num_tasks = int(num_tasks)
        self.add_observation_noise = bool(add_observation_noise)

    def _reshape(self, value: Tensor) -> Tensor:
        """Reshape an interleaved ``q * task`` event into ``q, task``."""
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value.reshape(*value.shape[:-1], self.q, self.num_tasks)

    @property
    def device(self) -> torch.device:
        """Return the posterior device."""
        return self.latent_posterior.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the posterior dtype."""
        return self.latent_posterior.dtype

    @property
    def event_shape(self) -> torch.Size:
        """Return the public ``q x task`` event shape."""
        return torch.Size((self.q, self.num_tasks))

    @property
    def base_sample_shape(self) -> torch.Size:
        """Return the underlying joint latent base-sample shape."""
        return self.latent_posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        """Return the base-sampler batch range."""
        return self.latent_posterior.batch_range

    @property
    def task_covar(self) -> Tensor:
        """Return the full candidate-task latent covariance matrix."""
        return self.latent_posterior.distribution.covariance_matrix

    @property
    def mean(self) -> Tensor:
        """Return positive Poisson means with shape ``[..., q, m]``."""
        return self._reshape(self.likelihood.rate_from_f(self.latent_posterior.mean))

    @property
    def variance(self) -> Tensor:
        """Return finite, non-negative marginal response variances."""
        latent = self._reshape(self.latent_posterior.variance).clamp_min(0)
        rate_variance = self.mean.square() * latent
        return rate_variance + self.mean if self.add_observation_noise else rate_variance

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
        base_samples: Tensor | None = None,
    ) -> Tensor:
        """Draw differentiable positive samples while retaining task correlation."""
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        if base_samples is None:
            latent = self.latent_posterior.rsample(sample_shape=sample_shape)
        else:
            latent = self.latent_posterior.rsample_from_base_samples(
                sample_shape=sample_shape,
                base_samples=base_samples,
            )
        return self._reshape(self.likelihood.rate_from_f(latent))

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """Map common-random-number latent samples to Poisson response means."""
        latent = self.latent_posterior.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )
        return self._reshape(self.likelihood.rate_from_f(latent))


@GetSampler.register(PoissonMultiTaskPosterior)
def _get_poisson_multitask_sampler(
    posterior: PoissonMultiTaskPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return BoTorch's default normal base sampler for the latent posterior."""
    del posterior
    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


class _LatentPoissonMultiTaskSVGP(ApproximateGP):
    """Sparse variational GP using an ICM task covariance on long-form data."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_tasks: int,
        rank: int,
        num_inducing: int,
        inducing_points: Tensor | None,
        learn_inducing_locations: bool,
        mean_module: Mean | None,
        data_covar_module: Kernel | None,
        task_covar_module: IndexKernel | None,
    ) -> None:
        """Build the correlated latent process."""
        if inducing_points is None:
            count = min(int(num_inducing), train_X.shape[-2])
            indices = torch.linspace(0, train_X.shape[-2] - 1, count, device=train_X.device).long()
            inducing_points = train_X.index_select(-2, indices).clone()
        variational_distribution = CholeskyVariationalDistribution(inducing_points.shape[-2])
        strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(strategy)
        data_kernel = data_covar_module or ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1] - 1)
        )
        task_kernel = task_covar_module or IndexKernel(num_tasks=num_tasks, rank=rank)
        self.mean_module = mean_module or ConstantMean()
        self.data_covar_module = data_kernel
        self.task_covar_module = task_kernel
        self.covar_module = _TaskProductKernel(
            data_kernel=data_kernel,
            task_kernel=task_kernel,
            task_feature=train_X.shape[-1] - 1,
            input_dim=train_X.shape[-1],
        )
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Evaluate the joint latent distribution."""
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


def _wide_to_long(train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert observed wide cells to long form without imputing missing values."""
    observed = ~torch.isnan(train_Y)
    if not observed.any(dim=0).all():
        missing = (~observed.any(dim=0)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Each Poisson task requires an observation; empty tasks: {missing}.")
    rows, tasks = observed.nonzero(as_tuple=True)
    long_X = torch.cat([train_X[rows], tasks.to(train_X).unsqueeze(-1)], dim=-1)
    return long_X, train_Y[rows, tasks], observed


class _WidePoissonMultiTaskCore(ApproximateGPyTorchModel):
    """Correlated non-Gaussian multi-task GP for wide count targets.

    The model learns an ICM covariance ``K_x * K_task`` in a single variational
    GP. It is not an independent ``ModelList`` and is not an exact Gaussian
    multi-task GP.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        rank: int = 1,
        num_latents: int | None = None,
        num_inducing: int = 64,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        likelihood: PoissonLogLikelihood | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: Any | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        task_covar_module: IndexKernel | None = None,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        """Initialize a correlated Poisson multi-task model.

        Args:
            train_X: Raw inputs with shape ``[n, d]``.
            train_Y: Positive wide targets with shape ``[n, m]``; NaNs are omitted.
            rank: Rank of the learned task covariance.
            num_latents: Reserved LMC-compatible setting; defaults to ``rank``.
            num_inducing_points: Maximum number of long-form inducing points.
            inducing_points: Optional long-form inducing points including task id.
            learn_inducing_locations: Whether inducing locations are trainable.
            likelihood: Optional Poisson likelihood.
            input_transform: Raw-space BoTorch input transform.
            outcome_transform: Unsupported; must be ``None`` for raw counts.
            mean_module: Optional latent mean.
            covar_module: Optional data covariance.
            task_covar_module: Optional task covariance.
            link: Positive Poisson mean link.
            exp_clip: Maximum latent log-rate used by the exponential link.
            min_rate: Numerical lower bound for rates.
        """
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device, dtype=train_X.dtype)
        if train_X.ndim != 2 or train_Y.ndim != 2 or train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must have shapes [n, d] and [n, m].")
        finite = ~torch.isnan(train_Y)
        observed_values = train_Y[finite]
        if not torch.isfinite(observed_values).all():
            raise ValueError("Poisson targets must be finite or NaN for missing cells.")
        if bool((observed_values < 0).any()):
            raise ValueError("Poisson targets must be non-negative.")
        if not torch.isclose(
            observed_values, observed_values.round(), atol=1e-6, rtol=0.0
        ).all():
            raise ValueError("Poisson targets must be integer counts.")
        if outcome_transform is not None:
            raise ValueError(
                "Poisson models require raw counts; outcome_transform is not supported."
            )
        self.num_tasks = int(train_Y.shape[-1])
        self.rank = int(rank)
        self.num_latents = int(num_latents if num_latents is not None else rank)
        if self.rank < 1 or self.rank > self.num_tasks:
            raise ValueError("rank must be between 1 and num_tasks.")
        if self.num_latents < 1 or self.num_latents > self.num_tasks:
            raise ValueError("num_latents must be between 1 and num_tasks.")

        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        if input_transform is not None:
            input_transform.train()
            transformed_X = input_transform(train_X)
            input_transform.eval()
        else:
            transformed_X = train_X
        if transformed_X.shape[-2] != train_X.shape[-2]:
            raise ValueError("Training input_transform must preserve the row axis.")
        transformed_Y = train_Y.clone()

        long_X, long_Y, observed = _wide_to_long(transformed_X, transformed_Y)
        likelihood = likelihood or PoissonLogLikelihood(
            link=link, exp_clip=exp_clip, min_rate=min_rate
        )
        latent_model = _LatentPoissonMultiTaskSVGP(
            long_X,
            long_Y,
            num_tasks=self.num_tasks,
            rank=self.rank,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=covar_module,
            task_covar_module=task_covar_module,
        )
        super().__init__(latent_model, likelihood, num_outputs=self.num_tasks)
        self.input_transform = input_transform
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets_raw = train_Y.detach().clone()
        self.train_inputs = (train_X,)
        self.train_targets = transformed_Y
        self.observed_mask = observed
        self.num_inducing = int(num_inducing)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_rate = float(min_rate)
        self.to(train_X)

    def transform_inputs(self, X: Tensor) -> Tensor:
        """Apply the raw-space input transform without reducing one-to-many axes."""
        return X if self.input_transform is None else self.input_transform(X)

    def _task_grid(self, X: Tensor) -> Tensor:
        """Expand candidates into interleaved point-task long form."""
        X = self.transform_inputs(X)
        tasks = torch.arange(self.num_tasks, device=X.device, dtype=X.dtype)
        expanded = X.unsqueeze(-2).expand(*X.shape[:-1], self.num_tasks, X.shape[-1])
        task_col = tasks.expand(*X.shape[:-2], X.shape[-2], self.num_tasks).unsqueeze(-1)
        return torch.cat([expanded, task_col], dim=-1).reshape(*X.shape[:-2], -1, X.shape[-1] + 1)

    def latent_posterior(self, X: Tensor, **kwargs: Any) -> GPyTorchPosterior:
        """Return the joint latent posterior over candidate-task pairs."""
        del kwargs
        self.eval()
        return GPyTorchPosterior(self.model(self._task_grid(X)))

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = True,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> Posterior:
        """Return a positive response posterior with shape ``[..., q, m]``."""
        if output_indices is not None:
            raise NotImplementedError("Poisson multitask output subsetting is not implemented.")
        if torch.is_tensor(observation_noise):
            raise NotImplementedError("Tensor observation_noise is not supported.")
        posterior: Posterior = PoissonMultiTaskPosterior(
            self.latent_posterior(X, **kwargs), self.likelihood,
            q=X.shape[-2], num_tasks=self.num_tasks,
            add_observation_noise=bool(observation_noise),
        )
        return posterior if posterior_transform is None else posterior_transform(posterior)

    def rate_posterior(self, X: Tensor, **kwargs: Any) -> PoissonMultiTaskPosterior:
        """Return the differentiable rate posterior with shape ``[..., q, m]``.

        Args:
            X: Raw candidate inputs.
            **kwargs: Arguments forwarded to :meth:`posterior`.

        Returns:
            The joint task-correlated rate posterior, excluding conditional
            Poisson observation variance.
        """
        return self.posterior(X, observation_noise=False, **kwargs)  # type: ignore[return-value]

    def sample_observations(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        """Draw non-reparameterized count samples for prediction or diagnostics.

        Args:
            X: Raw candidate inputs.
            sample_shape: Leading sample dimensions.

        Returns:
            Integer-valued count samples with trailing ``q, task`` axes.
        """
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        rates = self.rate_posterior(X).rsample(sample_shape)
        return torch.distributions.Poisson(rates).sample()

    @property
    def task_covar_module(self) -> IndexKernel:
        """Expose the learned correlated task covariance module."""
        return self.model.task_covar_module

    @property
    def task_covar_matrix(self) -> Tensor:
        """Return the finite learned task covariance matrix."""
        return self.task_covar_module.covar_matrix.to_dense()

    def make_mll(self, **kwargs: Any) -> VariationalELBO:
        """Build the non-Gaussian variational training objective."""
        return VariationalELBO(
            self.likelihood, self.model, int(self.observed_mask.sum()), **kwargs
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> PoissonMultiTaskGPModel:
        """Rebuild the same correlated structure with additional raw observations."""
        if kwargs.get("noise") is not None:
            raise NotImplementedError("Poisson multitask conditioning does not accept noise.")
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        Y = torch.as_tensor(Y, device=X.device, dtype=X.dtype)
        return self.__class__(
            torch.cat([self.train_inputs_raw[0], X], dim=-2),
            torch.cat([self.train_targets_raw, Y], dim=-2),
            rank=self.rank,
            num_latents=self.num_latents,
            num_inducing=self.num_inducing,
            learn_inducing_locations=self.learn_inducing_locations,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.data_covar_module),
            task_covar_module=copy.deepcopy(self.model.task_covar_module),
            link=self.link,
            exp_clip=self.exp_clip,
            min_rate=self.min_rate,
        )

    def fantasize(self, X: Tensor, sampler: Any, **kwargs: Any) -> PoissonMultiTaskGPModel:
        """Condition on response-scale fantasy draws using the fitted joint covariance.

        This is a local variational conditioning approximation: variational
        parameters are reused rather than re-optimized for every acquisition call.
        """
        fantasies = sampler(self.posterior(X))
        if fantasies.ndim != 2:
            raise NotImplementedError(
                "Batched Poisson multitask fantasies require batched variational training data. "
                "Use posterior samples directly for MC acquisition functions."
            )
        return self.condition_on_observations(X, fantasies, **kwargs)


class WidePoissonMultiTaskGPModel(_WidePoissonMultiTaskCore):
    """Correlated wide Poisson model that omits, rather than imputes, NaN cells."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        """Initialize from a shared input design and wide targets.

        Args:
            train_X: Shared inputs with shape ``[n, d]``.
            train_Y: Wide targets with shape ``[n, m]``; partial NaNs are allowed.
            **kwargs: Family-specific variational model options.
        """
        super().__init__(train_X, train_Y, **kwargs)
        self.raw_train_X = self.train_inputs_raw[0]
        self.raw_train_Y = self.train_targets_raw


class PoissonMultiTaskGPModel(_WidePoissonMultiTaskCore):
    """Correlated non-Gaussian ICM model for task-feature long data.

    This is a sparse variational analogue of BoTorch's long-form ``MultiTaskGP``
    contract, not the exact Gaussian model itself.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        task_feature: int = -1,
        num_tasks: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize from task-feature long observations.

        Args:
            train_X: Inputs ``[n_observations, d + 1]`` including task ids.
            train_Y: Targets ``[n_observations]`` or ``[n_observations, 1]``.
            task_feature: Column containing zero-based integer task ids.
            num_tasks: Task count; inferred from ids when omitted.
            **kwargs: Family-specific variational model options.
        """
        from bochan.models.regression._multitask import (
            long_to_sparse_wide,
            validate_long_multitask_data,
        )

        long_X, long_Y, feature, count = validate_long_multitask_data(
            train_X, train_Y, task_feature=task_feature, num_tasks=num_tasks
        )
        data_X, wide_Y = long_to_sparse_wide(
            long_X, long_Y, task_feature=feature, num_tasks=count
        )
        super().__init__(data_X, wide_Y, **kwargs)
        self.task_feature = feature
        self.long_train_X = long_X.detach().clone()
        self.long_train_Y = long_Y.detach().clone()
        self.raw_train_X = self.long_train_X
        self.raw_train_Y = self.long_train_Y

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "PoissonMultiTaskGPModel":
        """Return a rebuilt long-form model including new observations.

        Args:
            X: New long-form rows including the task feature.
            Y: New scalar observations.
            **kwargs: Reserved conditioning options.

        Returns:
            A model trained on the concatenated observations.
        """
        if kwargs.get("noise") is not None:
            raise NotImplementedError("Non-Gaussian multitask conditioning does not accept noise.")
        X = torch.as_tensor(X, device=self.long_train_X.device, dtype=self.long_train_X.dtype)
        Y = torch.as_tensor(Y, device=X.device, dtype=X.dtype).reshape(-1)
        return self.__class__(
            torch.cat((self.long_train_X, X), dim=-2),
            torch.cat((self.long_train_Y, Y), dim=-1),
            task_feature=self.task_feature,
            num_tasks=self.num_tasks,
            rank=self.rank,
            num_latents=self.num_latents,
            num_inducing_points=self.num_inducing_points,
            learn_inducing_locations=self.learn_inducing_locations,
        )


class KroneckerMultiTaskPoissonGPModel(WidePoissonMultiTaskGPModel):
    """Separable variational Poisson GP for a complete block design.

    The latent covariance is ``K_x ⊗ K_task`` through the ICM product kernel.
    Inference is sparse variational and therefore is not BoTorch's exact Gaussian
    ``KroneckerMultiTaskGP``.
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        """Initialize the structured approximation from complete wide data.

        Args:
            train_X: Complete shared design with shape ``[n, d]``.
            train_Y: Finite targets with shape ``[n, m]``.
            **kwargs: Family-specific variational model options.
        """
        from bochan.models.regression._multitask import validate_complete_block

        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device, dtype=train_X.dtype)
        validate_complete_block(train_X, train_Y, family="Poisson")
        super().__init__(train_X, train_Y, **kwargs)
        self.is_kronecker_variational_approximation = True
        self.input_covar_module = self.model.data_covar_module


__all__ = [
    "PoissonMultiTaskGPModel",
    "PoissonMultiTaskPosterior",
    "WidePoissonMultiTaskGPModel",
    "KroneckerMultiTaskPoissonGPModel",
]
