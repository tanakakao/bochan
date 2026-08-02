"""Correlated variational multi-task Beta regression models."""

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
from bochan.models.components.beta import (
    BetaLogLikelihood,
    BetaMeanLink,
    positive_concentration_from_raw,
    prepare_beta_targets,
)

from .beta import (
    clone_input_transform,
    to_device_dtype_transform,
)


class BetaMultiTaskLikelihood(BetaLogLikelihood):
    """Beta likelihood with one learnable concentration per response task."""

    def __init__(
        self,
        num_tasks: int,
        *,
        task_indices: Tensor,
        init_concentration: float | Tensor = 20.0,
        learn_concentration: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize task-specific positive concentration parameters.

        Args:
            num_tasks: Number of response tasks.
            task_indices: Long-form task index for every observed cell.
            init_concentration: Scalar or length-``num_tasks`` initial precision.
            learn_concentration: Whether precision is optimized with the ELBO.
            **kwargs: Arguments forwarded to :class:`BetaLogLikelihood`.
        """
        super().__init__(
            init_concentration=1.0,
            learn_concentration=learn_concentration,
            **kwargs,
        )
        self.register_buffer("task_indices", task_indices.long().clone())
        del self.raw_concentration
        initial = torch.as_tensor(init_concentration, dtype=torch.get_default_dtype()).flatten()
        if initial.numel() == 1:
            initial = initial.expand(num_tasks).clone()
        if initial.numel() != num_tasks or bool((initial <= 0).any()):
            raise ValueError("concentration must be positive and scalar or length num_tasks.")
        raw = torch.log(torch.expm1(initial.clamp_min(self.min_concentration)))
        if learn_concentration:
            self.register_parameter("raw_concentration", torch.nn.Parameter(raw))
        else:
            self.register_buffer("raw_concentration", raw)

    @property
    def concentration(self) -> Tensor:
        """Return task-specific precision with shape ``[num_tasks]``."""
        return positive_concentration_from_raw(
            self.raw_concentration, min_concentration=self.min_concentration
        )

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate quadrature using interleaved long-form task precision."""
        del params, kwargs
        concentration = self.concentration.to(function_dist.mean)
        long_concentration = concentration[self.task_indices]

        def log_prob(function_samples: Tensor) -> Tensor:
            mean = self.mean_from_f(function_samples)
            phi = long_concentration.to(mean)
            return torch.distributions.Beta(
                concentration1=(mean * phi).clamp_min(self.eps),
                concentration0=((1 - mean) * phi).clamp_min(self.eps),
            ).log_prob(observations.to(mean))

        return self.quadrature(log_prob, function_dist)


class BetaMultiTaskPosterior(Posterior):
    """Beta response posterior that preserves the joint task covariance."""

    def __init__(
        self,
        latent_posterior: GPyTorchPosterior,
        likelihood: BetaLogLikelihood,
        *,
        q: int,
        num_tasks: int,
        add_observation_noise: bool = True,
    ) -> None:
        """Initialize a response-scale multi-task posterior.

        Args:
            latent_posterior: Joint latent posterior over interleaved point-task pairs.
            likelihood: Fitted Beta likelihood.
            q: Number of candidate points.
            num_tasks: Number of tasks.
            add_observation_noise: Whether to include Beta observation variance.
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
        """Return positive Beta means with shape ``[..., q, m]``."""
        return self._reshape(self.likelihood.mean_from_f(self.latent_posterior.mean))

    @property
    def variance(self) -> Tensor:
        """Return finite, non-negative marginal response variances."""
        latent = self._reshape(self.latent_posterior.variance).clamp_min(0)
        if not self.add_observation_noise:
            return latent
        concentration = self.likelihood.concentration.to(self.mean)
        return latent + self.mean * (1 - self.mean) / (concentration + 1)

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
        return self._reshape(self.likelihood.mean_from_f(latent))

    def sample_observations(self, sample_shape: torch.Size | None = None) -> Tensor:
        """Draw reparameterized Beta response samples including aleatoric noise."""
        mean = self.rsample(torch.Size() if sample_shape is None else sample_shape)
        concentration = self.likelihood.concentration.to(mean)
        return torch.distributions.Beta(
            concentration1=(mean * concentration).clamp_min(self.likelihood.eps),
            concentration0=((1 - mean) * concentration).clamp_min(self.likelihood.eps),
        ).rsample()

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """Map common-random-number latent samples to Beta response means."""
        latent = self.latent_posterior.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )
        return self._reshape(self.likelihood.mean_from_f(latent))


@GetSampler.register(BetaMultiTaskPosterior)
def _get_beta_multitask_sampler(
    posterior: BetaMultiTaskPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return BoTorch's default normal base sampler for the latent posterior."""
    del posterior
    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


class _LatentBetaMultiTaskSVGP(ApproximateGP):
    """Sparse variational GP using an ICM task covariance on long-form data."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_tasks: int,
        rank: int,
        num_inducing_points: int,
        inducing_points: Tensor | None,
        learn_inducing_locations: bool,
        mean_module: Mean | None,
        data_covar_module: Kernel | None,
        task_covar_module: IndexKernel | None,
    ) -> None:
        """Build the correlated latent process."""
        if inducing_points is None:
            count = min(int(num_inducing_points), train_X.shape[-2])
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
    observed = torch.isfinite(train_Y)
    if not observed.any(dim=0).all():
        missing = (~observed.any(dim=0)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Each Beta task requires an observation; empty tasks: {missing}.")
    rows, tasks = observed.nonzero(as_tuple=True)
    long_X = torch.cat([train_X[rows], tasks.to(train_X).unsqueeze(-1)], dim=-1)
    return long_X, train_Y[rows, tasks], observed


class BetaMultiTaskGPModel(ApproximateGPyTorchModel):
    """Correlated non-Gaussian multi-task GP for positive wide targets.

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
        num_inducing_points: int = 64,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        likelihood: BetaMultiTaskLikelihood | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        task_covar_module: IndexKernel | None = None,
        link: BetaMeanLink = "sigmoid",
        concentration: float | Tensor = 20.0,
        init_concentration: float | Tensor | None = None,
        learn_concentration: bool = True,
        min_concentration: float = 1e-6,
        boundary_policy: str = "error",
        boundary_epsilon: float = 1e-6,
    ) -> None:
        """Initialize a correlated Beta multi-task model.

        Args:
            train_X: Raw inputs with shape ``[n, d]``.
            train_Y: Positive wide targets with shape ``[n, m]``; NaNs are omitted.
            rank: Rank of the learned task covariance.
            num_latents: Reserved LMC-compatible setting; defaults to ``rank``.
            num_inducing_points: Maximum number of long-form inducing points.
            inducing_points: Optional long-form inducing points including task id.
            learn_inducing_locations: Whether inducing locations are trainable.
            likelihood: Optional Beta likelihood.
            input_transform: Raw-space BoTorch input transform.
            mean_module: Optional latent mean.
            covar_module: Optional data covariance.
            task_covar_module: Optional task covariance.
            link: Link from latent logits to the Beta mean.
            concentration: Initial scalar or task-specific Beta precision.
            init_concentration: Deprecated spelling of ``concentration``.
            learn_concentration: Whether task precision is optimized.
            min_concentration: Numerical lower bound for precision.
            boundary_policy: ``error`` (default) or explicit fixed-epsilon ``clip``.
            boundary_epsilon: Clipping distance used only by ``clip``.
        """
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device, dtype=train_X.dtype)
        if train_X.ndim != 2 or train_Y.ndim != 2 or train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must have shapes [n, d] and [n, m].")
        raw_targets = train_Y.detach().clone()
        train_Y = prepare_beta_targets(
            train_Y, train_X, eps=boundary_epsilon,
            boundary_policy=boundary_policy, allow_nan=True,
        )
        self.num_tasks = int(train_Y.shape[-1])
        self.rank = int(rank)
        self.num_latents = int(num_latents if num_latents is not None else rank)
        if self.rank < 1 or self.rank > self.num_tasks:
            raise ValueError("rank must be between 1 and num_tasks.")
        if self.num_latents < 1:
            raise ValueError("num_latents must be positive.")

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
        task_indices = long_X[:, -1].long()
        if init_concentration is not None:
            concentration = init_concentration
        likelihood = likelihood or BetaMultiTaskLikelihood(
            self.num_tasks,
            task_indices=task_indices,
            link=link,
            init_concentration=concentration,
            learn_concentration=learn_concentration,
            eps=boundary_epsilon,
            min_concentration=min_concentration,
        )
        latent_model = _LatentBetaMultiTaskSVGP(
            long_X,
            long_Y,
            num_tasks=self.num_tasks,
            rank=self.rank,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=covar_module,
            task_covar_module=task_covar_module,
        )
        super().__init__(latent_model, likelihood, num_outputs=self.num_tasks)
        self.input_transform = input_transform
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets_raw = raw_targets
        self.train_targets_model = transformed_Y.detach().clone()
        self.train_inputs = (train_X,)
        self.train_targets = transformed_Y
        self.observed_mask = observed
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.learn_concentration = bool(learn_concentration)
        self.min_concentration = float(min_concentration)
        self.boundary_policy = str(boundary_policy)
        self.boundary_epsilon = float(boundary_epsilon)
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
            raise NotImplementedError("Beta multitask output subsetting is not implemented.")
        if torch.is_tensor(observation_noise):
            raise NotImplementedError("Tensor observation_noise is not supported.")
        posterior: Posterior = BetaMultiTaskPosterior(
            self.latent_posterior(X, **kwargs), self.likelihood,
            q=X.shape[-2], num_tasks=self.num_tasks,
            add_observation_noise=bool(observation_noise),
        )
        return posterior if posterior_transform is None else posterior_transform(posterior)

    def mean_posterior(self, X: Tensor, **kwargs: Any) -> Posterior:
        """Return the response-mean posterior without Beta observation noise."""
        return self.posterior(X, observation_noise=False, **kwargs)

    def predictive_posterior(self, X: Tensor, **kwargs: Any) -> Posterior:
        """Return total predictive uncertainty including Beta response variance."""
        return self.posterior(X, observation_noise=True, **kwargs)

    def concentration(self, X: Tensor | None = None) -> Tensor:
        """Return task-specific positive Beta precision values."""
        value = self.likelihood.concentration
        if X is None:
            return value
        return value.to(X).expand(*X.shape[:-2], X.shape[-2], self.num_tasks)

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

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> BetaMultiTaskGPModel:
        """Rebuild the same correlated structure with additional raw observations."""
        if kwargs.get("noise") is not None:
            raise NotImplementedError("Beta multitask conditioning does not accept noise.")
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        Y = torch.as_tensor(Y, device=X.device, dtype=X.dtype)
        return self.__class__(
            torch.cat([self.train_inputs_raw[0], X], dim=-2),
            torch.cat([self.train_targets_raw, Y], dim=-2),
            rank=self.rank,
            num_latents=self.num_latents,
            num_inducing_points=self.num_inducing_points,
            learn_inducing_locations=self.learn_inducing_locations,
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.data_covar_module),
            task_covar_module=copy.deepcopy(self.model.task_covar_module),
            link=self.link,
            concentration=self.likelihood.concentration.detach().clone(),
            learn_concentration=self.learn_concentration,
            min_concentration=self.min_concentration,
            boundary_policy=self.boundary_policy,
            boundary_epsilon=self.boundary_epsilon,
        )

    def fantasize(self, X: Tensor, sampler: Any, **kwargs: Any) -> BetaMultiTaskGPModel:
        """Condition on response-scale fantasy draws using the fitted joint covariance.

        This is a local variational conditioning approximation: variational
        parameters are reused rather than re-optimized for every acquisition call.
        """
        fantasies = sampler(self.posterior(X))
        if fantasies.ndim != 2:
            raise NotImplementedError(
                "Batched Beta multitask fantasies require batched variational training data. "
                "Use posterior samples directly for MC acquisition functions."
            )
        return self.condition_on_observations(X, fantasies, **kwargs)


class WideBetaMultiTaskGPModel(BetaMultiTaskGPModel):
    """Beta multi-task model whose wide targets may contain partial NaNs."""


__all__ = [
    "BetaMultiTaskLikelihood",
    "BetaMultiTaskGPModel",
    "BetaMultiTaskPosterior",
    "WideBetaMultiTaskGPModel",
]
