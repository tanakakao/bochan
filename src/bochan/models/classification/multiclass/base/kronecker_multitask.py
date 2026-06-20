from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor
from torch.distributions import Categorical

from bochan.models.components.kronecker_multitask import (
    BlockDesignVariationalELBO,
    canonicalize_block_design_targets,
    canonicalize_shared_inducing_points,
)
from bochan.models.components.multiclass import (
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    clone_input_transform,
    to_device_dtype_transform,
)

from .multioutput import MultiOutputMulticlassProbsPosterior


class _ClassBatchedLMCVariationalStrategy(LMCVariationalStrategy):
    """LMC strategy that also reduces the independent class-batch KL terms."""

    def kl_divergence(self) -> Tensor:
        return super().kl_divergence().sum()


class BlockDesignMulticlassLikelihood(Likelihood):
    """Categorical likelihood for class-batched, task-correlated latent logits.

    The latent distribution has a class batch dimension and a multitask event:
    ``[..., C, q, m]``. This likelihood moves the class dimension to the end and
    evaluates categorical observations shaped ``[..., q, m]``.
    """

    def __init__(self, *, num_classes: int, temperature: float = 1.0) -> None:
        super().__init__()
        if int(num_classes) < 3:
            raise ValueError(f"num_classes must be >= 3, got {num_classes}.")
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)

    def _logits_from_samples(self, function_samples: Tensor) -> Tensor:
        if function_samples.ndim < 3:
            raise ValueError(
                "function_samples must contain class, point, and task dimensions; "
                f"got shape={tuple(function_samples.shape)}."
            )
        if function_samples.shape[-3] != self.num_classes:
            raise ValueError(
                "Expected the class batch dimension at -3 with size "
                f"{self.num_classes}, got shape={tuple(function_samples.shape)}."
            )
        return function_samples.movedim(-3, -1) / self.temperature

    @staticmethod
    def _expand_targets(target: Tensor, reference_shape: torch.Size) -> Tensor:
        target = target.long()
        while target.ndim < len(reference_shape):
            target = target.unsqueeze(0)
        return target.expand(reference_shape)

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Categorical:
        return Categorical(logits=self._logits_from_samples(function_samples))

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultitaskMultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        sample_shape = torch.Size([settings.num_likelihood_samples.value()])
        function_samples = function_dist.rsample(sample_shape)
        logits = self._logits_from_samples(function_samples)
        log_probs = F.log_softmax(logits, dim=-1)
        target = self._expand_targets(observations, log_probs.shape[:-1])
        selected = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        return selected.mean(dim=0)

    def log_marginal(
        self,
        observations: Tensor,
        function_dist: MultitaskMultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        sample_shape = torch.Size([settings.num_likelihood_samples.value()])
        function_samples = function_dist.rsample(sample_shape)
        probabilities = torch.softmax(self._logits_from_samples(function_samples), dim=-1).mean(dim=0)
        target = self._expand_targets(observations, probabilities.shape[:-1])
        selected = probabilities.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        return selected.clamp_min(torch.finfo(selected.dtype).tiny).log()

    def marginal(
        self,
        function_dist: MultitaskMultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Categorical:
        sample_shape = torch.Size([settings.num_likelihood_samples.value()])
        function_samples = function_dist.rsample(sample_shape)
        probabilities = torch.softmax(self._logits_from_samples(function_samples), dim=-1).mean(dim=0)
        return Categorical(probs=probabilities)


class _LatentKroneckerMultiTaskMulticlassGP(ApproximateGP):
    r"""Class-batched latent GP with an ICM/Kronecker prior over tasks.

    Each class logit has ``rank`` latent functions. Within class ``c``, the task
    covariance is

    .. math::

        K_c((x,t),(x',t')) = K_{X,c}(x,x') B_{c,tt'}.

    Classes are conditionally independent at the latent GP level and coupled by
    the categorical softmax likelihood.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        rank: Optional[int] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError(f"train_X must have shape [n, d], got {tuple(train_X.shape)}.")
        if train_Y.ndim != 2:
            raise ValueError(f"train_Y must have shape [n, m], got {tuple(train_Y.shape)}.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must have the same first dimension.")

        num_classes = int(num_classes)
        num_tasks = int(train_Y.shape[-1])
        rank = num_tasks if rank is None else int(rank)
        if rank < 1 or rank > num_tasks:
            raise ValueError(
                f"rank must satisfy 1 <= rank <= num_tasks ({num_tasks}), got {rank}."
            )

        shared_inducing_points = canonicalize_shared_inducing_points(
            train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        latent_inducing_points = shared_inducing_points.view(
            1,
            1,
            *shared_inducing_points.shape,
        ).expand(
            num_classes,
            rank,
            *shared_inducing_points.shape,
        ).clone()

        latent_batch_shape = torch.Size([num_classes, rank])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=shared_inducing_points.shape[-2],
            batch_shape=latent_batch_shape,
        )
        base_variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=latent_inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        variational_strategy = _ClassBatchedLMCVariationalStrategy(
            base_variational_strategy=base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=rank,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        self.mean_module = mean_module or ConstantMean(batch_shape=latent_batch_shape)
        class_kernel_batch_shape = torch.Size([num_classes, 1])
        self.data_covar_module = data_covar_module or ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                batch_shape=class_kernel_batch_shape,
            ),
            batch_shape=class_kernel_batch_shape,
        )
        self.covar_module = self.data_covar_module

        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.rank = rank
        self.num_inducing_points = int(shared_inducing_points.shape[-2])
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self.shared_inducing_points = shared_inducing_points
        self.to(device=train_X.device, dtype=train_X.dtype)

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(
            self.mean_module(X),
            self.data_covar_module(X),
        )

    @property
    def lmc_coefficients(self) -> Tensor:
        """Return class-specific latent-to-task coefficients ``[C, rank, m]``."""
        return self.variational_strategy.lmc_coefficients

    @property
    def task_covar_matrix(self) -> Tensor:
        """Return class-specific task covariance matrices with shape ``[C, m, m]``."""
        coefficients = self.lmc_coefficients
        return coefficients.transpose(-1, -2) @ coefficients

    def get_shared_inducing_points(self) -> Tensor:
        """Return one ``[p, d]`` copy of the current inducing locations."""
        points = self.variational_strategy.base_variational_strategy.inducing_points
        return points[0, 0]


class KroneckerMultiTaskMulticlassProbsPosterior(MultiOutputMulticlassProbsPosterior):
    """Probability posterior for correlated multiclass tasks.

    The latent posterior has class-batch shape ``[..., C]`` and multitask event
    shape ``[q, m]``. Public probability tensors have shape ``[..., q, m, C]``.
    """

    def __init__(
        self,
        latent_posterior: GPyTorchPosterior,
        *,
        num_classes: int,
        output_indices: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
    ) -> None:
        Posterior.__init__(self)
        self.latent_posterior = latent_posterior
        self.num_classes = int(num_classes)
        self.output_indices = None if output_indices is None else [int(i) for i in output_indices]
        self.temperature = float(temperature)

    @property
    def device(self) -> torch.device:
        return self.latent_posterior.mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.latent_posterior.mean.dtype

    def _probability_logits(self, latent: Tensor) -> Tensor:
        if latent.shape[-3] != self.num_classes:
            raise RuntimeError(
                "Expected latent class batch at dimension -3 with size "
                f"{self.num_classes}, got shape={tuple(latent.shape)}."
            )
        logits = latent.movedim(-3, -1) / self.temperature
        if self.output_indices is not None:
            index = torch.as_tensor(self.output_indices, device=logits.device, dtype=torch.long)
            logits = logits.index_select(dim=-2, index=index)
        return logits

    @property
    def logits(self) -> Tensor:
        return self._probability_logits(self.latent_posterior.mean)

    @property
    def mean(self) -> Tensor:
        return torch.softmax(self.logits, dim=-1)

    @property
    def variance(self) -> Tensor:
        probabilities = self.mean
        return probabilities * (1.0 - probabilities)

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size(self.mean.shape[-3:])

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.latent_posterior.base_sample_shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        if base_samples is not None:
            try:
                return self.rsample_from_base_samples(
                    sample_shape=torch.Size(sample_shape),
                    base_samples=base_samples,
                )
            except Exception:
                pass
        latent_samples = self.latent_posterior.rsample(sample_shape=torch.Size(sample_shape))
        return torch.softmax(self._probability_logits(latent_samples), dim=-1)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        try:
            latent_samples = self.latent_posterior.rsample_from_base_samples(
                sample_shape=torch.Size(sample_shape),
                base_samples=base_samples,
            )
        except Exception:
            latent_samples = self.latent_posterior.rsample(sample_shape=torch.Size(sample_shape))
        return torch.softmax(self._probability_logits(latent_samples), dim=-1)


class KroneckerMultiTaskMulticlassClassificationGPModel(ApproximateGPyTorchModel):
    r"""Block-design multiclass model with class-specific ICM task covariance.

    Args:
        train_X: Shared input locations with shape ``[n, d]``.
        train_Y: Integer class labels with shape ``[n, m]``.
        num_classes: Common number of classes used by every task.
        rank: ICM rank for task covariance within each class logit.

    The public probability posterior has shape ``[..., q, m, C]``. All tasks
    must use the same class vocabulary ``0, ..., C - 1``. For tasks with
    different class counts, use independent multiclass models wrapped by
    ``MultiOutputMulticlassClassificationModel`` instead.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        rank: Optional[int] = None,
        likelihood: Optional[BlockDesignMulticlassLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        raw_train_X = torch.as_tensor(train_X).contiguous()
        train_Y = canonicalize_block_design_targets(
            raw_train_X,
            train_Y,
            target_dtype=torch.long,
        )
        if num_classes is None:
            num_classes = int(train_Y.max().item()) + 1
        num_classes = int(num_classes)
        self._validate_multiclass_targets(train_Y, num_classes=num_classes)

        input_transform = to_device_dtype_transform(
            clone_input_transform(input_transform),
            raw_train_X,
        )
        train_X_tf = apply_input_transform_for_training(
            raw_train_X,
            input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )

        raw_inducing_points = canonicalize_shared_inducing_points(
            raw_train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        inducing_points_tf = apply_input_transform_for_training(
            raw_inducing_points,
            input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )

        latent_model = _LatentKroneckerMultiTaskMulticlassGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            num_classes=num_classes,
            rank=rank,
            num_inducing_points=inducing_points_tf.shape[-2],
            inducing_points=inducing_points_tf,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
        )
        likelihood = likelihood or BlockDesignMulticlassLikelihood(
            num_classes=num_classes,
            temperature=temperature,
        )
        if int(likelihood.num_classes) != num_classes:
            raise ValueError(
                "likelihood.num_classes must match num_classes: "
                f"got {likelihood.num_classes} and {num_classes}."
            )

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=train_Y.shape[-1],
        )

        self.input_transform = input_transform
        self.train_inputs = (train_X_tf,)
        self.train_inputs_raw = (raw_train_X,)
        self.transformed_train_inputs = self.train_inputs
        self.train_targets = train_Y
        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets

        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = inducing_points_tf
        self.num_classes = num_classes
        self.num_tasks = int(train_Y.shape[-1])
        self.rank = int(latent_model.rank)
        self.num_inducing_points = int(inducing_points_tf.shape[-2])
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.temperature = float(likelihood.temperature)
        self.to(device=raw_train_X.device, dtype=raw_train_X.dtype)

    @staticmethod
    def _validate_multiclass_targets(train_Y: Tensor, *, num_classes: int) -> None:
        if num_classes < 3:
            raise ValueError(f"num_classes must be >= 3, got {num_classes}.")
        invalid = (train_Y < 0) | (train_Y >= num_classes)
        if bool(invalid.any()):
            values = torch.unique(train_Y[invalid]).detach().cpu().tolist()
            raise ValueError(
                "train_Y must contain class labels in "
                f"[0, {num_classes - 1}], got invalid values {values}."
            )

    def _set_transformed_inputs(self) -> None:
        """Disable BoTorch's automatic re-transform of stored transformed inputs."""
        return None

    @property
    def num_outputs(self) -> int:
        """Return the number of correlated multiclass tasks."""
        return self.num_tasks

    @property
    def num_classes_list(self) -> list[int]:
        return [self.num_classes for _ in range(self.num_tasks)]

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def train_input(self) -> Tensor:
        return self.train_inputs[0]

    @property
    def train_input_raw(self) -> Tensor:
        return self.train_inputs_raw[0]

    @property
    def train_X(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_Y(self) -> Tensor:
        return self.train_targets

    @property
    def task_covar_matrix(self) -> Tensor:
        """Return class-specific task covariances with shape ``[C, m, m]``."""
        return self.model.task_covar_matrix

    def transform_inputs(self, X: Tensor) -> Tensor:
        return apply_input_transform_for_eval(X, self.input_transform)

    def _expand_X_for_latent_batch(self, X: Tensor) -> Tensor:
        if X.ndim <= 2:
            return X
        return X.unsqueeze(-3).unsqueeze(-3)

    def forward(self, X: Tensor) -> MultitaskMultivariateNormal:
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        X_tf = self.transform_inputs(X)
        return self.model(self._expand_X_for_latent_batch(X_tf))

    def _normalize_output_indices(
        self,
        output_indices: Optional[Sequence[int]],
    ) -> list[int]:
        if output_indices is None:
            return list(range(self.num_tasks))
        indices = [int(i) for i in output_indices]
        for index in indices:
            if index < 0 or index >= self.num_tasks:
                raise IndexError(
                    f"output index {index} is outside [0, {self.num_tasks - 1}]."
                )
        return indices

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if output_indices is not None:
            raise NotImplementedError(
                "latent_posterior returns all correlated tasks together; "
                "select tasks on posterior() or class_probs() instead."
            )
        if observation_noise is not False:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support observation_noise."
            )

        self.eval()
        self.model.eval()
        self.likelihood.eval()
        posterior = GPyTorchPosterior(self(X))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> KroneckerMultiTaskMulticlassProbsPosterior:
        if observation_noise is not False:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support observation_noise."
            )
        indices = self._normalize_output_indices(output_indices)
        posterior = KroneckerMultiTaskMulticlassProbsPosterior(
            latent_posterior=self.latent_posterior(X),
            num_classes=self.num_classes,
            output_indices=indices,
            temperature=self.temperature,
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def probability_posterior(self, *args: Any, **kwargs: Any) -> KroneckerMultiTaskMulticlassProbsPosterior:
        """Alias of :meth:`posterior` for multi-output multiclass API consistency."""
        return self.posterior(*args, **kwargs)

    def class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.posterior(X, output_indices=output_indices, **kwargs).mean

    def class_probs_list(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        probabilities = self.class_probs(X, output_indices=output_indices, **kwargs)
        return [probabilities[..., i, :] for i in range(probabilities.shape[-2])]

    def padded_class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return class probabilities; padding is unnecessary because all tasks share C."""
        return self.class_probs(X, output_indices=output_indices, **kwargs)

    def probability_variance(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.posterior(X, output_indices=output_indices, **kwargs).variance

    @torch.no_grad()
    def predict_class(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.class_probs(X, output_indices=output_indices, **kwargs).argmax(dim=-1)

    def expected_utility(
        self,
        X: Tensor,
        utility_values: Optional[Tensor | Sequence[float] | Sequence[Sequence[float]]] = None,
        output_indices: Optional[Sequence[int]] = None,
    ) -> Tensor:
        indices = self._normalize_output_indices(output_indices)
        probabilities = self.class_probs(X, output_indices=indices)
        if utility_values is None:
            utilities = torch.arange(
                self.num_classes,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
        else:
            utilities = torch.as_tensor(
                utility_values,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )

        if utilities.ndim == 1:
            if utilities.shape[0] != self.num_classes:
                raise ValueError(
                    f"utility_values must have {self.num_classes} class values, "
                    f"got {utilities.shape[0]}."
                )
            return (probabilities * utilities).sum(dim=-1)

        if utilities.ndim == 2:
            if utilities.shape[-1] != self.num_classes:
                raise ValueError(
                    "utility_values last dimension must equal num_classes: "
                    f"expected {self.num_classes}, got {utilities.shape[-1]}."
                )
            if utilities.shape[0] == self.num_tasks:
                utilities = utilities[indices]
            elif utilities.shape[0] != len(indices):
                raise ValueError(
                    "utility_values first dimension must equal num_tasks or the "
                    f"number of selected tasks ({len(indices)}), got {utilities.shape[0]}."
                )
            return (probabilities * utilities).sum(dim=-1)

        raise ValueError(
            "utility_values must have shape [C] or [m, C], "
            f"got {tuple(utilities.shape)}."
        )

    def make_mll(
        self,
        *,
        num_data: Optional[int] = None,
        beta: float = 1.0,
    ) -> BlockDesignVariationalELBO:
        if num_data is None:
            num_data = self.train_inputs[0].shape[-2]
        return BlockDesignVariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=int(num_data),
            beta=float(beta),
        )

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        **kwargs: Any,
    ) -> KroneckerMultiTaskMulticlassClassificationGPModel:
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support noise in condition_on_observations."
            )
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2 or X.shape[-1] != self.train_inputs_raw[0].shape[-1]:
            raise ValueError(
                "X must have shape [n_new, d] with the same feature dimension as train_X."
            )
        Y = canonicalize_block_design_targets(X, Y, target_dtype=torch.long)
        if Y.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Y must contain {self.num_tasks} tasks, got {Y.shape[-1]}."
            )
        self._validate_multiclass_targets(Y, num_classes=self.num_classes)

        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=-2)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            num_classes=self.num_classes,
            rank=self.rank,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            temperature=self.temperature,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


__all__ = [
    "BlockDesignMulticlassLikelihood",
    "KroneckerMultiTaskMulticlassClassificationGPModel",
    "KroneckerMultiTaskMulticlassProbsPosterior",
]
