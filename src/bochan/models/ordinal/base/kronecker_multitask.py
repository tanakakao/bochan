from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from torch import Tensor

from bochan.fit.ordinal import fit_ordinal_gp
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.components.kronecker_multitask import (
    BlockDesignVariationalELBO,
    LatentKroneckerMultiTaskGP,
    canonicalize_block_design_targets,
    canonicalize_shared_inducing_points,
)

from .models import (
    _BaseOrdinalGPModel,
    _infer_num_classes_from_train_Y,
    _prepare_input_transform,
    _transform_tensor_for_training,
)


class _KroneckerOrdinalTaskView:
    """Single-task read-only view of a joint Kronecker ordinal model.

    Existing ordinal multi-output acquisition functions were originally written
    for :class:`MultiOutputOrdinalModel` and therefore iterate over
    ``model.models``. This adapter exposes each task of the joint Kronecker
    posterior through the same minimal interface without copying parameters or
    discarding the parent model during posterior evaluation.
    """

    def __init__(
        self,
        parent: KroneckerMultiTaskOrdinalGPModel,
        task_index: int,
    ) -> None:
        self.parent = parent
        self.task_index = int(task_index)
        self.num_outputs = 1
        self.cat_dims: list[int] = []

    @property
    def likelihood(self) -> OrdinalLogitLikelihood:
        return self.parent.likelihood

    @property
    def ordinal_likelihood(self) -> OrdinalLogitLikelihood:
        return self.parent.ordinal_likelihood

    @property
    def input_transform(self) -> InputTransform | None:
        return self.parent.input_transform

    @property
    def train_X(self) -> Tensor:
        return self.parent.train_inputs_raw[0]

    @property
    def train_input_raw(self) -> Tensor:
        return self.parent.train_inputs_raw[0]

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        return self.parent.train_inputs_raw

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return self.parent.train_inputs

    @property
    def train_targets(self) -> Tensor:
        return self.parent.train_targets[..., self.task_index]

    @property
    def train_Y(self) -> Tensor:
        return self.train_targets

    @property
    def batch_shape(self) -> torch.Size:
        return self.parent.batch_shape

    def eval(self) -> _KroneckerOrdinalTaskView:
        self.parent.eval()
        self.parent.likelihood.eval()
        return self

    def train(self, mode: bool = True) -> _KroneckerOrdinalTaskView:
        self.parent.train(mode)
        self.parent.likelihood.train(mode)
        return self

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """Return the selected task marginal while preserving data covariance."""
        if output_indices is not None:
            raise NotImplementedError("A Kronecker ordinal task view is single-output.")
        if observation_noise is not False:
            raise NotImplementedError("Ordinal task views do not support observation_noise.")

        joint_posterior = self.parent.posterior(X)
        task_distribution = joint_posterior.distribution[..., :, self.task_index]
        posterior = GPyTorchPosterior(task_distribution)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def class_probs(self, X: Tensor) -> Tensor:
        return self.parent.class_probs(X, output_indices=[self.task_index]).squeeze(-2)

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)

    def condition_on_observations(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "A single Kronecker task cannot be conditioned independently. "
            "Condition the parent model with block-design Y shaped [n_new, m]."
        )


class KroneckerMultiTaskOrdinalGPModel(_BaseOrdinalGPModel):
    r"""Block-design ordinal model with an ICM/Kronecker latent GP.

    All ordinal tasks are observed at the same input locations:

    - ``train_X`` has shape ``[n, d]``.
    - ``train_Y`` has shape ``[n, m]`` and contains labels in
      ``0, ..., num_classes - 1``.

    A correlated latent score vector follows an ICM prior ``K_X ⊗ K_task``.
    :class:`~bochan.likelihoods.ordinal.OrdinalLogitLikelihood` maps each task's
    latent score to ordered class probabilities using shared cutpoints.

    Notes:
        This model shares one ordered-logit cutpoint set across tasks. That is
        appropriate when the tasks use the same ordinal scale. Tasks requiring
        different class counts or different cutpoints should use independent
        ordinal models instead.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        num_classes: int | None = None,
        *,
        rank: int | None = None,
        likelihood: OrdinalLogitLikelihood | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Mean | None = None,
        data_covar_module: Kernel | None = None,
        num_inducing_points: int = 128,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        conditioning_steps: int = 50,
        conditioning_lr: float | None = None,
        conditioning_batch_size: int | None = None,
    ) -> None:
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = canonicalize_block_design_targets(
            raw_train_X,
            train_Y,
            target_dtype=torch.long,
        )

        if num_classes is None:
            num_classes = _infer_num_classes_from_train_Y(train_Y.reshape(-1))
        else:
            num_classes = int(num_classes)
            self._validate_ordinal_targets(train_Y, num_classes=num_classes)

        input_transform = _prepare_input_transform(input_transform, raw_train_X)
        train_X_tf = _transform_tensor_for_training(
            raw_train_X,
            input_transform=input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )

        raw_inducing_points = canonicalize_shared_inducing_points(
            raw_train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        inducing_points_tf = _transform_tensor_for_training(
            raw_inducing_points,
            input_transform=input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )

        latent_model = LatentKroneckerMultiTaskGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            rank=rank,
            num_inducing_points=inducing_points_tf.shape[-2],
            inducing_points=inducing_points_tf,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
        )
        likelihood = likelihood or OrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        )
        if int(likelihood.num_classes) != num_classes:
            raise ValueError(
                "likelihood.num_classes must match num_classes: "
                f"got {likelihood.num_classes} and {num_classes}."
            )

        ApproximateGPyTorchModel.__init__(
            self,
            model=latent_model,
            likelihood=likelihood,
            num_outputs=train_Y.shape[-1],
        )

        self.input_transform = input_transform
        self.train_inputs = (train_X_tf,)
        self.train_inputs_raw = (raw_train_X,)
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

        self.eps = float(likelihood.eps)
        self.init_gap = float(init_gap)
        self.fix_first_cutpoint = bool(likelihood.fix_first_cutpoint)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.to(device=raw_train_X.device, dtype=raw_train_X.dtype)

    @staticmethod
    def _validate_ordinal_targets(train_Y: Tensor, *, num_classes: int) -> None:
        if num_classes < 3:
            raise ValueError(f"num_classes must be >= 3, got {num_classes}.")
        invalid = (train_Y < 0) | (train_Y >= num_classes)
        if bool(invalid.any()):
            values = torch.unique(train_Y[invalid]).detach().cpu().tolist()
            raise ValueError(
                "train_Y must contain ordinal labels in "
                f"[0, {num_classes - 1}], got invalid values {values}."
            )

    @property
    def num_outputs(self) -> int:
        return self.num_tasks

    @property
    def models(self) -> tuple[_KroneckerOrdinalTaskView, ...]:
        """Return task views supported with independent multi-output acquisitions."""
        return tuple(
            _KroneckerOrdinalTaskView(self, task_index)
            for task_index in range(self.num_tasks)
        )

    @property
    def task_covar_matrix(self) -> Tensor:
        """Return the learned positive-semidefinite latent task covariance."""
        return self.model.task_covar_matrix

    def _normalize_output_indices(
        self,
        output_indices: Sequence[int] | None,
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

    def posterior(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """Return the correlated latent ordinal posterior ``[..., q, m]``."""
        if output_indices is not None:
            raise NotImplementedError(
                "posterior currently returns all correlated tasks together; "
                "select tasks on class_probs or other pointwise outputs instead."
            )
        if observation_noise is not False:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support observation_noise. "
                "posterior() returns the latent score posterior."
            )

        self.eval()
        self.model.eval()
        self.likelihood.eval()
        posterior = GPyTorchPosterior(distribution=self(X))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def latent_posterior(self, *args, **kwargs) -> GPyTorchPosterior:
        """Alias of :meth:`posterior` for API consistency."""
        return self.posterior(*args, **kwargs)

    def class_probs_from_posterior(
        self,
        posterior: GPyTorchPosterior,
        output_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Return ``[..., q, m, K]`` class probabilities from a latent posterior."""
        probs = self.ordinal_likelihood.marginal_class_probs(posterior.distribution)
        indices = self._normalize_output_indices(output_indices)
        return probs[..., indices, :]

    def class_probs(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
    ) -> Tensor:
        return self.class_probs_from_posterior(
            self.posterior(X),
            output_indices=output_indices,
        )

    def class_probs_list(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
    ) -> list[Tensor]:
        """Return one ``[..., q, K]`` probability tensor per selected task."""
        probs = self.class_probs(X, output_indices=output_indices)
        return [probs[..., i, :] for i in range(probs.shape[-2])]

    @torch.no_grad()
    def predict_class(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
    ) -> Tensor:
        return self.class_probs(X, output_indices=output_indices).argmax(dim=-1)

    def expected_utility(
        self,
        X: Tensor,
        utilities: Tensor,
        output_indices: Sequence[int] | None = None,
    ) -> Tensor:
        utilities = torch.as_tensor(
            utilities,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        if utilities.shape != torch.Size([self.num_classes]):
            raise ValueError(
                "utilities must have one value per ordinal class: "
                f"expected [{self.num_classes}], got {tuple(utilities.shape)}."
            )
        values = self.ordinal_likelihood.marginal_expected_utility(
            self.posterior(X).distribution,
            utilities,
        )
        indices = self._normalize_output_indices(output_indices)
        return values[..., indices]

    def make_mll(
        self,
        *,
        num_data: int | None = None,
        beta: float = 1.0,
    ) -> BlockDesignVariationalELBO:
        """Build the recommended variational ELBO for this model."""
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
        refit: bool = True,
        num_steps: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> KroneckerMultiTaskOrdinalGPModel:
        """Rebuild the variational model after appending block-design observations."""
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                f"noise is not supported for {self.__class__.__name__}."
            )

        X = self._canonicalize_observation_X(X)
        Y = canonicalize_block_design_targets(X, Y, target_dtype=torch.long)
        if Y.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Y must contain {self.num_tasks} tasks, got {Y.shape[-1]}."
            )
        self._validate_ordinal_targets(Y, num_classes=self.num_classes)

        train_X_full = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        train_Y_full = torch.cat([self.train_targets, Y], dim=-2)

        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            num_classes=self.num_classes,
            rank=self.rank,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            eps=self.eps,
            init_gap=self.init_gap,
            fix_first_cutpoint=self.fix_first_cutpoint,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            refit_batch_size = (
                self.conditioning_batch_size if batch_size is None else batch_size
            )
            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_batch_size,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()
        return new_model
