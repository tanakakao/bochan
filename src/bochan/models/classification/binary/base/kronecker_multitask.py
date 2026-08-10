from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from torch import Tensor

from bochan.models.components.kronecker_multitask import (
    BlockDesignVariationalELBO,
    LatentKroneckerMultiTaskGP,
    canonicalize_block_design_targets,
    canonicalize_shared_inducing_points,
)

from .models import _to_device_dtype_transform
from .multioutput import MultiOutputBernoulliPosterior


class KroneckerMultiTaskBinaryClassificationGPModel(
    ApproximateGPyTorchModel,
    FantasizeMixin,
):
    r"""Block-design binary classifier with an ICM/Kronecker latent GP.

    All tasks are observed at the same input locations:

    - ``train_X`` has shape ``[n, d]``.
    - ``train_Y`` has shape ``[n, m]`` and contains labels in ``{0, 1}``.

    The latent function uses an ICM prior ``K_X ⊗ K_task``. A probit
    :class:`~gpytorch.likelihoods.BernoulliLikelihood` maps each latent task
    output to ``P(y_t = 1)``.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        rank: int | None = None,
        likelihood: BernoulliLikelihood | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Mean | None = None,
        data_covar_module: Kernel | None = None,
        num_inducing: int = 128,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        raw_train_X = torch.as_tensor(train_X).contiguous()
        train_Y = canonicalize_block_design_targets(
            raw_train_X,
            train_Y,
            target_dtype=raw_train_X.dtype,
        )
        self._validate_binary_targets(train_Y)
        train_Yvar = self._canonicalize_train_Yvar(
            train_Yvar,
            train_X=raw_train_X,
            train_Y=train_Y,
        )

        input_transform = _to_device_dtype_transform(input_transform, raw_train_X)
        train_X_tf = self._transform_for_training(
            raw_train_X,
            input_transform=input_transform,
        )

        raw_inducing_points = canonicalize_shared_inducing_points(
            raw_train_X,
            num_inducing_points=num_inducing,
            inducing_points=inducing_points,
        )
        inducing_points_tf = self._transform_for_training(
            raw_inducing_points,
            input_transform=input_transform,
        )

        latent_model = LatentKroneckerMultiTaskGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            rank=rank,
            num_inducing=inducing_points_tf.shape[-2],
            inducing_points=inducing_points_tf,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
        )
        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=train_Y.shape[-1],
        )

        self.input_transform = input_transform
        self.train_inputs = (train_X_tf,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y
        self.train_Yvar = train_Yvar
        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets

        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = inducing_points_tf
        self.num_tasks = int(train_Y.shape[-1])
        self.rank = int(latent_model.rank)
        self.num_inducing = int(inducing_points_tf.shape[-2])
        self.learn_inducing_locations = bool(learn_inducing_locations)

        self.to(device=raw_train_X.device, dtype=raw_train_X.dtype)

    @staticmethod
    def _validate_binary_targets(train_Y: Tensor) -> None:
        invalid = (train_Y != 0) & (train_Y != 1)
        if bool(invalid.any()):
            values = torch.unique(train_Y[invalid]).detach().cpu().tolist()
            raise ValueError(
                "train_Y must contain binary labels encoded as 0 or 1; "
                f"got invalid values {values}."
            )

    @staticmethod
    def _canonicalize_train_Yvar(
        train_Yvar: Tensor | None,
        *,
        train_X: Tensor,
        train_Y: Tensor,
    ) -> Tensor | None:
        if train_Yvar is None:
            return None
        train_Yvar = torch.as_tensor(
            train_Yvar,
            device=train_X.device,
            dtype=train_X.dtype,
        )
        if train_Yvar.shape != train_Y.shape:
            raise ValueError(
                "train_Yvar must have the same block-design shape as train_Y: "
                f"expected {tuple(train_Y.shape)}, got {tuple(train_Yvar.shape)}."
            )
        return train_Yvar.clamp_min(0.0).contiguous()

    @staticmethod
    def _transform_for_training(
        X: Tensor,
        *,
        input_transform: InputTransform | None,
    ) -> Tensor:
        if input_transform is None:
            return X.detach().clone().contiguous()

        input_transform.train()
        with torch.no_grad():
            X_tf = input_transform(X).detach().clone().contiguous()
        input_transform.eval()

        if X_tf.shape[-2] != X.shape[-2]:
            raise RuntimeError(
                "input_transform expanded the training inputs, which is insupported "
                "with block-design train_Y. Configure perturbation transforms with "
                "transform_on_train=False."
            )
        return X_tf

    def _set_transformed_inputs(self) -> None:
        """Disable BoTorch's automatic re-transform of stored training inputs."""
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        if self.input_transform is None:
            return X
        return self.input_transform(X)

    def forward(self, X: Tensor):
        if isinstance(X, tuple):
            X = X[0]
        return self.model(self.transform_inputs(X))

    @property
    def num_outputs(self) -> int:
        return self.num_tasks

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

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
    ) -> MultiOutputBernoulliPosterior:
        """Return the Bernoulli probability posterior ``[..., q, m]``."""
        self.eval()
        self.model.eval()
        self.likelihood.eval()

        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        X_tf = self.transform_inputs(X)
        latent_dist = self.model(X_tf)
        pred_dist = self.likelihood(latent_dist)

        indices = self._normalize_output_indices(output_indices)
        probs = pred_dist.mean[..., indices]
        variance = pred_dist.variance[..., indices]

        noise = self._resolve_observation_noise(
            observation_noise,
            X=X,
            output_indices=indices,
        )
        if noise is not None:
            variance = variance + noise

        posterior = MultiOutputBernoulliPosterior(mean=probs, variance=variance)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def probability_posterior(self, *args, **kwargs) -> MultiOutputBernoulliPosterior:
        """Alias of :meth:`posterior` for multi-output API consistency."""
        return self.posterior(*args, **kwargs)

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """Return the correlated latent multi-task Gaussian posterior."""
        if output_indices is not None:
            raise NotImplementedError(
                "latent_posterior currently returns all correlated tasks together; "
                "output_indices is not supported."
            )
        if observation_noise is not False:
            raise NotImplementedError(
                "observation_noise is defined on the probability posterior, not the latent GP posterior."
            )

        self.eval()
        self.model.eval()
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        posterior = GPyTorchPosterior(self.model(self.transform_inputs(X)))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def _resolve_observation_noise(
        self,
        observation_noise: bool | Tensor,
        *,
        X: Tensor,
        output_indices: Sequence[int],
    ) -> Tensor | None:
        selected_outputs = len(output_indices)
        expected_prefix = X.shape[:-1]

        if torch.is_tensor(observation_noise):
            noise = observation_noise.to(device=X.device, dtype=X.dtype)
            if noise.shape == expected_prefix:
                noise = noise.unsqueeze(-1)
            if noise.shape[:-1] != expected_prefix:
                raise ValueError(
                    "observation_noise must match X.shape[:-1] with an optional output dimension."
                )
            if noise.shape[-1] == 1:
                return noise.expand(*expected_prefix, selected_outputs)
            if noise.shape[-1] == self.num_tasks:
                return noise[..., list(output_indices)]
            if noise.shape[-1] == selected_outputs:
                return noise
            raise ValueError(
                "observation_noise last dimension must be 1, num_tasks, or the number "
                f"of selected outputs; got {noise.shape[-1]}."
            )

        if observation_noise and self.train_Yvar is not None:
            mean_noise = self.train_Yvar.mean(dim=0)[list(output_indices)]
            return mean_noise.expand(*expected_prefix, selected_outputs)
        return None

    def class_probs(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Return ``[..., q, m, 2]`` probabilities ordered as class 0 and 1."""
        p1 = self.posterior(X, output_indices=output_indices).mean
        return torch.stack([1.0 - p1, p1], dim=-1)

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
        if utilities.shape != torch.Size([2]):
            raise ValueError(
                f"binary utilities must have shape [2], got {tuple(utilities.shape)}."
            )
        probs = self.class_probs(X, output_indices=output_indices)
        return (probs * utilities).sum(dim=-1)

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
        noise: Tensor | None = None,
        **kwargs: Any,
    ) -> KroneckerMultiTaskBinaryClassificationGPModel:
        """Rebuild the variational model after appending block-design observations."""
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        if X.ndim != 2 or X.shape[-1] != self.train_inputs_raw[0].shape[-1]:
            raise ValueError(
                "X must have shape [n_new, d] with the same feature dimension as train_X."
            )
        Y = canonicalize_block_design_targets(X, Y, target_dtype=X.dtype)
        if Y.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Y must contain {self.num_tasks} tasks, got {Y.shape[-1]}."
            )
        self._validate_binary_targets(Y)
        Yvar = self._canonicalize_train_Yvar(noise, train_X=X, train_Y=Y)

        train_X_full = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        train_Y_full = torch.cat([self.train_targets, Y], dim=-2)

        train_Yvar_full = None
        if self.train_Yvar is not None or Yvar is not None:
            old_noise = self.train_Yvar
            if old_noise is None:
                old_noise = torch.zeros_like(self.train_targets)
            if Yvar is None:
                Yvar = torch.zeros_like(Y)
            train_Yvar_full = torch.cat([old_noise, Yvar], dim=-2)

        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            train_Yvar=train_Yvar_full,
            rank=self.rank,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            num_inducing=self.num_inducing,
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model
