from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Sequence

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import IndexKernel, Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from torch import Tensor

from bochan.models.components.mixed_multitask import (
    MixedTaskProductKernel,
    build_mixed_task_data_kernel,
    normalize_mixed_task_dims,
    transform_mixed_task_inputs,
    validate_mixed_task_input_transform,
)

from .models import _concat_optional_noise, _prepare_binary_conditioning_data
from .multitask import MultiTaskBinaryClassificationGPModel


class MultiTaskBinaryClassificationMixedGPModel(MultiTaskBinaryClassificationGPModel):
    """Long-format mixed-input binary multi-task GP.

    ``train_X`` contains continuous features, categorical features, and one
    explicit integer task-id column. The latent covariance is
    ``K_mixed(x, x') * K_task(t, t')``. Unlike the Kronecker model, tasks may be
    observed at different input locations and observations may be missing.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        *,
        num_tasks: int,
        task_feature: int = -1,
        rank: int = 1,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        task_covar_module: Optional[IndexKernel] = None,
        num_inducing: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        cat_dims, task_feature = normalize_mixed_task_dims(
            cat_dims,
            task_feature=task_feature,
            d=train_X.shape[-1],
        )
        validate_mixed_task_input_transform(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            task_feature=task_feature,
        )
        if data_covar_module is None:
            data_covar_module = build_mixed_task_data_kernel(
                d=train_X.shape[-1],
                cat_dims=cat_dims,
                task_feature=task_feature,
            )

        self.cat_dims = list(cat_dims)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            num_tasks=num_tasks,
            task_feature=task_feature,
            rank=rank,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            task_covar_module=task_covar_module,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.cat_dims = list(cat_dims)
        self.model.covar_module = MixedTaskProductKernel(
            self.model.data_covar_module,
            self.model.task_covar_module,
            task_feature=self.task_feature,
            input_dim=train_X.shape[-1],
        ).to(device=train_X.device, dtype=train_X.dtype)

    @property
    def task_covar_matrix(self) -> Tensor:
        """Return the latent task covariance matrix with shape ``[m, m]``."""
        return self.model.task_covar_module.covar_matrix.to_dense()

    def transform_inputs(self, X: Tensor) -> Tensor:
        X_tf = transform_mixed_task_inputs(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
            task_feature=self.task_feature,
        )
        self._validate_task_feature(
            X_tf,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            name="transformed X",
        )
        return X_tf

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "MultiTaskBinaryClassificationMixedGPModel":
        X_new, Y_new, Yvar_new = _prepare_binary_conditioning_data(X, Y, noise)
        self._validate_task_feature(
            X_new,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            name="new X",
        )

        train_X_old = self.train_inputs_raw[0]
        train_Y_old = self.train_targets
        if train_Y_old.ndim > 1 and train_Y_old.shape[-1] == 1:
            train_Y_old = train_Y_old.squeeze(-1)

        train_X_full = torch.cat(
            [train_X_old, X_new.to(dtype=train_X_old.dtype, device=train_X_old.device)],
            dim=0,
        )
        train_Y_full = torch.cat(
            [train_Y_old, Y_new.to(dtype=train_Y_old.dtype, device=train_Y_old.device)],
            dim=0,
        )
        train_Yvar_full = _concat_optional_noise(
            old_Y=train_Y_old,
            old_Yvar=self.train_Yvar,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        inducing_points = self.model.variational_strategy.inducing_points.detach().clone()
        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            cat_dims=self.cat_dims,
            train_Yvar=train_Yvar_full,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            rank=self.rank,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(self.input_transform),
            mean_module=deepcopy(self.model.mean_module),
            data_covar_module=deepcopy(self.model.data_covar_module),
            task_covar_module=deepcopy(self.model.task_covar_module),
            num_inducing=inducing_points.shape[-2],
            inducing_points=inducing_points,
            learn_inducing_locations=getattr(
                self.model.variational_strategy,
                "learn_inducing_locations",
                True,
            ),
        )
        new_model.load_state_dict(deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


__all__ = ["MultiTaskBinaryClassificationMixedGPModel"]
