from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from torch import Tensor

from bochan.models.components.mixed_multitask import (
    build_full_input_mixed_kernel,
    normalize_mixed_task_dims,
    transform_mixed_task_inputs,
    validate_mixed_task_input_transform,
)


class MixedMultiTaskGP(MultiTaskGP):
    """Exact Gaussian task-feature GP for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: Sequence[int],
        *,
        covar_module: Optional[Kernel] = None,
        input_transform: Optional[InputTransform] = None,
        **kwargs: Any,
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
        if covar_module is None:
            covar_module = build_full_input_mixed_kernel(
                d=train_X.shape[-1],
                cat_dims=cat_dims,
                task_feature=task_feature,
            )

        self.cat_dims = list(cat_dims)
        self.task_feature_index = int(task_feature)
        self._mixed_full_input_dim = int(train_X.shape[-1])
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            covar_module=covar_module,
            input_transform=input_transform,
            **kwargs,
        )
        self.cat_dims = list(cat_dims)
        self.task_feature_index = int(task_feature)

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Optional[InputTransform] = None,
    ) -> Tensor:
        transform = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        if X.shape[-1] == self._mixed_full_input_dim:
            return transform_mixed_task_inputs(
                X,
                transform,
                cat_dims=self.cat_dims,
                task_feature=self.task_feature_index,
            )
        if transform is None:
            return X
        return transform(X)


MultiTaskMixedGP = MixedMultiTaskGP


__all__ = ["MixedMultiTaskGP", "MultiTaskMixedGP"]
