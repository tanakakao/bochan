from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import IndexKernel, Kernel
from torch import Tensor

from .mixed_kronecker import (
    build_mixed_kronecker_kernel,
    normalize_mixed_dims,
    transform_mixed_inputs,
    validate_mixed_input_transform_for_training,
)


def normalize_task_feature(task_feature: int, d: int) -> int:
    """Normalize a possibly negative task-feature index."""
    index = int(task_feature)
    if index < 0:
        index += int(d)
    if index < 0 or index >= int(d):
        raise ValueError(f"task_feature={task_feature} is out of bounds for input dim {d}.")
    return index


def normalize_mixed_task_dims(
    cat_dims: Sequence[int],
    *,
    task_feature: int,
    d: int,
) -> tuple[list[int], int]:
    """Normalize category and task indices and reject overlap."""
    task_feature = normalize_task_feature(task_feature, d)
    cat_dims = normalize_mixed_dims(cat_dims, d)
    if task_feature in cat_dims:
        raise ValueError("task_feature must not also be included in cat_dims.")
    return cat_dims, task_feature


def remap_dims_without_task_feature(
    dims: Sequence[int],
    *,
    task_feature: int,
    d: int,
) -> list[int]:
    """Map full-input dimensions to the data-only tensor used by task kernels."""
    task_feature = normalize_task_feature(task_feature, d)
    normalized = []
    for raw_index in dims:
        index = int(raw_index)
        if index < 0:
            index += int(d)
        if index == task_feature:
            raise ValueError("task_feature cannot be remapped as a data feature.")
        if index < 0 or index >= int(d):
            raise ValueError(f"Invalid feature dim {raw_index} for input dim {d}.")
        normalized.append(index - 1 if index > task_feature else index)
    return sorted(set(normalized))


def build_mixed_task_data_kernel(
    *,
    d: int,
    cat_dims: Sequence[int],
    task_feature: int,
    batch_shape: torch.Size = torch.Size(),
) -> Kernel:
    """Build a mixed kernel after removing the explicit task-id column."""
    cat_dims, task_feature = normalize_mixed_task_dims(
        cat_dims,
        task_feature=task_feature,
        d=d,
    )
    data_cat_dims = remap_dims_without_task_feature(
        cat_dims,
        task_feature=task_feature,
        d=d,
    )
    return build_mixed_kronecker_kernel(
        d=int(d) - 1,
        cat_dims=data_cat_dims,
        batch_shape=batch_shape,
    )


class MixedTaskProductKernel(Kernel):
    """Product of a mixed data kernel and an ``IndexKernel`` over task ids.

    The task covariance is materialized for non-diagonal cross-covariances. This
    avoids root decompositions on non-square lazy operators during variational
    posterior evaluation with optimizer t-batches.
    """

    has_lengthscale = False

    def __init__(
        self,
        data_kernel: Kernel,
        task_kernel: IndexKernel,
        *,
        task_feature: int,
        input_dim: int,
    ) -> None:
        super().__init__()
        if int(input_dim) < 2:
            raise ValueError(
                "Multi-task models require at least one data feature and one task feature."
            )
        self.data_kernel = data_kernel
        self.task_kernel = task_kernel
        self.task_feature = normalize_task_feature(task_feature, input_dim)
        self.input_dim = int(input_dim)
        self.data_dims = [
            index for index in range(self.input_dim) if index != self.task_feature
        ]

    def _split(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dim {self.input_dim}, got {X.shape[-1]}."
            )
        data = X[..., self.data_dims]
        task = X[..., self.task_feature].round().long().unsqueeze(-1)
        return data, task

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        x1_data, task_1 = self._split(x1)
        x2_data, task_2 = self._split(x2)
        data_covar = self.data_kernel(
            x1_data,
            x2_data,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )
        task_covar = self.task_kernel(task_1, task_2, diag=diag, **params)
        if diag:
            return data_covar * task_covar
        return data_covar.mul(task_covar.to_dense())


def validate_mixed_task_input_transform(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Sequence[int],
    task_feature: int,
) -> None:
    """Validate that neither categories nor task ids are transformed."""
    cat_dims, task_feature = normalize_mixed_task_dims(
        cat_dims,
        task_feature=task_feature,
        d=X.shape[-1],
    )
    validate_mixed_input_transform_for_training(
        X,
        input_transform,
        cat_dims=[*cat_dims, task_feature],
    )


def transform_mixed_task_inputs(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Sequence[int],
    task_feature: int,
) -> Tensor:
    """Apply a transform while preserving category and task-id columns."""
    cat_dims, task_feature = normalize_mixed_task_dims(
        cat_dims,
        task_feature=task_feature,
        d=X.shape[-1],
    )
    return transform_mixed_inputs(
        X,
        input_transform,
        cat_dims=[*cat_dims, task_feature],
    )


__all__ = [
    "MixedTaskProductKernel",
    "build_mixed_task_data_kernel",
    "normalize_mixed_task_dims",
    "normalize_task_feature",
    "remap_dims_without_task_feature",
    "transform_mixed_task_inputs",
    "validate_mixed_task_input_transform",
]
