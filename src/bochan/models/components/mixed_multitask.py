from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
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
    """Build a mixed kernel for the input after removing the task-id column."""
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


def build_full_input_mixed_kernel(
    *,
    d: int,
    cat_dims: Sequence[int],
    task_feature: int,
    batch_shape: torch.Size = torch.Size(),
) -> Kernel:
    """Build a mixed kernel that receives the full task-feature input tensor.

    This is used by BoTorch's exact ``MultiTaskGP``. Inner kernels select only
    continuous and categorical data columns, while the outer kernel is marked as
    accepting all columns so ``MultiTaskGP`` does not re-index it a second time.
    """
    cat_dims, task_feature = normalize_mixed_task_dims(
        cat_dims,
        task_feature=task_feature,
        d=d,
    )
    kernel = build_mixed_kronecker_kernel(
        d=d,
        cat_dims=cat_dims,
        batch_shape=batch_shape,
    )
    kernel.active_dims = torch.arange(int(d))
    return kernel


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
    "build_full_input_mixed_kernel",
    "build_mixed_task_data_kernel",
    "normalize_mixed_task_dims",
    "normalize_task_feature",
    "remap_dims_without_task_feature",
    "transform_mixed_task_inputs",
    "validate_mixed_task_input_transform",
]
