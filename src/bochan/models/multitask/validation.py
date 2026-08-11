"""Shared validation and reshaping for non-Gaussian multi-task regression."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_long_multitask_data(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    task_feature: int,
    num_tasks: int | None,
) -> tuple[Tensor, Tensor, int, int]:
    """Validate task-feature long data.

    Args:
        train_X: Long inputs of shape ``[n, d + 1]``.
        train_Y: Scalar observations of shape ``[n]`` or ``[n, 1]``.
        task_feature: Column containing zero-based task identifiers.
        num_tasks: Declared task count, or ``None`` to infer it.

    Returns:
        Validated inputs, flattened targets, normalized task column, and task count.

    Raises:
        ValueError: If shapes, finite values, task identifiers, or coverage are invalid.
    """
    X = torch.as_tensor(train_X)
    Y = torch.as_tensor(train_Y, device=X.device, dtype=X.dtype)
    if X.ndim != 2 or Y.ndim not in (1, 2) or (Y.ndim == 2 and Y.shape[-1] != 1):
        raise ValueError("Long multitask data require train_X=[n,d+1] and train_Y=[n] or [n,1].")
    Y = Y.reshape(-1)
    if X.shape[0] != Y.shape[0] or not torch.isfinite(X).all() or not torch.isfinite(Y).all():
        raise ValueError("Long multitask train_X and train_Y must be aligned and contain no NaN/Inf.")
    feature = task_feature % X.shape[-1]
    ids = X[:, feature]
    if not torch.allclose(ids, ids.round()):
        raise ValueError("Task identifiers must be integer-valued.")
    ids_long = ids.long()
    inferred = int(ids_long.max().item()) + 1 if ids_long.numel() else 0
    count = inferred if num_tasks is None else int(num_tasks)
    if count < 1 or bool((ids_long < 0).any()) or bool((ids_long >= count).any()):
        raise ValueError("Task identifiers must be in [0, num_tasks - 1].")
    present = torch.bincount(ids_long, minlength=count) > 0
    if not present.all():
        missing = (~present).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Each task requires at least one observation; missing tasks: {missing}.")
    return X, Y, feature, count


def long_to_sparse_wide(
    train_X: Tensor, train_Y: Tensor, *, task_feature: int, num_tasks: int
) -> tuple[Tensor, Tensor]:
    """Convert observed long rows to a sparse wide table without imputation.

    Duplicate observations for the same input/task pair are rejected because a
    wide cell cannot represent replicates.

    Args:
        train_X: Validated long inputs.
        train_Y: Flattened scalar observations.
        task_feature: Normalized task-feature column.
        num_tasks: Number of tasks.

    Returns:
        Unique data inputs and a NaN-masked wide target matrix.
    """
    task_ids = train_X[:, task_feature].long()
    data_X = torch.cat((train_X[:, :task_feature], train_X[:, task_feature + 1 :]), dim=-1)
    unique_X, inverse = torch.unique(data_X, dim=0, return_inverse=True)
    wide = train_Y.new_full((unique_X.shape[0], num_tasks), torch.nan)
    for row, task, value in zip(inverse.tolist(), task_ids.tolist(), train_Y, strict=True):
        if torch.isfinite(wide[row, task]):
            raise ValueError("Duplicate input/task observations cannot be represented unambiguously.")
        wide[row, task] = value
    return unique_X, wide


def validate_complete_block(train_X: Tensor, train_Y: Tensor, *, family: str) -> None:
    """Validate a complete wide block design for a Kronecker approximation.

    Args:
        train_X: Shared input design of shape ``[n, d]``.
        train_Y: Complete targets of shape ``[n, m]``.
        family: Family name used in diagnostics.
    """
    if train_X.ndim != 2 or train_Y.ndim != 2 or train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("Kronecker models require train_X=[n,d] and train_Y=[n,m].")
    if not torch.isfinite(train_X).all() or not torch.isfinite(train_Y).all():
        raise ValueError(
            f"{family} Kronecker models require a complete block design without NaN; "
            f"use {family.lower()}_wide_multitask for partially observed targets."
        )
