"""Explicit task-index contracts for material-aware Gaussian models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MaterialExplicitTaskSpec:
    """Describe an explicit task-index input contract for ``f(x, task)`` models.

    Args:
        task_feature: Column index of the task identifier in long-format ``train_X``.
        all_tasks: Optional complete task id set. When omitted, task ids are inferred
            from the observed training rows.
        output_tasks: Optional subset of task ids requested at prediction time.
    """

    task_feature: int = -1
    all_tasks: tuple[int, ...] | None = None
    output_tasks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.task_feature, bool) or not isinstance(self.task_feature, int):
            raise TypeError("task_feature must be an integer column index.")
        if self.all_tasks is not None:
            _validate_task_id_sequence(self.all_tasks, name="all_tasks")
        if self.output_tasks is not None:
            _validate_task_id_sequence(self.output_tasks, name="output_tasks")
            if self.all_tasks is not None and not set(self.output_tasks).issubset(self.all_tasks):
                raise ValueError("output_tasks must be a subset of all_tasks.")

    def as_dict(self) -> dict[str, object]:
        return {
            "task_feature": self.task_feature,
            "all_tasks": list(self.all_tasks) if self.all_tasks is not None else None,
            "output_tasks": list(self.output_tasks) if self.output_tasks is not None else None,
        }


def _validate_task_id_sequence(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integer task ids.")
        if value < 0:
            raise ValueError(f"{name} must contain non-negative task ids.")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate task ids.")
    return tuple(normalized)


def normalize_material_task_feature(task_feature: int, d: int) -> int:
    """Normalize a possibly negative task-feature index for a ``d``-column tensor."""

    if isinstance(task_feature, bool) or not isinstance(task_feature, int):
        raise TypeError("task_feature must be an integer column index.")
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer.")
    normalized = task_feature + d if task_feature < 0 else task_feature
    if normalized < 0 or normalized >= d:
        raise ValueError(f"task_feature {task_feature} is out of bounds for d={d}.")
    return normalized


def validate_explicit_material_task_data(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    *,
    task_feature: int = -1,
    all_tasks: Sequence[int] | None = None,
    model_name: str = "material explicit-task model",
) -> tuple[int, tuple[int, ...]]:
    """Validate long-format explicit-task training data.

    The expected representation is one row per observed ``(x, task)`` pair, with
    the task id stored in one ``train_X`` column and a scalar target in
    ``train_Y``. This is intentionally distinct from wide-output ``[n, m]``
    targets used by correlated or independent multi-output models.

    Returns:
        Normalized task-feature index and sorted observed task ids.
    """

    if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor):
        raise TypeError("train_X and train_Y must be torch.Tensor instances.")
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if train_Y.ndim == 1 or train_Y.ndim == 2 and train_Y.shape[-1] == 1:
        n_y = train_Y.shape[0]
    else:
        raise ValueError(f"{model_name} requires scalar train_Y with shape [n] or [n, 1].")
    if train_X.shape[0] != n_y:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if train_Yvar is not None:
        if not isinstance(train_Yvar, Tensor):
            raise TypeError("train_Yvar must be a torch.Tensor when provided.")
        if train_Yvar.shape != train_Y.shape:
            raise ValueError("train_Yvar must match train_Y shape exactly.")

    normalized_feature = normalize_material_task_feature(task_feature, train_X.shape[-1])
    task_column = train_X[:, normalized_feature]
    if not torch.isfinite(task_column).all():
        raise ValueError("task ids must be finite.")
    rounded = task_column.round()
    if not torch.equal(task_column, rounded):
        raise ValueError("task ids must be integer-valued.")
    if (rounded < 0).any():
        raise ValueError("task ids must be non-negative.")

    observed = tuple(sorted({int(value) for value in rounded.detach().cpu().tolist()}))
    if not observed:
        raise ValueError("at least one explicit task must be observed.")
    if all_tasks is not None:
        declared = _validate_task_id_sequence(all_tasks, name="all_tasks")
        if not set(observed).issubset(declared):
            raise ValueError("observed task ids must be a subset of all_tasks.")
    return normalized_feature, observed


def stack_material_task_observations(
    base_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    *,
    task_values: Sequence[int] | None = None,
    task_feature: int = -1,
    drop_missing: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, int]:
    """Convert wide task observations into long-format explicit-task training data.

    ``base_X`` has shape ``[n, d]`` and ``train_Y`` has shape ``[n, t]``. The
    returned ``long_X`` has shape ``[n*t, d+1]`` before optional missing-value
    filtering, with one integer-valued task-id column.
    """

    if not isinstance(base_X, Tensor) or not isinstance(train_Y, Tensor):
        raise TypeError("base_X and train_Y must be torch.Tensor instances.")
    if base_X.ndim != 2:
        raise ValueError("base_X must have shape [n, d].")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError("train_Y must have shape [n, t] with at least two tasks.")
    if base_X.shape[0] != train_Y.shape[0]:
        raise ValueError("base_X and train_Y must contain the same number of rows.")
    if train_Yvar is not None and train_Yvar.shape != train_Y.shape:
        raise ValueError("train_Yvar must match train_Y shape exactly.")

    n, d = base_X.shape
    num_tasks = train_Y.shape[-1]
    tasks = tuple(range(num_tasks)) if task_values is None else _validate_task_id_sequence(task_values, name="task_values")
    if len(tasks) != num_tasks:
        raise ValueError("task_values length must match train_Y.shape[-1].")

    insert_at = task_feature + (d + 1) if task_feature < 0 else task_feature
    if insert_at < 0 or insert_at > d:
        raise ValueError(f"task_feature {task_feature} is out of bounds for d+1={d + 1}.")

    repeated_X = base_X.repeat_interleave(num_tasks, dim=0)
    task_tensor = torch.tensor(tasks, dtype=base_X.dtype, device=base_X.device).repeat(n).unsqueeze(-1)
    long_X = torch.cat((repeated_X[:, :insert_at], task_tensor, repeated_X[:, insert_at:]), dim=-1)
    long_Y = train_Y.reshape(-1, 1)
    long_Yvar = train_Yvar.reshape(-1, 1) if train_Yvar is not None else None

    if drop_missing:
        mask = torch.isfinite(long_Y.squeeze(-1))
        if long_Yvar is not None:
            mask = mask & torch.isfinite(long_Yvar.squeeze(-1))
        long_X = long_X[mask]
        long_Y = long_Y[mask]
        if long_Yvar is not None:
            long_Yvar = long_Yvar[mask]

    normalized_feature, _ = validate_explicit_material_task_data(
        long_X,
        long_Y,
        long_Yvar,
        task_feature=insert_at,
        all_tasks=tasks,
    )
    return long_X, long_Y, long_Yvar, normalized_feature


def split_material_task_feature(train_X: Tensor, *, task_feature: int = -1) -> tuple[Tensor, Tensor, int]:
    """Split base material inputs and task ids from long-format ``train_X``."""

    if not isinstance(train_X, Tensor):
        raise TypeError("train_X must be a torch.Tensor instance.")
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    normalized = normalize_material_task_feature(task_feature, train_X.shape[-1])
    task_ids = train_X[:, normalized].to(dtype=torch.long)
    base_X = torch.cat((train_X[:, :normalized], train_X[:, normalized + 1 :]), dim=-1)
    return base_X, task_ids, normalized


__all__ = [
    "MaterialExplicitTaskSpec",
    "normalize_material_task_feature",
    "split_material_task_feature",
    "stack_material_task_observations",
    "validate_explicit_material_task_data",
]
