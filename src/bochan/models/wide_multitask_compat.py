"""Compatibility adapters for wide-format multi-task models.

The public API receives ``X=[n, d]`` and ``Y=[n, m]``. The underlying task-feature
models operate on long inputs with an appended task-id column. This module keeps
public input transforms in the original ``d``-dimensional space, preserves task
ids, and exposes the task count as the public output count.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import StratifiedStandardize

from .wide_multitask import (
    WideMultiTaskBinaryClassificationGPModel as _WideMultiTaskBinaryClassificationGPModel,
)
from .wide_multitask import WideMultiTaskGP as _WideMultiTaskGP
from .wide_multitask import (
    WideMultiTaskMulticlassClassificationGPModel as _WideMultiTaskMulticlassClassificationGPModel,
)
from .wide_multitask import WideMultiTaskOrdinalGPModel as _WideMultiTaskOrdinalGPModel
from .wide_multitask import wide_to_long


def _align_task_feature(task_feature: Tensor, transformed: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast and repeat task ids to match transformed public inputs."""

    batch_shape = torch.broadcast_shapes(
        task_feature.shape[:-2],
        transformed.shape[:-2],
    )
    task_feature = task_feature.expand(
        *batch_shape,
        task_feature.shape[-2],
        task_feature.shape[-1],
    )
    transformed = transformed.expand(
        *batch_shape,
        transformed.shape[-2],
        transformed.shape[-1],
    )

    q_raw = int(task_feature.shape[-2])
    q_transformed = int(transformed.shape[-2])
    if q_transformed == q_raw:
        return task_feature, transformed
    if q_raw > 0 and q_transformed % q_raw == 0:
        return (
            task_feature.repeat_interleave(q_transformed // q_raw, dim=-2),
            transformed,
        )
    raise RuntimeError(
        "Could not align task ids with transformed inputs: "
        f"task_feature.shape={tuple(task_feature.shape)}, "
        f"transformed.shape={tuple(transformed.shape)}."
    )


class TaskFeatureInputTransform(InputTransform):
    """Apply a public-space transform while preserving the appended task id."""

    def __init__(self, base_transform: InputTransform, *, data_dim: int) -> None:
        super().__init__()
        self.base_transform = base_transform
        self.data_dim = int(data_dim)
        self.transform_on_train = bool(
            getattr(base_transform, "transform_on_train", True)
        )
        self.transform_on_eval = bool(
            getattr(base_transform, "transform_on_eval", True)
        )
        self.transform_on_fantasize = bool(
            getattr(base_transform, "transform_on_fantasize", True)
        )
        self.is_one_to_many = bool(
            getattr(base_transform, "is_one_to_many", False)
        )

    def _split(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if X.shape[-1] != self.data_dim + 1:
            raise RuntimeError(
                "Expected public data columns followed by one task-id column. "
                f"Expected d={self.data_dim + 1}, got {X.shape[-1]}."
            )
        return X[..., : self.data_dim], X[..., self.data_dim :]

    def transform(self, X: Tensor) -> Tensor:
        data, task_feature = self._split(X)
        transformed = self.base_transform(data)
        task_feature, transformed = _align_task_feature(
            task_feature,
            transformed,
        )
        return torch.cat([transformed, task_feature], dim=-1)

    def untransform(self, X: Tensor) -> Tensor:
        data, task_feature = self._split(X)
        untransformed = self.base_transform.untransform(data)
        task_feature, untransformed = _align_task_feature(
            task_feature,
            untransformed,
        )
        return torch.cat([untransformed, task_feature], dim=-1)


def _prepare_kwargs(
    train_X: Tensor,
    train_Y: Tensor,
    kwargs: dict[str, Any],
    *,
    regression: bool,
) -> dict[str, Any]:
    """Adapt public transforms before the original wide adapter builds long data."""

    prepared = dict(kwargs)
    input_transform = prepared.get("input_transform")
    if input_transform is not None and not isinstance(
        input_transform,
        TaskFeatureInputTransform,
    ):
        prepared["input_transform"] = TaskFeatureInputTransform(
            input_transform,
            data_dim=int(train_X.shape[-1]),
        )

    if regression:
        outcome_transform = prepared.get("outcome_transform")
        if (
            outcome_transform is not None
            and outcome_transform.__class__.__name__
            == "AutoStandardizeOutcomeTransform"
        ):
            num_tasks = int(train_Y.shape[-1])
            prepared["outcome_transform"] = StratifiedStandardize(
                stratification_idx=-1,
                all_task_values=torch.arange(
                    num_tasks,
                    device=train_X.device,
                    dtype=train_X.dtype,
                ),
                dtype=train_X.dtype,
            )
    return prepared


class _PublicTaskOutputMixin:
    """Expose task count rather than latent class count as ``num_outputs``."""

    @property
    def num_outputs(self) -> int:
        return int(getattr(self, "num_tasks", 1))


class WideMultiTaskGP(_PublicTaskOutputMixin, _WideMultiTaskGP):
    """Wide regression MultiTaskGP with per-task outcome standardization."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            **_prepare_kwargs(
                train_X,
                train_Y,
                kwargs,
                regression=True,
            ),
        )


class WideMultiTaskBinaryClassificationGPModel(
    _PublicTaskOutputMixin,
    _WideMultiTaskBinaryClassificationGPModel,
):
    """Wide binary multi-task model with task-safe input transforms."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            **_prepare_kwargs(train_X, train_Y, kwargs, regression=False),
        )


class WideMultiTaskOrdinalGPModel(
    _PublicTaskOutputMixin,
    _WideMultiTaskOrdinalGPModel,
):
    """Wide ordinal multi-task model with task-safe input transforms."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            **_prepare_kwargs(train_X, train_Y, kwargs, regression=False),
        )


class WideMultiTaskMulticlassClassificationGPModel(
    _PublicTaskOutputMixin,
    _WideMultiTaskMulticlassClassificationGPModel,
):
    """Wide multiclass multi-task model exposing an explicit task output axis.

    Class ids remain categorical labels and are not standardized as continuous
    outcomes. All tasks share the configured class set, while task correlation is
    learned by the task kernel.
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        observed = train_Y[~torch.isnan(train_Y)]
        if bool((observed < 0).any()):
            raise ValueError("Observed multiclass targets must be non-negative.")
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            **_prepare_kwargs(train_X, train_Y, kwargs, regression=False),
        )

    @property
    def num_classes_list(self) -> list[int]:
        return [int(self.num_classes) for _ in range(self.num_tasks)]

    def class_probs_list(self, X: Tensor) -> list[Tensor]:
        """Return one ``[..., q, C]`` probability tensor per task."""

        probabilities = self.class_probs(X)
        if int(probabilities.shape[-2]) != self.num_tasks:
            raise RuntimeError(
                "Wide multiclass posterior did not preserve the task axis: "
                f"shape={tuple(probabilities.shape)}, num_tasks={self.num_tasks}."
            )
        return [probabilities[..., index, :] for index in range(self.num_tasks)]

    def padded_class_probs(self, X: Tensor) -> Tensor:
        return self.class_probs(X)


__all__ = [
    "TaskFeatureInputTransform",
    "WideMultiTaskGP",
    "WideMultiTaskBinaryClassificationGPModel",
    "WideMultiTaskOrdinalGPModel",
    "WideMultiTaskMulticlassClassificationGPModel",
    "wide_to_long",
]
