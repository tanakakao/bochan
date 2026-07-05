"""Compatibility adapters for wide-format multi-task models.

The public API receives ``X=[n, d]`` and ``Y=[n, m]``. The underlying task-feature
models operate on long inputs with an appended task-id column. This module keeps
public input transforms in the original ``d``-dimensional space, preserves task
ids, and exposes the task count as the public output count.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import StratifiedStandardize
from torch import Tensor

from .wide_multitask import (
    WideMultiTaskBinaryClassificationGPModel as _WideMultiTaskBinaryClassificationGPModel,
)
from .wide_multitask import WideMultiTaskGP as _WideMultiTaskGP
from .wide_multitask import (
    WideMultiTaskMulticlassClassificationGPModel as _WideMultiTaskMulticlassClassificationGPModel,
)
from .wide_multitask import WideMultiTaskOrdinalGPModel as _WideMultiTaskOrdinalGPModel
from .wide_multitask import _WidePosterior, wide_to_long


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
    """Apply a public-space transform with optional appended task ids.

    Model internals call this transform with long-format inputs ``[..., d + 1]``
    whose final column is the task id. Acquisition functions may also use the
    model transform directly for distance and diversity calculations, in which
    case they pass public inputs ``[..., d]``. Both layouts are supported:

    - public input ``[..., d]``: apply only ``base_transform``;
    - internal input ``[..., d + 1]``: transform data columns and preserve the
      task-id column.
    """

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

    def _has_task_feature(self, X: Tensor) -> bool:
        feature_dim = int(X.shape[-1])
        if feature_dim == self.data_dim:
            return False
        if feature_dim == self.data_dim + 1:
            return True
        raise RuntimeError(
            "Expected either public data columns or public data columns followed "
            "by one task-id column. "
            f"Expected d={self.data_dim} or d={self.data_dim + 1}, got {feature_dim}."
        )

    def _split(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if not self._has_task_feature(X):
            raise RuntimeError(
                "Task-feature splitting requires an appended task-id column. "
                f"Expected d={self.data_dim + 1}, got {X.shape[-1]}."
            )
        return X[..., : self.data_dim], X[..., self.data_dim :]

    def transform(self, X: Tensor) -> Tensor:
        if not self._has_task_feature(X):
            return self.base_transform(X)

        data, task_feature = self._split(X)
        transformed = self.base_transform(data)
        task_feature, transformed = _align_task_feature(
            task_feature,
            transformed,
        )
        return torch.cat([transformed, task_feature], dim=-1)

    def untransform(self, X: Tensor) -> Tensor:
        if not self._has_task_feature(X):
            return self.base_transform.untransform(X)

        data, task_feature = self._split(X)
        untransformed = self.base_transform.untransform(data)
        task_feature, untransformed = _align_task_feature(
            task_feature,
            untransformed,
        )
        return torch.cat([untransformed, task_feature], dim=-1)


class PerturbationAwareStratifiedStandardize(StratifiedStandardize):
    """Align task-wise outcome statistics with one-to-many input transforms.

    BoTorch passes the pre-transform long-format ``X`` to
    ``untransform_posterior``. An ``InputPerturbation`` expands every long row by
    ``n_w`` inside the model, so the posterior contains more points than ``X``.
    Repeating each task-id row by the inferred expansion factor keeps the
    per-task means and standard deviations aligned with the posterior.
    """

    @staticmethod
    def _repeat_X_to_length(X: Tensor, target_n: int) -> Tensor:
        current_n = int(X.shape[-2])
        if target_n == current_n:
            return X
        if current_n <= 0 or target_n % current_n != 0:
            raise RuntimeError(
                "Could not align StratifiedStandardize inputs with the posterior. "
                f"X has {current_n} rows, posterior has {target_n} points."
            )
        return X.repeat_interleave(target_n // current_n, dim=-2)

    def untransform(
        self,
        Y: Tensor,
        Yvar: Tensor | None = None,
        X: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if X is not None:
            X = self._repeat_X_to_length(X, int(Y.shape[-2]))
        return super().untransform(Y=Y, Yvar=Yvar, X=X)

    def untransform_posterior(self, posterior, X: Tensor | None = None):
        if X is not None:
            distribution = getattr(posterior, "distribution", None)
            mean = getattr(distribution, "mean", None)
            if mean is not None:
                if bool(getattr(posterior, "_is_mt", False)):
                    target_n = int(mean.shape[-2])
                else:
                    target_n = int(mean.shape[-1])
                X = self._repeat_X_to_length(X, target_n)
        return super().untransform_posterior(posterior=posterior, X=X)


def _build_stratified_standardize(
    train_X: Tensor,
    train_Y: Tensor,
) -> StratifiedStandardize:
    """Construct StratifiedStandardize across supported BoTorch signatures.

    BoTorch 0.15/0.16 uses ``task_values`` while newer releases use
    ``observed_task_values`` / ``all_task_values`` and optionally accept
    ``dtype``. Inspecting the installed constructor avoids pinning the adapter to
    either API generation.
    """

    task_values = torch.arange(
        int(train_Y.shape[-1]),
        device=train_X.device,
        dtype=train_X.dtype,
    )
    parameters = inspect.signature(StratifiedStandardize.__init__).parameters
    transform_kwargs: dict[str, Any] = {"stratification_idx": -1}

    if "observed_task_values" in parameters:
        transform_kwargs["observed_task_values"] = task_values
    if "all_task_values" in parameters:
        transform_kwargs["all_task_values"] = task_values
        if "dtype" in parameters:
            transform_kwargs["dtype"] = train_X.dtype
    elif "task_values" in parameters:
        transform_kwargs["task_values"] = task_values
    else:
        available = ", ".join(parameters)
        raise RuntimeError(
            "Unsupported StratifiedStandardize constructor. Expected either "
            f"'task_values' or 'all_task_values'; available parameters: {available}."
        )

    return PerturbationAwareStratifiedStandardize(**transform_kwargs)


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
            prepared["outcome_transform"] = _build_stratified_standardize(
                train_X,
                train_Y,
            )
    return prepared


class _PublicTaskOutputMixin:
    """Expose task count and perturbation-expanded q as public dimensions."""

    @property
    def num_outputs(self) -> int:
        return int(getattr(self, "num_tasks", 1))

    def _wrap_wide_posterior(
        self,
        base,
        *,
        X: Tensor,
        selected: list[int],
        posterior_transform: Any = None,
    ):
        flat_points = int(base.mean.shape[-2])
        num_tasks = int(self.num_tasks)
        if flat_points % num_tasks != 0:
            raise RuntimeError(
                "Wide posterior point count must be divisible by num_tasks. "
                f"Got flat_points={flat_points}, num_tasks={num_tasks}."
            )
        posterior = _WidePosterior(
            base,
            q=flat_points // num_tasks,
            num_tasks=num_tasks,
            output_indices=selected,
            input_ndim=X.ndim,
        )
        return posterior_transform(posterior) if posterior_transform is not None else posterior


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
    """Wide ordinal multi-task GP accepting public input transforms."""

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
    """Wide multiclass multi-task GP accepting public input transforms."""

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
    "PerturbationAwareStratifiedStandardize",
    "WideMultiTaskGP",
    "WideMultiTaskBinaryClassificationGPModel",
    "WideMultiTaskOrdinalGPModel",
    "WideMultiTaskMulticlassClassificationGPModel",
    "wide_to_long",
]
