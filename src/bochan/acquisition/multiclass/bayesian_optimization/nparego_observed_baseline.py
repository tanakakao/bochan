"""Observed-baseline helpers for multiclass NParEGO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def complete_multiclass_label_rows(train_Y: Tensor) -> Tensor:
    """Return rows where every multiclass output has an observed label."""

    train_Y = torch.as_tensor(train_Y)
    if train_Y.ndim != 2:
        raise ValueError(
            "Multiclass NParEGO baseline labels must have shape [n, m]. "
            f"Got shape={tuple(train_Y.shape)}."
        )
    if not train_Y.is_floating_point():
        return train_Y

    finite = torch.isfinite(train_Y)
    complete = finite.all(dim=-1)
    if bool(complete.any()):
        return train_Y[complete]

    observed_counts = finite.sum(dim=0).detach().cpu().tolist()
    raise ValueError(
        "Multiclass NParEGO requires at least one training row with every "
        "output observed to construct its objective-space baseline. Partially "
        "observed rows remain usable for model fitting. "
        f"Observed counts per output={observed_counts}."
    )


def same_tensor_storage(left: Any, right: Any) -> bool:
    """Return whether two tensor references point to the same stored values."""

    if left is right and left is not None:
        return True
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return False
    return (
        left.shape == right.shape
        and left.device == right.device
        and left.data_ptr() == right.data_ptr()
    )


def wide_multiclass_training_labels(model: Any) -> Tensor | None:
    """Return the model's retained wide multiclass labels, if available."""

    expected_outputs = getattr(model, "num_tasks", None)
    if expected_outputs is None:
        expected_outputs = getattr(model, "num_outputs", None)
    try:
        expected_outputs = None if expected_outputs is None else int(expected_outputs)
    except (TypeError, ValueError):
        expected_outputs = None

    for name in ("train_Y_wide", "train_Y", "train_targets"):
        value = getattr(model, name, None)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 2:
            continue
        if expected_outputs is not None and tensor.shape[-1] != expected_outputs:
            continue
        return tensor
    return None


def infer_multiclass_train_y(model: Any) -> Tensor | None:
    """Return complete wide multiclass labels retained by a model, if available."""

    train_Y = wide_multiclass_training_labels(model)
    return None if train_Y is None else complete_multiclass_label_rows(train_Y)


def is_training_label_baseline(model: Any, baseline: Any) -> bool:
    """Return whether ``baseline`` is the model's raw wide label tensor."""

    return same_tensor_storage(baseline, wide_multiclass_training_labels(model))


def objective_baseline_from_labels(
    module: Any,
    *,
    train_Y: Tensor,
    target_class: int | Sequence[int] | None,
    output_target_classes: Sequence[int] | None,
    class_reduction: str,
    utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None,
    objective_signs: Sequence[float] | Tensor | None,
    class_offset: int,
) -> Tensor:
    """Convert complete multiclass labels to objective-space baseline values."""

    return module.compute_observed_multiclass_utility(
        train_Y=complete_multiclass_label_rows(train_Y),
        target_class=target_class,
        output_target_classes=output_target_classes,
        class_reduction=class_reduction,
        utility_values=utility_values,
        objective_signs=objective_signs,
        class_offset=class_offset,
    )


__all__ = [
    "complete_multiclass_label_rows",
    "infer_multiclass_train_y",
    "is_training_label_baseline",
    "objective_baseline_from_labels",
    "same_tensor_storage",
    "wide_multiclass_training_labels",
]
