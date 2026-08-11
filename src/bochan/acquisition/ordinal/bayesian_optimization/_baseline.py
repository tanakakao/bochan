"""Baseline helpers for multi-output ordinal Bayesian optimization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def complete_ordinal_baseline_rows(train_Y: Tensor) -> Tensor:
    """Return rows where every ordinal output has an observed label."""

    tensor = torch.as_tensor(train_Y)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(
            "Ordinal multi-output baseline labels must have shape [n, m]. "
            f"Got shape={tuple(tensor.shape)}."
        )
    if not tensor.is_floating_point():
        return tensor

    finite = torch.isfinite(tensor)
    complete = finite.all(dim=-1)
    if bool(complete.any()):
        return tensor[complete]

    observed_counts = finite.sum(dim=0).detach().cpu().tolist()
    raise ValueError(
        "Ordinal NParEGO requires at least one training row with every output "
        "observed to construct its baseline scalarization. Partially observed "
        f"rows remain usable for model fitting. Observed counts per output={observed_counts}."
    )


def infer_multioutput_ordinal_train_y(model: Any) -> Tensor | None:
    """Infer complete wide ordinal labels without mistaking long targets for outputs."""

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
        return complete_ordinal_baseline_rows(tensor)

    submodels = getattr(model, "models", None)
    if submodels is None:
        return None

    columns: list[Tensor] = []
    for submodel in submodels:
        value = getattr(submodel, "train_Y", None)
        if value is None:
            value = getattr(submodel, "train_targets", None)
        if value is None:
            return None
        column = torch.as_tensor(value)
        if column.ndim == 1:
            column = column.unsqueeze(-1)
        elif column.shape[-1] != 1:
            return None
        columns.append(column)

    if not columns:
        return None
    return complete_ordinal_baseline_rows(torch.cat(columns, dim=-1))


__all__ = [
    "complete_ordinal_baseline_rows",
    "infer_multioutput_ordinal_train_y",
]
