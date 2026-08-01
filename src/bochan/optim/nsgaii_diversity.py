"""Diversity-aware final candidate selection for NSGA-II."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _normalize_columns(
    values: Tensor,
    *,
    lower: Tensor | None = None,
    upper: Tensor | None = None,
) -> Tensor:
    """Normalize columns while mapping constant columns to zero."""

    values = torch.as_tensor(values)
    if values.ndim != 2:
        raise ValueError(
            "values must have shape (n, d). "
            f"Got shape={tuple(values.shape)}."
        )
    if lower is None:
        lower = values.min(dim=0).values
    else:
        lower = torch.as_tensor(lower, device=values.device, dtype=values.dtype)
    if upper is None:
        upper = values.max(dim=0).values
    else:
        upper = torch.as_tensor(upper, device=values.device, dtype=values.dtype)

    span = upper - lower
    eps = torch.finfo(values.dtype).eps if values.is_floating_point() else 0.0
    active = span.abs() > eps
    safe_span = torch.where(active, span, torch.ones_like(span))
    normalized = (values - lower) / safe_span
    return torch.where(active.unsqueeze(0), normalized, torch.zeros_like(normalized))


def _initial_selection(
    *,
    initial_indices: Sequence[int] | Tensor | None,
    ideal_distance: Tensor,
    q: int,
) -> list[int]:
    """Resolve required initial rows or choose a balanced objective anchor."""

    if initial_indices is None:
        return [int(torch.argmin(ideal_distance).item())]

    indices = torch.as_tensor(
        initial_indices,
        device=ideal_distance.device,
        dtype=torch.long,
    ).reshape(-1)
    selected: list[int] = []
    for index in indices.tolist():
        index = int(index)
        if index < 0 or index >= ideal_distance.shape[0]:
            raise IndexError(
                f"initial index {index} is out of range for "
                f"n={ideal_distance.shape[0]}."
            )
        if index not in selected:
            selected.append(index)
    if len(selected) > q:
        raise ValueError(
            "initial_indices contains more unique rows than q. "
            f"Got {len(selected)} and q={q}."
        )
    return selected or [int(torch.argmin(ideal_distance).item())]


def select_diverse_nsgaii_candidates(
    candidates: Tensor,
    values: Tensor,
    *,
    q: int,
    bounds: Tensor,
    input_weight: float = 0.7,
    initial_indices: Sequence[int] | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Select a quality-anchored maximin subset in input and objective space.

    BoTorch's NSGA-II backend selects a requested subset by hypervolume in the
    objective space. That does not guarantee that the corresponding experimental
    conditions are separated in the input space. This selector starts from the
    Pareto point nearest the normalized ideal point, or from explicitly required
    rows, and greedily adds the point whose minimum distance from the selected set
    is largest.

    Distances are calculated independently in normalized input and objective
    spaces, then combined using ``input_weight``. The candidate pool is already
    Pareto-oriented, so input diversity receives the larger default weight.

    Args:
        candidates: Candidate pool with shape ``(n, d)``.
        values: Objective values with shape ``(n, m)``.
        q: Number of candidates to return.
        bounds: Search bounds with shape ``(2, d)``.
        input_weight: Weight assigned to normalized input-space distance. The
            objective-space distance receives ``1 - input_weight``.
        initial_indices: Optional rows that must be retained before maximin
            filling. This is used when fewer than q nondominated rows exist.

    Returns:
        The selected candidate and objective tensors, both preserving a shared
        row order.
    """

    candidates = torch.as_tensor(candidates)
    values = torch.as_tensor(values, device=candidates.device)
    bounds = torch.as_tensor(bounds, device=candidates.device, dtype=candidates.dtype)

    if candidates.ndim != 2:
        raise ValueError(
            "candidates must have shape (n, d). "
            f"Got shape={tuple(candidates.shape)}."
        )
    if values.ndim != 2:
        raise ValueError(
            "values must have shape (n, m). "
            f"Got shape={tuple(values.shape)}."
        )
    if candidates.shape[0] != values.shape[0]:
        raise ValueError(
            "candidates and values must have the same number of rows. "
            f"Got {candidates.shape[0]} and {values.shape[0]}."
        )
    if bounds.shape != torch.Size([2, candidates.shape[-1]]):
        raise ValueError(
            "bounds must have shape (2, d). "
            f"Got shape={tuple(bounds.shape)} for d={candidates.shape[-1]}."
        )
    if q < 1:
        raise ValueError("q must be at least 1.")
    if not 0.0 <= float(input_weight) <= 1.0:
        raise ValueError("input_weight must be between 0 and 1.")
    if candidates.shape[0] <= q:
        return candidates, values

    x_normalized = _normalize_columns(
        candidates,
        lower=bounds[0],
        upper=bounds[1],
    )
    y_normalized = _normalize_columns(values)

    ideal_distance = (1.0 - y_normalized).pow(2).mean(dim=-1)
    selected = _initial_selection(
        initial_indices=initial_indices,
        ideal_distance=ideal_distance,
        q=q,
    )

    x_scale = math.sqrt(max(1, x_normalized.shape[-1]))
    y_scale = math.sqrt(max(1, y_normalized.shape[-1]))
    weight = float(input_weight)

    while len(selected) < q:
        selected_tensor = torch.as_tensor(
            selected,
            device=candidates.device,
            dtype=torch.long,
        )
        input_distance = torch.cdist(
            x_normalized,
            x_normalized[selected_tensor],
        ) / x_scale
        objective_distance = torch.cdist(
            y_normalized,
            y_normalized[selected_tensor],
        ) / y_scale
        combined_distance = (
            weight * input_distance
            + (1.0 - weight) * objective_distance
        )
        minimum_distance = combined_distance.min(dim=-1).values
        minimum_distance[selected_tensor] = -torch.inf
        selected.append(int(torch.argmax(minimum_distance).item()))

    indices = torch.as_tensor(selected, device=candidates.device, dtype=torch.long)
    return candidates[indices], values[indices]


__all__ = ["select_diverse_nsgaii_candidates"]
