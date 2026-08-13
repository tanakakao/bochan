"""Column-addressed configuration resolution for the tabular API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from bochan.api import CandidateRepairConfig, OptimizeConfig

from ..config import ColumnKey


def _torch():
    import torch

    return torch


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _column_mapping(feature_names: Sequence[ColumnKey]) -> dict[Any, int]:
    mapping: dict[Any, int] = {}
    for index, name in enumerate(feature_names):
        mapping[name] = index
        mapping[str(name)] = index
    return mapping


def resolve_column_indices(
    columns: Sequence[ColumnKey] | ColumnKey | None,
    feature_names: Sequence[ColumnKey],
    *,
    none_means_all: bool = False,
) -> list[int] | None:
    """Resolve column names / integer positions to positional indices."""

    if columns is None:
        return list(range(len(feature_names))) if none_means_all else None

    mapping = _column_mapping(feature_names)
    resolved: list[int] = []
    for column in _as_list(columns):
        if isinstance(column, int) and 0 <= column < len(feature_names):
            resolved.append(int(column))
        elif column in mapping:
            resolved.append(mapping[column])
        elif str(column) in mapping:
            resolved.append(mapping[str(column)])
        else:
            raise KeyError(
                f"Unknown column {column!r}. Available columns: {list(feature_names)!r}."
            )
    return resolved


def _lookup_mapping_value(
    mapping: Mapping[Any, Any],
    key: Any,
    feature_names: Sequence[ColumnKey],
) -> Any:
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    if isinstance(key, int) and 0 <= key < len(feature_names):
        name = feature_names[key]
        if name in mapping:
            return mapping[name]
        if str(name) in mapping:
            return mapping[str(name)]
    raise KeyError(f"No value found for column {key!r}.")


def _to_tensor(data: Any, *, dtype: Any, device: Any | None) -> Any:
    return _torch().as_tensor(data, dtype=dtype, device=device)


def bounds_to_tensor(
    bounds: Any | Mapping[ColumnKey, Sequence[float]] | None,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> Any | None:
    """Convert mapping / array bounds to a BoTorch-style ``2 x d`` tensor."""

    if bounds is None:
        return None

    torch = _torch()
    if torch.is_tensor(bounds):
        return bounds.to(device=device, dtype=dtype)

    if isinstance(bounds, Mapping):
        lower: list[float] = []
        upper: list[float] = []
        for name in feature_names:
            value = _lookup_mapping_value(bounds, name, feature_names)
            if len(value) != 2:
                raise ValueError(f"Bounds for {name!r} must have length 2.")
            lower.append(float(value[0]))
            upper.append(float(value[1]))
        return _to_tensor([lower, upper], dtype=dtype, device=device)

    return _to_tensor(bounds, dtype=dtype, device=device)


def _resolve_steps(
    steps: Any,
    numeric_columns: Sequence[ColumnKey] | None,
    feature_names: Sequence[ColumnKey],
) -> Any:
    if steps is None or not isinstance(steps, Mapping):
        return steps
    columns = list(numeric_columns) if numeric_columns is not None else list(feature_names)
    return [_lookup_mapping_value(steps, column, feature_names) for column in columns]


def _resolve_feature_mapping(
    mapping: Mapping[Any, Any] | None,
    feature_names: Sequence[ColumnKey],
) -> dict[int, float] | None:
    if mapping is None:
        return None
    resolved: dict[int, float] = {}
    for key, value in mapping.items():
        index = resolve_column_indices([key], feature_names)
        assert index is not None
        resolved[int(index[0])] = float(value)
    return resolved


def _resolve_fixed_features_list(
    values: Sequence[Mapping[Any, Any]] | None,
    feature_names: Sequence[ColumnKey],
) -> list[dict[int, float]] | None:
    if values is None:
        return None
    return [_resolve_feature_mapping(item, feature_names) or {} for item in values]


def _is_tensor(value: Any) -> bool:
    return bool(_torch().is_tensor(value))


def _resolve_constraint_indices(
    indices: Any,
    feature_names: Sequence[ColumnKey],
    *,
    device: Any | None,
) -> Any:
    torch = _torch()
    if torch.is_tensor(indices):
        return indices.to(device=device, dtype=torch.long).reshape(-1)
    resolved = resolve_column_indices(indices, feature_names)
    return torch.as_tensor(resolved, dtype=torch.long, device=device).reshape(-1)


def _resolve_constraint_coefficients(
    coefficients: Any,
    *,
    dtype: Any,
    device: Any | None,
) -> Any:
    torch = _torch()
    if torch.is_tensor(coefficients):
        return coefficients.to(device=device, dtype=dtype).reshape(-1)
    return torch.as_tensor(coefficients, dtype=dtype, device=device).reshape(-1)


def _resolve_linear_constraints(
    constraints: Any | None,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> Any | None:
    if constraints is None:
        return None
    return [
        (
            _resolve_constraint_indices(indices, feature_names, device=device),
            _resolve_constraint_coefficients(coefficients, dtype=dtype, device=device),
            rhs,
        )
        for indices, coefficients, rhs in constraints
    ]


def _resolve_sum_constraint_indices(
    indices: Any,
    feature_names: Sequence[ColumnKey],
) -> list[int]:
    if _is_tensor(indices):
        return [int(item) for item in indices.detach().cpu().reshape(-1).tolist()]
    resolved = resolve_column_indices(indices, feature_names)
    return [] if resolved is None else [int(item) for item in resolved]


def _resolve_final_sum_constraint(
    value: tuple[Sequence[Any], float] | None,
    feature_names: Sequence[ColumnKey],
) -> tuple[list[int], float] | None:
    if value is None:
        return None
    indices, rhs = value
    return (_resolve_sum_constraint_indices(indices, feature_names), rhs)


def resolve_repair_config_columns(
    repair: CandidateRepairConfig | None,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> CandidateRepairConfig | None:
    """Resolve column names inside ``CandidateRepairConfig`` to indices."""

    if repair is None:
        return None

    numeric_cols = _as_list(repair.numeric_indices) if repair.numeric_indices is not None else None
    if numeric_cols is None and isinstance(repair.steps, Mapping):
        numeric_cols = list(repair.steps.keys())
    comp_cols = _as_list(repair.comp_idx) if repair.comp_idx is not None else None

    numeric_indices = resolve_column_indices(numeric_cols, feature_names) if numeric_cols is not None else None
    comp_idx = resolve_column_indices(comp_cols, feature_names) if comp_cols is not None else None
    steps = _resolve_steps(repair.steps, numeric_cols, feature_names)
    bounds = bounds_to_tensor(repair.bounds, feature_names, dtype=dtype, device=device)

    return replace(
        repair,
        bounds=bounds,
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=comp_idx,
        equality_constraints=_resolve_linear_constraints(
            repair.equality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        inequality_constraints=_resolve_linear_constraints(
            repair.inequality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        fixed_features=_resolve_feature_mapping(repair.fixed_features, feature_names),
        final_sum_constraint=_resolve_final_sum_constraint(repair.final_sum_constraint, feature_names),
    )


def resolve_optimize_config_columns(
    config: OptimizeConfig,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> OptimizeConfig:
    """Resolve column names inside ``OptimizeConfig`` to tensor API indices."""

    return replace(
        config,
        fixed_features=_resolve_feature_mapping(config.fixed_features, feature_names),
        fixed_features_list=_resolve_fixed_features_list(config.fixed_features_list, feature_names),
        equality_constraints=_resolve_linear_constraints(
            config.equality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        inequality_constraints=_resolve_linear_constraints(
            config.inequality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        repair_config=resolve_repair_config_columns(
            config.repair_config,
            feature_names,
            dtype=dtype,
            device=device,
        ),
    )


__all__ = [
    "bounds_to_tensor",
    "resolve_column_indices",
    "resolve_optimize_config_columns",
    "resolve_repair_config_columns",
]
