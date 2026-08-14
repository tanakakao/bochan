"""Shared encoding and candidate post-processing helpers for Web workflows."""

from __future__ import annotations

from typing import Any


def _encode_features(*, data: Any, feature_columns: list[str], search_space: list[Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    specs = {spec.name: spec for spec in search_space}
    arrays: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    cat_dims: list[int] = []
    numeric_indices: list[int] = []
    category_maps: dict[str, dict[str, int]] = {}
    inverse_category_maps: dict[str, dict[int, str]] = {}
    fixed_features: dict[int, float] = {}
    steps: dict[int, float] = {}

    for idx, column in enumerate(feature_columns):
        series = data[column]
        spec = specs.get(column)
        requested_type = getattr(spec, "type", "auto") if spec is not None else "auto"
        is_categorical = requested_type == "categorical" or (
            requested_type == "auto" and not pd.api.types.is_numeric_dtype(series)
        )

        if is_categorical:
            values = _ordered_categories(series, getattr(spec, "categories", None) if spec is not None else None)
            mapping = {str(value): i for i, value in enumerate(values)}
            encoded = series.astype(str).map(mapping)
            if encoded.isna().any():
                unknown = sorted(set(series.astype(str)) - set(mapping))
                raise ValueError(f"Unknown categorical values in column {column}: {unknown}")
            arrays.append(encoded.to_numpy(dtype=float))
            lower.append(0.0)
            upper.append(float(max(len(values) - 1, 0)))
            cat_dims.append(idx)
            category_maps[column] = mapping
            inverse_category_maps[column] = {i: key for key, i in mapping.items()}
            if spec is not None and spec.fixed:
                if spec.fixed_value is None:
                    raise ValueError(f"fixed_value is required for fixed feature: {column}")
                fixed_features[idx] = float(mapping[str(spec.fixed_value)])
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Feature column contains non-numeric values: {column}")
        values = numeric.to_numpy(dtype=float)
        arrays.append(values)
        lower.append(_resolve_lower(values, spec))
        upper.append(_resolve_upper(values, spec))
        numeric_indices.append(idx)
        if spec is not None and spec.fixed:
            if spec.fixed_value is None:
                raise ValueError(f"fixed_value is required for fixed feature: {column}")
            fixed_features[idx] = float(spec.fixed_value)
        if spec is not None and spec.step is not None and float(spec.step) > 0:
            steps[idx] = float(spec.step)

    X = np.column_stack(arrays)
    return {
        "X": X,
        "bounds": [lower, upper],
        "feature_columns": feature_columns,
        "cat_dims": cat_dims,
        "numeric_indices": numeric_indices,
        "category_maps": category_maps,
        "inverse_category_maps": inverse_category_maps,
        "fixed_features": fixed_features,
        "steps": steps,
    }


def _ordered_categories(series: Any, requested: list[Any] | None) -> list[str]:
    if requested:
        return [str(value) for value in requested]
    return sorted(str(value) for value in series.dropna().astype(str).unique())


def _resolve_lower(values: Any, spec: Any | None) -> float:
    if spec is not None and spec.lower is not None:
        return float(spec.lower)
    return float(values.min())


def _resolve_upper(values: Any, spec: Any | None) -> float:
    if spec is not None and spec.upper is not None:
        return float(spec.upper)
    return float(values.max())


def _requires_best_f(acq_name: str) -> bool:
    return acq_name.replace("_", "").replace("-", "").lower() in {
        "ei",
        "qei",
        "expectedimprovement",
        "logei",
        "qlogei",
        "pi",
        "qpi",
        "probabilityofimprovement",
    }


def _requires_beta(acq_name: str) -> bool:
    return acq_name.replace("_", "").replace("-", "").lower() in {"ucb", "qucb", "upperconfidencebound"}


def _build_repair_config(*, request: Any, encoded: dict[str, Any], bounds: Any) -> Any:
    import torch

    from bochan.api import CandidateRepairConfig

    equality_constraints, inequality_constraints = _linear_constraints(request.constraints or [], encoded)
    comp_idx = _sparse_indices(request.k_sparse, encoded["feature_columns"])
    has_steps = bool(encoded["steps"])
    has_repair = bool(equality_constraints or inequality_constraints or comp_idx or has_steps or encoded["fixed_features"])
    if not has_repair:
        return None

    steps_tensor = None
    repair_numeric_indices = encoded["numeric_indices"]
    if has_steps:
        step_indices = sorted(int(idx) for idx in encoded["steps"])
        steps_tensor = torch.as_tensor([float(encoded["steps"][idx]) for idx in step_indices], dtype=torch.double)
        repair_numeric_indices = step_indices

    return CandidateRepairConfig(
        bounds=bounds,
        numeric_indices=repair_numeric_indices,
        steps=steps_tensor,
        comp_idx=comp_idx or None,
        k=request.k_sparse.k if request.k_sparse and request.k_sparse.enabled else 0,
        equality_constraints=equality_constraints or None,
        inequality_constraints=inequality_constraints or None,
        inequality_sense="le",
        fixed_features=encoded["fixed_features"] or None,
        score=request.k_sparse.score if request.k_sparse else "abs",
        support_selection=request.k_sparse.support_selection if request.k_sparse else "topk",
        final_priority=request.k_sparse.final_priority if request.k_sparse else "grid",
    )


def _linear_constraints(constraints: list[Any], encoded: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    import torch

    feature_to_index = {name: idx for idx, name in enumerate(encoded["feature_columns"])}
    equality_constraints = []
    inequality_constraints = []
    for constraint in constraints:
        if not constraint.enabled:
            continue
        indices: list[int] = []
        coefficients: list[float] = []
        for term in constraint.terms:
            if term.column not in feature_to_index:
                raise ValueError(f"Constraint column is not a selected feature: {term.column}")
            indices.append(feature_to_index[term.column])
            coefficients.append(float(term.coefficient))
        rhs = float(constraint.rhs)
        if constraint.sense == "eq":
            equality_constraints.append((torch.as_tensor(indices, dtype=torch.long), torch.as_tensor(coefficients), rhs))
        elif constraint.sense == "le":
            inequality_constraints.append((torch.as_tensor(indices, dtype=torch.long), torch.as_tensor(coefficients), rhs))
        elif constraint.sense == "ge":
            inequality_constraints.append((torch.as_tensor(indices, dtype=torch.long), -torch.as_tensor(coefficients), -rhs))
        else:
            raise ValueError(f"Unknown constraint sense: {constraint.sense}")
    return equality_constraints, inequality_constraints


def _sparse_indices(k_sparse: Any, feature_columns: list[str]) -> list[int]:
    if k_sparse is None or not k_sparse.enabled:
        return []
    feature_to_index = {name: idx for idx, name in enumerate(feature_columns)}
    missing = [column for column in k_sparse.columns if column not in feature_to_index]
    if missing:
        raise ValueError(f"k-sparse columns are not selected features: {missing}")
    if int(k_sparse.k) <= 0:
        raise ValueError("k must be positive when k-sparse is enabled.")
    return [feature_to_index[column] for column in k_sparse.columns]


def _postprocess_candidates(candidates: Any, *, request: Any, encoded: dict[str, Any]) -> Any:
    import torch

    result = candidates.detach().clone()
    if result.ndim == 1:
        result = result.unsqueeze(0)

    lower = torch.as_tensor(encoded["bounds"][0], dtype=result.dtype, device=result.device)
    upper = torch.as_tensor(encoded["bounds"][1], dtype=result.dtype, device=result.device)
    result = torch.maximum(torch.minimum(result, upper), lower)

    for idx, value in encoded["fixed_features"].items():
        result[..., int(idx)] = float(value)

    for idx in encoded["cat_dims"]:
        result[..., idx] = result[..., idx].round().clamp(lower[idx], upper[idx])

    for idx, step in encoded["steps"].items():
        if step > 0:
            result[..., int(idx)] = lower[int(idx)] + torch.round((result[..., int(idx)] - lower[int(idx)]) / step) * step
            result[..., int(idx)] = result[..., int(idx)].clamp(lower[int(idx)], upper[int(idx)])

    if request.k_sparse is not None and request.k_sparse.enabled:
        comp_idx = _sparse_indices(request.k_sparse, encoded["feature_columns"])
        k = min(int(request.k_sparse.k), len(comp_idx))
        if k < len(comp_idx):
            values = result[..., comp_idx]
            score = values.abs() if request.k_sparse.score == "abs" else values
            keep_local = torch.topk(score, k=k, dim=-1).indices
            mask = torch.zeros_like(values, dtype=torch.bool)
            mask.scatter_(-1, keep_local, True)
            result[..., comp_idx] = torch.where(mask, values, torch.zeros_like(values))

    return result


__all__ = [
    "_build_repair_config",
    "_encode_features",
    "_postprocess_candidates",
    "_requires_best_f",
    "_requires_beta",
]
