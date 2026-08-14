"""Compatibility layer for the legacy desktop shell.

Shared dataset and candidate-workflow helpers now live under
:mod:`bochan.serving.webapp`.  This module re-exports them so the legacy desktop
entry point and older imports keep working until the desktop package is removed.
"""

from __future__ import annotations

from typing import Any

from bochan.serving.webapp.datasets import (
    DatasetRecord,
    DatasetStore,
    build_dataset_record,
    dataframe_preview,
    decode_base64_payload,
    infer_column_kind,
    load_dataframe_from_payload,
    profile_dataframe,
)
from bochan.serving.webapp.workflow_utils import (
    _build_repair_config,
    _encode_features,
    _linear_constraints,
    _ordered_categories,
    _postprocess_candidates,
    _requires_best_f,
    _requires_beta,
    _resolve_lower,
    _resolve_upper,
    _sparse_indices,
)


def run_regression_workflow(request: Any, store: DatasetStore) -> dict[str, Any]:
    """Fit a single-output regression model for the legacy desktop shell."""

    import pandas as pd
    import torch

    from bochan.api import AcquisitionConfig, BayesianOptimizer, FitConfig, InputTransformConfig, ModelConfig, OptimizeConfig
    from bochan.serving.fastapi.converters import to_serializable

    record = store.get(request.dataset_id)
    data = record.data.copy()
    feature_columns = list(request.feature_columns)
    target_column = request.target_column

    _validate_regression_columns(data, feature_columns, target_column)
    data = _clean_regression_rows(data, feature_columns, target_column, drop_missing=request.drop_missing)

    encoded = _encode_features(
        data=data,
        feature_columns=feature_columns,
        search_space=list(request.search_space or []),
    )
    target = pd.to_numeric(data[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Target column contains non-numeric values after conversion: {target_column}")

    direction_sign = 1.0 if request.direction == "maximize" else -1.0
    train_X = torch.as_tensor(encoded["X"], dtype=torch.double)
    train_Y = torch.as_tensor((target.to_numpy(dtype=float) * direction_sign).reshape(-1, 1), dtype=torch.double)
    bounds = torch.as_tensor(encoded["bounds"], dtype=torch.double)

    model_config = ModelConfig(
        task_type="regression",
        model_type=request.model_type,
        cat_dims=encoded["cat_dims"] or None,
        input_transform_config=InputTransformConfig(
            normalize=request.normalize,
            perturbation=request.input_perturbation,
            n_w=request.n_w,
            std=request.perturbation_std,
            bounds=bounds,
            categorical_idx=encoded["cat_dims"] or None,
        ),
        outcome_transform=request.outcome_transform,
        model_kwargs=dict(request.model_kwargs or {}),
    )
    fit_config = FitConfig(maxiter=request.fit_maxiter)

    optimizer = BayesianOptimizer(model_config=model_config, fit_config=fit_config, bounds=bounds)
    optimizer.fit(train_X, train_Y)

    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    acq_name = request.acquisition.name
    if _requires_best_f(acq_name) and "best_f" not in acqf_kwargs:
        acqf_kwargs["best_f"] = train_Y.max()
    if _requires_beta(acq_name) and "beta" not in acqf_kwargs:
        acqf_kwargs["beta"] = request.acquisition.beta

    acq_config = AcquisitionConfig(name=acq_name, acqf_kwargs=acqf_kwargs)

    repair_config = _build_repair_config(
        request=request,
        encoded=encoded,
        bounds=bounds,
    )
    opt_config = OptimizeConfig(
        q=request.optimizer.q,
        num_restarts=request.optimizer.num_restarts,
        raw_samples=request.optimizer.raw_samples,
        sequential=request.optimizer.sequential,
        optimizer=request.optimizer.name,
        repair_config=repair_config,
        fixed_features=encoded["fixed_features"] or None,
    )

    raw_candidates, raw_acq_value = optimizer.candidate(acq_config, opt_config)
    candidates = _postprocess_candidates(raw_candidates, request=request, encoded=encoded)

    mean, variance = optimizer.predict(candidates, return_type="mean_variance")
    std = variance.clamp_min(0).sqrt()

    rows = _candidate_rows(
        candidates=candidates,
        acq_value=raw_acq_value,
        mean=mean,
        std=std,
        encoded=encoded,
        request=request,
    )

    best_observed = float(target.max()) if request.direction == "maximize" else float(target.min())
    return {
        "dataset_id": record.dataset_id,
        "dataset_name": record.name,
        "task_type": "regression",
        "model_type": request.model_type,
        "n_train": int(train_X.shape[0]),
        "n_features": int(train_X.shape[1]),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "direction": request.direction,
        "cat_dims": encoded["cat_dims"],
        "category_maps": encoded["category_maps"],
        "best_observed": best_observed,
        "bounds": to_serializable(bounds),
        "raw_acq_value": to_serializable(raw_acq_value),
        "candidates": rows,
        "metadata": {
            "dropped_rows": int(record.profile["n_rows"] - len(data)),
            "acquisition": acq_name,
            "optimizer": request.optimizer.name,
            "repair_enabled": repair_config is not None,
        },
    }


def _validate_regression_columns(data: Any, feature_columns: list[str], target_column: str) -> None:
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    if not target_column:
        raise ValueError("target_column is required.")
    missing = [column for column in [*feature_columns, target_column] if column not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    if target_column in feature_columns:
        raise ValueError("target_column must not be included in feature_columns.")


def _clean_regression_rows(data: Any, feature_columns: list[str], target_column: str, *, drop_missing: bool) -> Any:
    if not drop_missing:
        if data[[*feature_columns, target_column]].isna().any().any():
            raise ValueError("Missing values are present. Enable drop_missing or clean the dataset first.")
        return data
    return data.dropna(subset=[*feature_columns, target_column]).reset_index(drop=True)


def _candidate_rows(
    *,
    candidates: Any,
    acq_value: Any,
    mean: Any,
    std: Any,
    encoded: dict[str, Any],
    request: Any,
) -> list[dict[str, Any]]:
    from bochan.serving.fastapi.converters import to_serializable

    direction_sign = 1.0 if request.direction == "maximize" else -1.0
    feature_columns = encoded["feature_columns"]
    inverse_maps = encoded["inverse_category_maps"]
    candidate_values = candidates.detach().cpu().tolist()
    means = mean.detach().cpu().reshape(candidates.shape[0], -1)[:, 0].tolist()
    stds = std.detach().cpu().reshape(candidates.shape[0], -1)[:, 0].tolist()
    acq_values = _broadcast_acq_values(acq_value, candidates.shape[0])

    rows: list[dict[str, Any]] = []
    for rank, values in enumerate(candidate_values, start=1):
        decoded: dict[str, Any] = {}
        encoded_values: dict[str, float] = {}
        for idx, column in enumerate(feature_columns):
            value = float(values[idx])
            encoded_values[column] = value
            if column in inverse_maps:
                decoded[column] = inverse_maps[column].get(int(round(value)), str(int(round(value))))
            else:
                decoded[column] = value
        constraints = _evaluate_constraints(decoded, request.constraints or [])
        rows.append(
            {
                "rank": rank,
                "values": decoded,
                "encoded_values": encoded_values,
                "acq_value": acq_values[rank - 1],
                "predicted_objective_mean": float(means[rank - 1]),
                "predicted_target_mean": float(means[rank - 1] * direction_sign),
                "predicted_target_std": float(stds[rank - 1]),
                "constraints_ok": all(item["ok"] for item in constraints),
                "constraints": constraints,
                "raw": {"candidate": to_serializable(candidates[rank - 1])},
            }
        )
    return rows


def _broadcast_acq_values(acq_value: Any, n: int) -> list[float | None]:
    try:
        values = acq_value.detach().cpu().reshape(-1).tolist()
    except Exception:
        values = [acq_value]
    values = [float(value) for value in values if value is not None]
    if not values:
        return [None for _ in range(n)]
    if len(values) == 1:
        return values * n
    if len(values) < n:
        return values + [values[-1]] * (n - len(values))
    return values[:n]


def _evaluate_constraints(values: dict[str, Any], constraints: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for constraint in constraints:
        if not constraint.enabled:
            continue
        lhs = 0.0
        for term in constraint.terms:
            try:
                lhs += float(values[term.column]) * float(term.coefficient)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Constraint evaluation requires numeric candidate values: {term.column}") from exc
        rhs = float(constraint.rhs)
        tol = 1e-8
        if constraint.sense == "le":
            ok = lhs <= rhs + tol
            violation = max(lhs - rhs, 0.0)
        elif constraint.sense == "ge":
            ok = lhs >= rhs - tol
            violation = max(rhs - lhs, 0.0)
        else:
            ok = abs(lhs - rhs) <= tol
            violation = abs(lhs - rhs)
        results.append(
            {
                "name": constraint.name,
                "sense": constraint.sense,
                "lhs": lhs,
                "rhs": rhs,
                "ok": bool(ok),
                "violation": float(violation),
            }
        )
    return results


__all__ = [
    "DatasetRecord",
    "DatasetStore",
    "build_dataset_record",
    "dataframe_preview",
    "decode_base64_payload",
    "infer_column_kind",
    "load_dataframe_from_payload",
    "profile_dataframe",
    "run_regression_workflow",
]
