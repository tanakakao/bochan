"""Regression-only desktop workflow service.

This module intentionally keeps the desktop workflow thin. It translates
column-oriented UI settings into the tensor-oriented :mod:`bochan.api` objects
and returns JSON-friendly results for the local desktop UI.
"""

from __future__ import annotations

import base64
import io
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DatasetRecord:
    """Loaded tabular dataset kept in memory by the desktop app."""

    dataset_id: str
    name: str
    data: Any
    source_type: str
    profile: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetStore:
    """Simple in-memory dataset store for a local desktop process."""

    def __init__(self) -> None:
        self._records: dict[str, DatasetRecord] = {}

    def add(self, record: DatasetRecord) -> None:
        self._records[record.dataset_id] = record

    def get(self, dataset_id: str) -> DatasetRecord:
        try:
            return self._records[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset_id: {dataset_id}") from exc

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "n_rows": record.profile["n_rows"],
                "n_columns": record.profile["n_columns"],
            }
            for record in self._records.values()
        ]


def decode_base64_payload(value: str) -> bytes:
    """Decode browser FileReader base64 payloads.

    The UI may send either a raw base64 string or a data URL. Both are accepted.
    """

    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    return base64.b64decode(payload)


def load_dataframe_from_payload(
    *,
    source_type: Literal["csv", "excel", "sqlite", "duckdb"],
    content_base64: str | None = None,
    name: str | None = None,
    encoding: str = "utf-8-sig",
    sep: str | None = None,
    sheet_name: str | int | None = 0,
    sql: str | None = None,
    database_path: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a pandas DataFrame from a UI payload.

    Args:
        source_type: Data source type.
        content_base64: Browser-side file content encoded as base64.
        name: Original file name.
        encoding: CSV text encoding.
        sep: CSV delimiter. ``None`` lets pandas infer the delimiter.
        sheet_name: Excel sheet name or index.
        sql: SQL query for database sources.
        database_path: Local database path for SQLite / DuckDB.

    Returns:
        Loaded DataFrame and source metadata.
    """

    import pandas as pd

    metadata: dict[str, Any] = {"name": name, "source_type": source_type}
    if source_type == "csv":
        if not content_base64:
            raise ValueError("content_base64 is required for CSV input.")
        raw = decode_base64_payload(content_base64)
        text = raw.decode(encoding)
        read_kwargs: dict[str, Any] = {}
        if sep:
            read_kwargs["sep"] = sep
        else:
            read_kwargs["sep"] = None
            read_kwargs["engine"] = "python"
        return pd.read_csv(io.StringIO(text), **read_kwargs), metadata

    if source_type == "excel":
        if not content_base64:
            raise ValueError("content_base64 is required for Excel input.")
        raw = decode_base64_payload(content_base64)
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name), metadata

    if source_type == "sqlite":
        if not database_path:
            raise ValueError("database_path is required for SQLite input.")
        if not sql:
            raise ValueError("sql is required for SQLite input.")
        with sqlite3.connect(database_path) as conn:
            return pd.read_sql_query(sql, conn), {**metadata, "database_path": database_path, "sql": sql}

    if source_type == "duckdb":
        if not database_path:
            raise ValueError("database_path is required for DuckDB input.")
        if not sql:
            raise ValueError("sql is required for DuckDB input.")
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Install the desktop optional dependency to use DuckDB: pip install -e '.[desktop]'") from exc
        with duckdb.connect(database_path) as conn:
            return conn.execute(sql).df(), {**metadata, "database_path": database_path, "sql": sql}

    raise ValueError(f"Unsupported source_type: {source_type}")


def build_dataset_record(
    *,
    data: Any,
    name: str,
    source_type: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetRecord:
    """Create a dataset record from a pandas DataFrame."""

    dataset_id = str(uuid.uuid4())
    profile = profile_dataframe(data)
    return DatasetRecord(
        dataset_id=dataset_id,
        name=name,
        data=data,
        source_type=source_type,
        profile=profile,
        metadata=metadata or {},
    )


def dataframe_preview(data: Any, *, limit: int = 100) -> list[dict[str, Any]]:
    """Return a JSON-friendly preview of a DataFrame."""

    import pandas as pd

    preview = data.head(max(0, int(limit))).copy()
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def profile_dataframe(data: Any) -> dict[str, Any]:
    """Build a compact UI profile for a DataFrame."""

    import pandas as pd

    columns: list[dict[str, Any]] = []
    for name in data.columns:
        series = data[name]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        kind = infer_column_kind(series)
        column: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "kind": kind,
            "missing_count": missing_count,
            "missing_rate": float(missing_count / max(len(series), 1)),
            "unique_count": unique_count,
        }
        if pd.api.types.is_numeric_dtype(series):
            column.update(
                {
                    "min": _safe_float(non_null.min()) if len(non_null) else None,
                    "max": _safe_float(non_null.max()) if len(non_null) else None,
                    "mean": _safe_float(non_null.mean()) if len(non_null) else None,
                    "std": _safe_float(non_null.std()) if len(non_null) else None,
                }
            )
        else:
            values = [str(value) for value in non_null.astype(str).unique()[:30]]
            column["values"] = values
        columns.append(column)

    return {
        "n_rows": int(len(data)),
        "n_columns": int(len(data.columns)),
        "columns": columns,
    }


def infer_column_kind(series: Any) -> str:
    """Infer a UI-friendly column kind."""

    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    if unique_count <= max(20, int(0.2 * max(len(non_null), 1))):
        return "categorical"
    return "string"


def run_regression_workflow(request: Any, store: DatasetStore) -> dict[str, Any]:
    """Fit a single-output regression model and generate candidates.

    Args:
        request: Pydantic request object from the desktop API.
        store: Dataset store containing the selected dataset.

    Returns:
        JSON-friendly candidate and model summary.
    """

    import pandas as pd
    import torch

    from bochan.api import (
        AcquisitionConfig,
        BayesianOptimizer,
        CandidateRepairConfig,
        FitConfig,
        InputTransformConfig,
        ModelConfig,
        OptimizeConfig,
    )
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
    if has_steps:
        steps = [0.0 for _ in encoded["feature_columns"]]
        for idx, step in encoded["steps"].items():
            steps[int(idx)] = float(step)
        steps_tensor = torch.as_tensor(steps, dtype=torch.double)

    return CandidateRepairConfig(
        bounds=bounds,
        numeric_indices=encoded["numeric_indices"],
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


def _candidate_rows(*, candidates: Any, acq_value: Any, mean: Any, std: Any, encoded: dict[str, Any], request: Any) -> list[dict[str, Any]]:
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
                "raw": {
                    "candidate": to_serializable(candidates[rank - 1]),
                },
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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None
