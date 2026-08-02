"""TabularBayesianOptimizer endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from bochan.tabular import TabularBayesianOptimizer

from ..converters import model_metadata, to_serializable
from ..dependencies import TabularOptimizerStore, get_tabular_optimizer_store
from ..schemas import ModelDeleteResponse, ModelListResponse
from ..schemas.tabular import (
    TabularCandidateRequest,
    TabularCandidateResponse,
    TabularFitModelRequest,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)

router = APIRouter(prefix="/tabular/models", tags=["tabular"])

_TABULAR_CANDIDATE_DIRECT_FIELDS = (
    "outcome_constraint_config",
    "objective_mode",
    "objective_output",
    "objective_outputs",
    "objective_specs",
    "objective_directions",
    "objective_weights",
    "objective_eq_targets",
    "objective_direction",
    "objective_weight",
    "objective_eq_target",
    "objective_n_w",
    "objective_risk_type",
    "objective_alpha",
    "objective_maximize",
    "objective_aggregate_mean_when_no_risk",
    "objective_allow_unexpanded",
    "objective_utility_values",
    "objective_ordinal_likelihood",
    "evo_method",
)

_TABULAR_CANDIDATE_OPTIMIZE_ALIASES = (
    "constraints",
    "repair_config",
)


def _normalize_string_dtypes(frame: Any, pd: Any) -> Any:
    """Convert pandas string extension columns to mutable object columns.

    Pandas may infer JSON string fields as ``StringDtype``. The tabular
    converter subsequently replaces categorical labels with integer codes, and
    extension string arrays reject that dtype-changing assignment. Normalizing
    string columns at the HTTP boundary keeps behavior consistent across pandas
    versions while preserving the original string values.
    """

    for column in frame.columns:
        series = frame.loc[:, column]
        if pd.api.types.is_string_dtype(series.dtype):
            frame[column] = series.astype(object)
    return frame


def _to_dataframe(data: Any):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for tabular FastAPI endpoints.") from exc

    if isinstance(data, list):
        frame = pd.DataFrame.from_records(data)
    elif isinstance(data, dict):
        frame = pd.DataFrame(data)
    else:
        raise TypeError("data must be a list of records or a column-oriented object.")
    if frame.empty:
        raise ValueError("data must contain at least one row.")
    return _normalize_string_dtypes(frame, pd)


def _schema_dict(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _candidate_direct_kwargs(request: TabularCandidateRequest) -> dict[str, Any]:
    """Return explicitly supplied acquisition/objective candidate arguments."""

    fields_set = getattr(request, "model_fields_set", set())
    kwargs: dict[str, Any] = {}
    for name in _TABULAR_CANDIDATE_DIRECT_FIELDS:
        if name not in fields_set:
            continue
        value = getattr(request, name)
        if hasattr(value, "model_dump"):
            value = _schema_dict(value)
        kwargs[name] = value
    return kwargs


def _candidate_optimize_config(request: TabularCandidateRequest) -> dict[str, Any]:
    """Return optimize config with top-level input-constraint aliases applied.

    ``outcome_constraint_config`` describes constraints on model outputs and is
    forwarded directly to the acquisition builder. In contrast, linear input
    ``constraints`` and candidate ``repair_config`` belong to ``OptimizeConfig``.
    The FastAPI request accepts those two values either nested under
    ``optimize_config`` or at the candidate-request top level for convenience.
    """

    fields_set = getattr(request, "model_fields_set", set())
    opt_config = dict(request.opt_config or {})
    for name in _TABULAR_CANDIDATE_OPTIMIZE_ALIASES:
        if name not in fields_set:
            continue
        if name in opt_config:
            raise ValueError(
                f"Specify {name!r} either at the candidate request top level or "
                f"inside optimize_config.{name}, not both."
            )
        value = getattr(request, name)
        if hasattr(value, "model_dump"):
            value = _schema_dict(value)
        opt_config[name] = value
    return opt_config


def _frame_records(frame: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Return strict JSON records, converting NaN and timestamps safely."""

    columns = [str(column) for column in frame.columns]
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    return columns, records


def _fit_response(model_id: str, optimizer: TabularBayesianOptimizer) -> TabularModelFitResponse:
    dataset = optimizer.dataset
    bundle = optimizer.bo.bundle
    if dataset is None or bundle is None:
        raise RuntimeError("Tabular optimizer has no fitted dataset or model bundle.")

    categorical_cols = [
        dataset.feature_names[index]
        for index in dataset.cat_dims
        if 0 <= index < len(dataset.feature_names)
    ]
    return TabularModelFitResponse(
        model_id=model_id,
        task_type=str(bundle.task_type),
        model_type=str(bundle.model_type),
        n_train=int(dataset.X.shape[-2]) if hasattr(dataset.X, "shape") else None,
        feature_names=to_serializable(dataset.feature_names),
        target_names=to_serializable(dataset.target_names),
        categorical_cols=to_serializable(categorical_cols),
        category_maps=to_serializable(dataset.category_maps or {}),
        target_category_maps=to_serializable(dataset.target_category_maps or {}),
        metadata=model_metadata(optimizer.bo),
        cross_validation=to_serializable(optimizer.cross_validation_result_),
    )


@router.post("", response_model=TabularModelFitResponse)
def fit_tabular_model(
    request: TabularFitModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Fit a tabular optimizer from JSON records or column-oriented data."""

    try:
        frame = _to_dataframe(request.data)
        cv_config = _schema_dict(request.cv_config)
        if cv_config and cv_config.get("splitter") == "stratified_kfold":
            cv_config["splitter"] = "stratified"
        if cv_config and cv_config.get("splitter") != "loo" and int(cv_config["n_splits"]) > len(frame):
            raise ValueError("n_splits must not exceed the number of data rows.")
        task_type = str(request.bo_model_config.task_type)
        if cv_config and cv_config.get("splitter") != "loo" and task_type in {"binary", "multiclass", "ordinal"}:
            target = request.target_cols[0] if isinstance(request.target_cols, list) else request.target_cols
            if int(cv_config["n_splits"]) > int(frame[target].value_counts().min()):
                raise ValueError("n_splits must not exceed the smallest target class count.")
        direct_model_kwargs: dict[str, Any] = {}
        if request.multi_output_config is not None:
            direct_model_kwargs["multi_output_config"] = _schema_dict(
                request.multi_output_config
            )
        optimizer = TabularBayesianOptimizer(
            model_config=_schema_dict(request.bo_model_config),
            fit_config=_schema_dict(request.fit_config),
            input_cols=request.input_cols,
            target_cols=request.target_cols,
            categorical_cols=request.categorical_cols,
            target_categorical_cols=request.target_categorical_cols,
            bounds=request.bounds,
            dtype=request.dtype,
            device=request.device,
            dropna=request.dropna,
            missing_strategy=request.missing_strategy,
            continuous_impute_strategy=request.continuous_impute_strategy,
            categorical_impute_strategy=request.categorical_impute_strategy,
            impute_targets=request.impute_targets,
            impute_random_state=request.impute_random_state,
            impute_max_iter=request.impute_max_iter,
            multiple_impute_sample_posterior=request.multiple_impute_sample_posterior,
            encode_categories=request.encode_categories,
            category_maps=request.category_maps,
            target_category_maps=request.target_category_maps,
            return_original_categories=request.return_original_categories,
            cross_validation=request.cross_validation,
            cv_config=cv_config,
            **direct_model_kwargs,
        )
        optimizer.fit(frame)
        model_id = store.add(optimizer)
        return _fit_response(model_id, optimizer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ModelListResponse)
def list_tabular_models(
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> ModelListResponse:
    return ModelListResponse(model_ids=store.list_ids())


@router.post("/{model_id}/predict", response_model=TabularPredictResponse)
def predict_tabular_model(
    model_id: str,
    request: TabularPredictRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularPredictResponse:
    try:
        optimizer = store.get(model_id)
        frame = _to_dataframe(request.data)
        value = optimizer.predict(
            frame,
            return_type=request.return_type,
            include_input=request.include_input,
            posterior_kwargs=request.posterior_kwargs,
        )
        if hasattr(value, "columns") and hasattr(value, "to_json"):
            columns, records = _frame_records(value)
            return TabularPredictResponse(
                model_id=model_id,
                columns=columns,
                records=records,
            )
        return TabularPredictResponse(model_id=model_id, value=to_serializable(value))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _generate_candidates(
    model_id: str,
    request: TabularCandidateRequest,
    store: TabularOptimizerStore,
    *,
    use_ask: bool = False,
) -> TabularCandidateResponse:
    optimizer = store.get(model_id)
    method = optimizer.ask if use_ask else optimizer.candidate
    candidates, acq_value = method(
        acq_config=request.acq_config,
        opt_config=_candidate_optimize_config(request),
        bounds=request.bounds,
        return_dataframe=True,
        **_candidate_direct_kwargs(request),
    )
    columns, records = _frame_records(candidates)
    return TabularCandidateResponse(
        model_id=model_id,
        columns=columns,
        candidates=records,
        acq_value=to_serializable(acq_value),
    )


@router.post("/{model_id}/candidates", response_model=TabularCandidateResponse)
def generate_tabular_candidates(
    model_id: str,
    request: TabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    try:
        return _generate_candidates(model_id, request, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/ask", response_model=TabularCandidateResponse)
def ask_tabular_candidates(
    model_id: str,
    request: TabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    try:
        return _generate_candidates(model_id, request, store, use_ask=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
def delete_tabular_model(
    model_id: str,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> ModelDeleteResponse:
    try:
        store.delete(model_id)
        return ModelDeleteResponse(model_id=model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
