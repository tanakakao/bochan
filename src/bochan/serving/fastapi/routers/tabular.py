"""Batch candidate generation endpoints for column-oriented tabular data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from bochan.tabular import TabularBayesianOptimizer

from ..converters import to_serializable
from ..schemas import (
    TabularBatchCandidateRequest,
    TabularBatchCandidateResponse,
    TabularBatchCandidateResult,
)

router = APIRouter(prefix="/tabular", tags=["tabular"])


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats and recursively normalize JSON containers."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _serialize_candidates(value: Any) -> Any:
    """Convert tabular candidate outputs to JSON-compatible records."""
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    elif isinstance(value, pd.Series):
        value = value.tolist()
    return _json_safe(to_serializable(value))


def _serialize_value(value: Any) -> Any:
    """Convert acquisition values to JSON-compatible objects."""
    return _json_safe(to_serializable(value))


def _validate_request(request: TabularBatchCandidateRequest, df: pd.DataFrame) -> None:
    """Validate required columns and non-empty execution settings."""
    if df.empty:
        raise ValueError("data must contain at least one row.")

    required_columns = list(dict.fromkeys([*request.input_cols, *request.target_cols]))
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}.")

    if not request.model_types:
        raise ValueError("model_types must contain at least one model type.")
    if not request.acquisition_names:
        raise ValueError("acquisition_names must contain at least one acquisition name.")
    if not request.optimizers:
        non_nsgaii = [name for name in request.acquisition_names if name.lower() != "nsgaii"]
        if non_nsgaii:
            raise ValueError(
                "optimizers must contain at least one optimizer when non-NSGA-II "
                "acquisitions are requested."
            )


def _candidate_opt_configs(
    request: TabularBatchCandidateRequest,
    acquisition_name: str,
) -> list[tuple[str | None, dict[str, Any]]]:
    """Build optimizer configurations for one acquisition function."""
    base_config = dict(request.optimize_config)
    if acquisition_name.lower() == "nsgaii":
        base_config.pop("optimizer", None)
        return [(None, base_config)]

    return [
        (optimizer_name, {**base_config, "optimizer": optimizer_name})
        for optimizer_name in request.optimizers
    ]


def _run_batch(request: TabularBatchCandidateRequest) -> TabularBatchCandidateResponse:
    """Fit every requested tabular model and generate all candidate combinations."""
    df = pd.DataFrame.from_records(request.data)
    _validate_request(request, df)

    results: list[TabularBatchCandidateResult] = []

    for model_type in request.model_types:
        model_config = dict(request.bo_model_config)
        model_config.setdefault("task_type", "regression")
        model_config["model_type"] = model_type

        try:
            optimizer = TabularBayesianOptimizer(
                model_config=model_config,
                fit_config=dict(request.fit_config),
                input_cols=list(request.input_cols),
                target_cols=list(request.target_cols),
            )
            optimizer.fit(df)
        except Exception as exc:
            if not request.continue_on_error:
                raise
            results.append(
                TabularBatchCandidateResult(
                    model_type=model_type,
                    stage="fit",
                    status="error",
                    error=str(exc),
                )
            )
            continue

        for acquisition_name in request.acquisition_names:
            for optimizer_name, opt_config in _candidate_opt_configs(
                request,
                acquisition_name,
            ):
                try:
                    candidates, acq_value = optimizer.candidate(
                        acq_config={"name": acquisition_name},
                        opt_config=opt_config,
                    )
                    results.append(
                        TabularBatchCandidateResult(
                            model_type=model_type,
                            acquisition_name=acquisition_name,
                            optimizer=optimizer_name,
                            stage="candidate",
                            status="ok",
                            candidates=_serialize_candidates(candidates),
                            acq_value=_serialize_value(acq_value),
                        )
                    )
                except Exception as exc:
                    if not request.continue_on_error:
                        raise
                    results.append(
                        TabularBatchCandidateResult(
                            model_type=model_type,
                            acquisition_name=acquisition_name,
                            optimizer=optimizer_name,
                            stage="candidate",
                            status="error",
                            error=str(exc),
                        )
                    )

    n_success = sum(result.status == "ok" for result in results)
    return TabularBatchCandidateResponse(
        n_models=len(request.model_types),
        n_runs=len(results),
        n_success=n_success,
        n_failed=len(results) - n_success,
        results=results,
    )


@router.post("/batch-candidates", response_model=TabularBatchCandidateResponse)
async def generate_tabular_batch_candidates(
    request: TabularBatchCandidateRequest,
) -> TabularBatchCandidateResponse:
    """Run the requested model, acquisition, and optimizer matrix.

    The numerical work is executed in a worker thread so it does not block the
    FastAPI event loop. This endpoint is still request/response based; production
    deployments should move very large matrices to a durable job queue.
    """
    try:
        return await run_in_threadpool(_run_batch, request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
