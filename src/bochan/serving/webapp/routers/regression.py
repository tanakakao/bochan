"""Regression workflow routes for the Web API."""

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException

from bochan.serving.fastapi.converters import to_serializable

from ..logging import log_event
from ..schemas.regression import RegressionRunRequest
from ..workflows import run_regression_web_workflow


def run_regression_request(
    request: RegressionRunRequest,
    *,
    dataset_store: Any,
    logger: Any,
) -> dict[str, Any]:
    """Execute one regression request with Web logging and HTTP error translation."""

    started = perf_counter()
    target_columns = request.target_columns or ([request.target_column] if request.target_column else [])
    log_event(
        logger,
        logging.INFO,
        "regression_run_requested",
        "Regression workflow requested",
        dataset_id=request.dataset_id,
        model_type=request.model_type,
        acquisition=request.acquisition.name,
        optimizer=request.optimizer.name,
        q=request.optimizer.q,
        n_features=len(request.feature_columns),
        target_columns=target_columns,
        directions=request.directions,
        normalize=request.normalize,
        input_perturbation=request.input_perturbation,
        n_w=request.n_w,
        perturbation_std=request.perturbation_std,
        n_feature_constraints=len(request.constraints),
        k_sparse=to_serializable(request.k_sparse),
    )
    try:
        result = run_regression_web_workflow(request, dataset_store)
        log_event(
            logger,
            logging.INFO,
            "regression_run_completed",
            "Regression workflow completed",
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            acquisition=request.acquisition.name,
            n_train=result.get("n_train"),
            n_candidates=len(result.get("candidates", [])),
            model_details=result.get("metadata", {}).get("model_details"),
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        return result
    except KeyError as exc:
        log_event(
            logger,
            logging.WARNING,
            "regression_dataset_not_found",
            "Regression dataset was not found",
            dataset_id=request.dataset_id,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Regression workflow failed",
            extra={
                "event": "regression_run_failed",
                "dataset_id": request.dataset_id,
                "model_type": request.model_type,
                "acquisition": request.acquisition.name,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_regression_router(*, dataset_store: Any, logger: Any) -> APIRouter:
    """Create the Web regression workflow router."""

    router = APIRouter()

    @router.post("/regression/run")
    def run_regression(request: RegressionRunRequest) -> dict[str, Any]:
        return run_regression_request(request, dataset_store=dataset_store, logger=logger)

    return router


__all__ = ["create_regression_router", "run_regression_request"]
