"""Regression workflow routes for the Web API."""

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException

from bochan.api.progress import progress_reporting
from bochan.serving.fastapi.converters import to_serializable

from ..logging import log_event
from ..schemas.regression import RegressionRunRequest
from ..services.model_reuse import current_model_reuse_state
from ..workflows import run_regression_web_workflow

_CRABNET_MULTITASK_WEB_BASE_MODELS = {
    "crabnet_multitask": "crabnet_gp",
    "crabnet_multitask_dkl": "crabnet_dkl",
    "crabnet_mixed_multitask": "crabnet_mixed_gp",
    "crabnet_mixed_multitask_dkl": "crabnet_mixed_dkl",
}
_WEB_CRABNET_MULTITASK_MODEL_KEY = "web_correlated_crabnet_model_type"

_PROGRESS_MESSAGES = {
    "model_fit_started": "Model fitting started",
    "model_fit_completed": "Model fitting completed",
    "model_fit_failed": "Model fitting failed",
    "model_output_fit_started": "Output model fitting started",
    "model_output_fit_completed": "Output model fitting completed",
    "model_output_fit_failed": "Output model fitting failed",
    "candidate_generation_started": "Candidate generation started",
    "candidate_generation_completed": "Candidate generation completed",
    "candidate_generation_failed": "Candidate generation failed",
}


def _workflow_request(request: RegressionRunRequest) -> RegressionRunRequest:
    """Use existing CrabNet Web validation while preserving correlated model intent."""

    model_type = str(request.model_type).lower()
    base_model_type = _CRABNET_MULTITASK_WEB_BASE_MODELS.get(model_type)
    if base_model_type is None:
        return request
    model_kwargs = dict(request.model_kwargs or {})
    model_kwargs[_WEB_CRABNET_MULTITASK_MODEL_KEY] = model_type
    return request.model_copy(
        update={
            "model_type": base_model_type,
            "model_kwargs": model_kwargs,
        }
    )


def _cv_fold_count(request: RegressionRunRequest) -> int | None:
    """Return configured K-fold count; LOO stays unknown until preprocessing.

    K-fold and stratified splitters always execute the configured ``n_splits``
    when validation succeeds. Leave-one-out depends on the effective row count
    after Web missing-value preprocessing, so the progress layer deliberately
    avoids inventing a denominator for LOO.
    """

    if not bool(request.cross_validation):
        return 0
    config = request.cv_config
    if hasattr(config, "model_dump"):
        values = config.model_dump(exclude_none=True)
    elif isinstance(config, dict):
        values = dict(config)
    else:
        values = {}
    splitter = str(values.get("splitter", "auto")).lower().replace("-", "_")
    if splitter in {"loo", "leave_one_out"}:
        return None
    try:
        return max(2, int(values.get("n_splits", 5)))
    except (TypeError, ValueError):
        return 5


def _progress_callback(
    request: RegressionRunRequest,
    *,
    logger: Any,
    workflow_started: float,
):
    """Translate request-local core progress into structured Web log events."""

    cv_enabled = bool(request.cross_validation)
    cv_total = _cv_fold_count(request)
    state = {
        "fit_cycle": 0,
        "prepare_logged": False,
        "reuse_logged": False,
    }

    def log_prepared() -> None:
        if state["prepare_logged"]:
            return
        state["prepare_logged"] = True
        log_event(
            logger,
            logging.INFO,
            "workflow_data_prepared",
            "Data and model configuration prepared",
            duration_ms=round((perf_counter() - workflow_started) * 1000, 3),
        )

    def callback(event: str, payload: Any) -> None:
        fields = dict(payload or {})
        if event == "model_fit_started":
            log_prepared()
            state["fit_cycle"] += 1
        elif event == "candidate_generation_started":
            log_prepared()
            reuse_state = current_model_reuse_state() or {}
            if bool(reuse_state.get("fit_skipped")) and not state["reuse_logged"]:
                state["reuse_logged"] = True
                log_event(
                    logger,
                    logging.INFO,
                    "model_reuse_completed",
                    "Fitted model reused; model fitting skipped",
                    source_run_id=reuse_state.get("source_run_id"),
                )

        if event.startswith("model_") and event != "model_reuse_completed":
            cycle = int(state["fit_cycle"])
            if cv_enabled and cv_total is None:
                fields.update(
                    {
                        "fit_phase": "cross_validation_or_final",
                        "fold_current": None,
                        "fold_total": None,
                    }
                )
            elif cv_total and 1 <= cycle <= cv_total:
                fields.update(
                    {
                        "fit_phase": "cross_validation",
                        "fold_current": cycle,
                        "fold_total": cv_total,
                    }
                )
            elif cycle > 0:
                fields.update(
                    {
                        "fit_phase": "final",
                        "fold_current": None,
                        "fold_total": cv_total or None,
                    }
                )

        message = _PROGRESS_MESSAGES.get(event)
        if message is None:
            return
        level = logging.ERROR if event.endswith("_failed") else logging.INFO
        log_event(logger, level, event, message, **fields)

    return callback


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
    log_event(
        logger,
        logging.INFO,
        "workflow_started",
        "Regression workflow started",
        dataset_id=request.dataset_id,
        target_columns=target_columns,
    )
    progress_callback = _progress_callback(
        request,
        logger=logger,
        workflow_started=started,
    )
    try:
        with progress_reporting(progress_callback):
            result = run_regression_web_workflow(
                _workflow_request(request),
                dataset_store,
            )
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
