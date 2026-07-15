"""React-oriented FastAPI application for the bochan web MVP."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from bochan.desktop.services import (
    DatasetStore,
    build_dataset_record,
    dataframe_preview,
    load_dataframe_from_payload,
)
from bochan.serving.fastapi import create_api_router

from .logging import (
    configure_logging,
    current_request_id,
    get_logger,
    log_event,
    log_file_path,
    read_recent_logs,
    reset_request_id,
    set_request_id,
)
from .workflows import run_regression_web_workflow


class _Schema(BaseModel):
    """Base request schema used by the web application API."""

    model_config = ConfigDict(extra="forbid")


class DatasetLoadRequest(_Schema):
    """Browser-uploaded tabular dataset encoded as base64."""

    source_type: Literal["csv", "excel"] = "csv"
    name: str | None = None
    content_base64: str
    encoding: str = "utf-8-sig"
    sep: str | None = None
    sheet_name: str | int | None = 0


class SearchVariableSchema(_Schema):
    """Search-space settings for one feature column."""

    name: str
    type: Literal["auto", "numeric", "categorical"] = "auto"
    lower: float | None = None
    upper: float | None = None
    step: float | None = None
    fixed: bool = False
    fixed_value: Any | None = None
    categories: list[Any] | None = None


class AcquisitionSettingsSchema(_Schema):
    """Acquisition-function settings supported by the first web MVP."""

    name: str = "EI"
    beta: float = 2.0
    acqf_kwargs: dict[str, Any] = Field(default_factory=dict)


class OptimizerSettingsSchema(_Schema):
    """Candidate optimizer settings."""

    name: str = "optimize_acqf"
    q: int = Field(default=1, ge=1)
    num_restarts: int = Field(default=10, ge=1)
    raw_samples: int = Field(default=256, ge=1)
    sequential: bool = True


class RegressionRunRequest(_Schema):
    """Run one single-objective regression optimization workflow."""

    dataset_id: str
    feature_columns: list[str]
    target_column: str
    direction: Literal["maximize", "minimize"] = "maximize"
    model_type: str = "base"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    fit_maxiter: int = Field(default=128, ge=1)
    normalize: bool = True
    outcome_transform: bool = True
    input_perturbation: bool = False
    n_w: int = Field(default=16, ge=1)
    perturbation_std: float = Field(default=0.1, gt=0.0)
    search_space: list[SearchVariableSchema] = Field(default_factory=list)
    constraints: list[Any] = Field(default_factory=list)
    k_sparse: Any | None = None
    acquisition: AcquisitionSettingsSchema = Field(default_factory=AcquisitionSettingsSchema)
    optimizer: OptimizerSettingsSchema = Field(default_factory=OptimizerSettingsSchema)
    drop_missing: bool = True


WEB_CAPABILITIES: dict[str, Any] = {
    "task_types": ["regression"],
    "model_types": ["base", "saas", "deepkernel"],
    "acquisitions": ["EI", "NEI", "UCB"],
    "optimizers": ["optimize_acqf"],
    "data_sources": ["csv", "excel"],
    "visualizations": ["yyplot", "prediction-1d", "prediction-2d"],
    "logging": {
        "format": "jsonl",
        "request_id_header": "X-Request-ID",
        "recent_logs_endpoint": "/api/v1/logs",
    },
}


def create_app(
    *,
    title: str = "bochan Web API",
    version: str = "0.1.0",
    api_prefix: str = "/api/v1",
    cors_origins: Sequence[str] | None = None,
    include_core_api: bool = True,
) -> FastAPI:
    """Create the FastAPI app used by the React web application.

    Args:
        title: OpenAPI application title.
        version: OpenAPI application version.
        api_prefix: Path prefix shared by the core FastAPI router and the
            web-specific endpoints.
        cors_origins: Browser origins allowed to call the web API. When
            omitted, Vite's localhost origins are allowed.
        include_core_api: Whether to mount the tensor-oriented bochan FastAPI
            router next to the web-specific endpoints.

    Returns:
        Configured FastAPI application for the React web interface.
    """

    configured_log_path = configure_logging()
    logger = get_logger("api")
    app = FastAPI(title=title, version=version)
    allowed_origins = list(cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any):
        """Log request lifecycle events and propagate the request id.

        Args:
            request: Incoming FastAPI request.
            call_next: ASGI callable that dispatches to the next middleware or
                route handler.

        Returns:
            FastAPI response enriched with the ``X-Request-ID`` header.
        """
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        token = set_request_id(request_id)
        started = perf_counter()
        log_event(
            logger,
            logging.INFO,
            "http_request_started",
            "HTTP request started",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            client=request.client.host if request.client else None,
        )
        try:
            response = await call_next(request)
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "http_request_failed",
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                exc_info=True,
            )
            raise
        else:
            duration_ms = round((perf_counter() - started) * 1000, 3)
            response.headers["X-Request-ID"] = request_id
            response_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            log_event(
                logger,
                response_level,
                "http_request_completed",
                "HTTP request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        finally:
            reset_request_id(token)

    if include_core_api:
        # Preserve the existing tensor-oriented HTTP API under the same prefix
        # as the React web endpoints.
        app.include_router(create_api_router(prefix=api_prefix))

    dataset_store = DatasetStore()
    router = APIRouter(prefix=api_prefix, tags=["web"])

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        """Return web-client capabilities advertised by this FastAPI app.

        Returns:
            Supported task types, model types, acquisitions, optimizers, data
            sources, visualization ids, and logging metadata.
        """
        capabilities_payload = dict(WEB_CAPABILITIES)
        logging_payload = dict(WEB_CAPABILITIES["logging"])
        logging_payload["recent_logs_endpoint"] = f"{api_prefix}/logs"
        capabilities_payload["logging"] = logging_payload
        return capabilities_payload

    @router.get("/logs")
    def recent_logs(
        limit: int = 200,
        level: str | None = None,
        event: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Return recent structured web API log entries.

        Args:
            limit: Maximum number of recent records to return.
            level: Optional log-level filter.
            event: Optional structured event-name filter.
            request_id: Optional request-id filter.

        Returns:
            JSON-safe log entries, entry count, and backing log file path.
        """
        entries = read_recent_logs(
            limit=limit,
            level=level,
            event=event,
            request_id=request_id,
        )
        return {
            "entries": entries,
            "count": len(entries),
            "log_file": str(log_file_path()),
        }

    @router.get("/datasets")
    def list_datasets() -> dict[str, Any]:
        """List datasets currently loaded in the FastAPI process.

        Returns:
            Dataset metadata records without tabular preview rows.
        """
        return {"datasets": dataset_store.list()}

    @router.post("/datasets")
    def load_dataset(request: DatasetLoadRequest) -> dict[str, Any]:
        """Load a browser-uploaded dataset into the in-memory store.

        Args:
            request: Base64-encoded CSV or Excel dataset payload.

        Returns:
            Loaded dataset id, metadata profile, and preview rows.
        """
        started = perf_counter()
        log_event(
            logger,
            logging.INFO,
            "dataset_load_started",
            "Dataset loading started",
            dataset_name=request.name,
            source_type=request.source_type,
        )
        try:
            data, metadata = load_dataframe_from_payload(
                source_type=request.source_type,
                content_base64=request.content_base64,
                name=request.name,
                encoding=request.encoding,
                sep=request.sep,
                sheet_name=request.sheet_name,
            )
            record = build_dataset_record(
                data=data,
                name=request.name or "dataset",
                source_type=request.source_type,
                metadata=metadata,
            )
            dataset_store.add(record)
            log_event(
                logger,
                logging.INFO,
                "dataset_load_completed",
                "Dataset loading completed",
                dataset_id=record.dataset_id,
                dataset_name=record.name,
                source_type=record.source_type,
                n_rows=record.profile["n_rows"],
                n_columns=record.profile["n_columns"],
                duration_ms=round((perf_counter() - started) * 1000, 3),
            )
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": record.profile,
                "preview": dataframe_preview(record.data, limit=50),
            }
        except Exception as exc:
            logger.exception(
                "Dataset loading failed",
                extra={
                    "event": "dataset_load_failed",
                    "dataset_name": request.name,
                    "source_type": request.source_type,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, limit: int = 100) -> dict[str, Any]:
        """Return one loaded dataset with preview rows.

        Args:
            dataset_id: Identifier returned by ``POST /datasets``.
            limit: Maximum preview rows to include.

        Returns:
            Dataset id, name, source type, profile, and tabular preview.
        """
        try:
            record = dataset_store.get(dataset_id)
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": record.profile,
                "preview": dataframe_preview(record.data, limit=limit),
            }
        except KeyError as exc:
            log_event(
                logger,
                logging.WARNING,
                "dataset_not_found",
                "Dataset was not found",
                dataset_id=dataset_id,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "Dataset retrieval failed",
                extra={"event": "dataset_get_failed", "dataset_id": dataset_id},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/regression/run")
    def run_regression(request: RegressionRunRequest) -> dict[str, Any]:
        """Run a single-objective regression optimization workflow.

        Args:
            request: Dataset id, feature/target columns, model, acquisition,
                optimizer, and visualization settings from the web client.

        Returns:
            Candidate rows, model metadata, visualization payloads, and warnings.
        """
        started = perf_counter()
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
            target_column=request.target_column,
            direction=request.direction,
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
                n_visualizations=len(result.get("visualizations", [])),
                visualization_warnings=len(result.get("visualization_warnings", [])),
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

    app.include_router(router)
    log_event(
        logger,
        logging.INFO,
        "application_configured",
        "bochan web application configured",
        title=title,
        version=version,
        log_file=str(configured_log_path),
        request_id=current_request_id(),
    )
    return app


app = create_app()


__all__ = [
    "DatasetLoadRequest",
    "RegressionRunRequest",
    "WEB_CAPABILITIES",
    "app",
    "create_app",
]
