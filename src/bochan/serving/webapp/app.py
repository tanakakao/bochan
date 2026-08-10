"""React-oriented FastAPI application for the bochan web workbench."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from bochan.api.model_capabilities import BETA_MODEL_TYPES
from bochan.desktop.services import (
    DatasetStore,
    build_dataset_record,
    dataframe_preview,
    load_dataframe_from_payload,
)
from bochan.serving.fastapi import create_api_router
from bochan.serving.fastapi.converters import to_serializable
from bochan.serving.fastapi.schemas.tabular import (
    FeatureImportanceConfigRequest,
    FeatureImportanceVisualizationRequest,
)

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
from .model_artifact_routes import create_model_artifact_router
from .visualization_sessions import build_visualization, get_visualization_session
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


class OutcomeConstraintSchema(_Schema):
    """Threshold constraint on one selected target in the original value scale."""

    id: str | None = None
    target: str
    operator: Literal["<=", ">="]
    value: float


class WebFeatureImportanceSettingsSchema(_Schema):
    """Optional feature-importance execution and presentation settings."""

    enabled: bool = False
    source: Literal["auto", "training", "cross_validation"] = "auto"
    config: FeatureImportanceConfigRequest = Field(default_factory=FeatureImportanceConfigRequest)
    visualization: FeatureImportanceVisualizationRequest = Field(default_factory=FeatureImportanceVisualizationRequest)


class AcquisitionSettingsSchema(_Schema):
    """Acquisition-function settings exposed by the web workbench."""

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
    minimum_candidate_distance_ratio: float = Field(default=1e-3, ge=0.0, le=1.0)


class KSparseSettingsSchema(_Schema):
    """Limit the number of non-zero variables selected from a feature subset."""

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    k: int = Field(default=1, ge=1)
    score: Literal["abs", "value"] = "abs"
    support_selection: str = "topk"
    final_priority: str = "grid"


class VisualizationRequestSchema(_Schema):
    """Select one existing Plotly visualization for a fitted Web run."""

    kind: Literal["yyplot", "target_relation", "pareto", "1d", "2d", "ternary"]
    target: str | None = None
    target_x: str | None = None
    target_y: str | None = None
    show_pareto_front: bool = False
    features: list[str] = Field(default_factory=list)
    fixed_values: dict[str, Any] = Field(default_factory=dict)
    show_type: Literal["pred", "acqf"] = "pred"
    n: int = Field(default=50, ge=10, le=150)
    sum_value: float | None = None


class RegressionRunRequest(_Schema):
    """Run single- or multi-objective regression optimization."""

    dataset_id: str
    feature_columns: list[str]
    target_column: str | None = None
    target_columns: list[str] = Field(default_factory=list)
    direction: Literal["maximize", "minimize"] = "maximize"
    directions: dict[str, Literal["maximize", "minimize"]] = Field(default_factory=dict)
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
    outcome_constraints: list[OutcomeConstraintSchema] = Field(default_factory=list)
    k_sparse: KSparseSettingsSchema | None = None
    acquisition: AcquisitionSettingsSchema = Field(default_factory=AcquisitionSettingsSchema)
    optimizer: OptimizerSettingsSchema = Field(default_factory=OptimizerSettingsSchema)
    drop_missing: bool = True
    cross_validation: bool = False
    cv_config: dict[str, Any] | None = None
    feature_importance: WebFeatureImportanceSettingsSchema | None = None


WEB_CAPABILITIES: dict[str, Any] = {
    "task_types": ["regression", "classification", "ordinal", "hybrid", "multi_objective"],
    "model_types": [
        "base",
        "deepgp",
        "deepkernel",
        "saas",
        "pca",
        "rembo",
        "robust",
        "hetero",
        "random_forest",
        "lightgbm_ensemble",
        "ngboost_ensemble",
        "tabpfn",
        "multitask",
    ],
    "gamma_model_types": [
        "gamma_base",
        "gamma_deepgp",
        "gamma_deepkernel",
        "gamma_saas",
        "gamma_pca",
        "gamma_rembo",
        "gamma_rrp",
        "gamma_hetero",
        "gamma_multitask",
    ],
    "beta_model_types": list(BETA_MODEL_TYPES),
    "acquisitions": [
        "EI",
        "PI",
        "UCB",
        "EHVI",
        "NEHVI",
        "NParEGO",
        "variance",
        "predictive_entropy",
        "BALD",
        "NIPV",
        "straddle",
        "boundary_variance",
        "ICU",
    ],
    "optimizers": [
        "optimize_acqf",
        "torch",
        "ga",
        "sa",
        "pso",
        "cmaes",
        "thompson_sampling",
        "nsgaii",
    ],
    "data_sources": ["csv", "excel", "model_artifact"],
    "visualizations": ["yyplot", "target_relation", "pareto", "prediction-1d", "prediction-2d", "ternary"],
    "model_artifacts": {
        "download_endpoint": "/api/v1/runs/{run_id}/model-artifact",
        "import_endpoint": "/api/v1/model-artifacts/import",
        "format": "bochan.pt",
        "pickle_trust_required": True,
    },
    "logging": {
        "format": "jsonl",
        "request_id_header": "X-Request-ID",
        "recent_logs_endpoint": "/api/v1/logs",
    },
}


def _profile_with_category_values(record: Any) -> dict[str, Any]:
    """Add complete low-cardinality values, including numeric categories, for UI selects."""

    profile = {
        **record.profile,
        "columns": [dict(column) for column in record.profile["columns"]],
    }
    for column in profile["columns"]:
        if int(column.get("unique_count", 0)) > 30:
            continue
        series = record.data[column["name"]].dropna()
        column["values"] = to_serializable(series.unique().tolist())
    return profile


def create_app(
    *,
    title: str = "bochan Web API",
    version: str = "0.3.0",
    api_prefix: str = "/api/v1",
    cors_origins: Sequence[str] | None = None,
    include_core_api: bool = True,
) -> FastAPI:
    """Create the FastAPI application used by the React workbench."""

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
        expose_headers=["X-Request-ID", "Content-Disposition", "X-Model-Artifact-Version"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any):
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
        app.include_router(create_api_router(prefix=api_prefix))

    dataset_store = DatasetStore()
    router = APIRouter(prefix=api_prefix, tags=["web"])

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        capabilities_payload = dict(WEB_CAPABILITIES)
        logging_payload = dict(WEB_CAPABILITIES["logging"])
        logging_payload["recent_logs_endpoint"] = f"{api_prefix}/logs"
        capabilities_payload["logging"] = logging_payload
        artifact_payload = dict(WEB_CAPABILITIES["model_artifacts"])
        artifact_payload["download_endpoint"] = f"{api_prefix}/runs/{{run_id}}/model-artifact"
        artifact_payload["import_endpoint"] = f"{api_prefix}/model-artifacts/import"
        capabilities_payload["model_artifacts"] = artifact_payload
        return capabilities_payload

    @router.get("/logs")
    def recent_logs(
        limit: int = 200,
        level: str | None = None,
        event: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
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
        return {"datasets": dataset_store.list()}

    @router.post("/datasets")
    def load_dataset(request: DatasetLoadRequest) -> dict[str, Any]:
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
                "profile": _profile_with_category_values(record),
                "preview": dataframe_preview(record.data, limit=100),
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
        try:
            record = dataset_store.get(dataset_id)
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": _profile_with_category_values(record),
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

    @router.post("/runs/{run_id}/visualizations")
    def result_visualization(
        run_id: str,
        request: VisualizationRequestSchema,
    ) -> dict[str, Any]:
        try:
            get_visualization_session(run_id)
        except KeyError as exc:
            log_event(
                logger,
                logging.WARNING,
                "visualization_session_not_found",
                "Visualization session was not found",
                visualization_run_id=run_id,
                visualization_kind=request.kind,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            return build_visualization(run_id, request.model_dump())
        except Exception as exc:
            logger.exception(
                "Result visualization failed",
                extra={
                    "event": "visualization_failed",
                    "visualization_run_id": run_id,
                    "visualization_kind": request.kind,
                },
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    router.include_router(create_model_artifact_router(dataset_store))
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
    "KSparseSettingsSchema",
    "OutcomeConstraintSchema",
    "RegressionRunRequest",
    "VisualizationRequestSchema",
    "WEB_CAPABILITIES",
    "app",
    "create_app",
]
