"""React-oriented FastAPI application for the bochan web MVP."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from bochan.desktop.services import (
    DatasetStore,
    build_dataset_record,
    dataframe_preview,
    load_dataframe_from_payload,
)
from bochan.serving.fastapi.routers import acquisitions, candidates, health, models, predictions

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
}


def create_app(*, title: str = "bochan Web API", version: str = "0.1.0") -> FastAPI:
    """Create the API used by the React web application.

    The first release keeps datasets and fitted models in the FastAPI process.
    The API is therefore intended for local use and single-process deployments.
    """

    app = FastAPI(title=title, version=version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Preserve the existing tensor-oriented HTTP API.
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predictions.router)
    app.include_router(candidates.router)
    app.include_router(acquisitions.router)

    dataset_store = DatasetStore()
    router = APIRouter(prefix="/api/v1", tags=["web"])

    @router.get("/health")
    def web_health() -> dict[str, str]:
        return {"status": "ok", "application": "bochan-web"}

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return WEB_CAPABILITIES

    @router.get("/datasets")
    def list_datasets() -> dict[str, Any]:
        return {"datasets": dataset_store.list()}

    @router.post("/datasets")
    def load_dataset(request: DatasetLoadRequest) -> dict[str, Any]:
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
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": record.profile,
                "preview": dataframe_preview(record.data, limit=50),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, limit: int = 100) -> dict[str, Any]:
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
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/regression/run")
    def run_regression(request: RegressionRunRequest) -> dict[str, Any]:
        try:
            return run_regression_web_workflow(request, dataset_store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.include_router(router)
    return app


app = create_app()


__all__ = [
    "DatasetLoadRequest",
    "RegressionRunRequest",
    "WEB_CAPABILITIES",
    "app",
    "create_app",
]
