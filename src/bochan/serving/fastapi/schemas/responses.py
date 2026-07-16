"""Response schemas for bochan FastAPI serving."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelFitResponse(BaseModel):
    model_id: str
    task_type: str
    model_type: str
    n_train: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    model_ids: list[str]


class ModelDeleteResponse(BaseModel):
    model_id: str
    deleted: bool = True


class PredictResponse(BaseModel):
    model_id: str
    task_type: str | None = None
    prediction_space: str | None = None
    variance_kind: str | None = None
    posterior: Any | None = None
    mean: Any | None = None
    variance: Any | None = None
    value: Any | None = None


class CandidateResponse(BaseModel):
    model_id: str
    candidates: Any
    acq_value: Any


class CompareCandidatesResponse(BaseModel):
    model_id: str
    results: dict[str, CandidateResponse]


class TabularBatchCandidateResult(BaseModel):
    """Result for one fit stage or candidate-generation combination."""

    model_type: str
    acquisition_name: str | None = None
    optimizer: str | None = None
    stage: Literal["fit", "candidate"]
    status: Literal["ok", "error"]
    candidates: Any | None = None
    acq_value: Any | None = None
    error: str | None = None


class TabularBatchCandidateResponse(BaseModel):
    """Summary of a tabular model/acquisition/optimizer matrix run."""

    n_models: int
    n_runs: int
    n_success: int
    n_failed: int
    results: list[TabularBatchCandidateResult] = Field(default_factory=list)


class AcquisitionNamesResponse(BaseModel):
    names: list[str]


class SavedModelsResponse(BaseModel):
    """Response listing saved optimizer artifact files."""

    root_dir: str
    filenames: list[str]


class SaveModelResponse(BaseModel):
    """Response returned after saving an optimizer artifact."""

    model_id: str
    filename: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadModelResponse(ModelFitResponse):
    """Response returned after loading an optimizer artifact."""

    filename: str
    path: str
