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


class AcquisitionNamesResponse(BaseModel):
    names: list[str]
