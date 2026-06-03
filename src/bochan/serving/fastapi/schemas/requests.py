"""Request schemas for bochan FastAPI serving."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .configs import AcquisitionConfigSchema, DataContextSchema, FitConfigSchema, ModelConfigSchema, OptimizeConfigSchema


class FitModelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    train_X: Any
    train_Y: Any
    bounds: Any | None = None
    fit_config: FitConfigSchema | None = None
    data_context: DataContextSchema | None = None


class PredictRequest(BaseModel):
    X: Any
    return_type: Literal["mean", "variance", "mean_variance"] = "mean_variance"
    posterior_kwargs: dict[str, Any] = Field(default_factory=dict)


class CandidateRequest(BaseModel):
    acq_config: AcquisitionConfigSchema
    opt_config: OptimizeConfigSchema = Field(default_factory=OptimizeConfigSchema)
    data_context: DataContextSchema | None = None
    bounds: Any | None = None
