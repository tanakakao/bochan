"""Request schemas for bochan FastAPI serving."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .configs import AcquisitionConfigSchema, DataContextSchema, FitConfigSchema, ModelConfigSchema, OptimizeConfigSchema


class APIRequest(BaseModel):
    """Base request model with API-compatible aliases enabled."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class TensorOptionsSchema(APIRequest):
    """Tensor conversion options for JSON payloads."""

    dtype: str = "float64"
    device: str | None = None


class FitModelRequest(APIRequest):
    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    train_X: Any
    train_Y: Any
    bounds: Any | None = None
    fit_config: FitConfigSchema | None = None
    data_context: DataContextSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class RefitModelRequest(APIRequest):
    """Request body for refitting an existing in-memory optimizer."""

    fit_config: FitConfigSchema | None = None


class TellRequest(APIRequest):
    """Append observations to an existing optimizer and optionally refit."""

    new_X: Any
    new_Y: Any
    refit: bool = True
    fit_config: FitConfigSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class PredictRequest(APIRequest):
    X: Any
    return_type: Literal["posterior", "mean", "variance", "mean_variance"] = "mean_variance"
    posterior_kwargs: dict[str, Any] = Field(default_factory=dict)
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class CandidateRequest(APIRequest):
    acq_config: AcquisitionConfigSchema = Field(alias="acquisition_config")
    opt_config: OptimizeConfigSchema = Field(default_factory=OptimizeConfigSchema, alias="optimize_config")
    data_context: DataContextSchema | None = None
    bounds: Any | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class CompareCandidatesRequest(APIRequest):
    """Generate candidates for multiple acquisition functions on the same fitted model."""

    acq_configs: list[AcquisitionConfigSchema] = Field(alias="acquisition_configs")
    opt_config: OptimizeConfigSchema = Field(default_factory=OptimizeConfigSchema, alias="optimize_config")
    data_context: DataContextSchema | None = None
    bounds: Any | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)
