"""Pydantic schemas for material MLIP workflow validation, configuration, and execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MaterialWorkflowSpecRequest(BaseModel):
    """User-facing MLIP workflow identity before canonical normalization."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    quantity: str
    model_mode: str
    workflow_mode: str = "model_only"


class MaterialWorkflowSpecResponse(BaseModel):
    """Canonical MLIP workflow identity returned by the API."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    quantity: str
    model_mode: str
    workflow_mode: str


class MaterialRelaxationConfig(BaseModel):
    """Backend-independent ASE relaxation execution settings."""

    model_config = ConfigDict(extra="forbid")

    optimizer: Literal["FIRE", "BFGS", "LBFGS"] = "FIRE"
    fmax: float = Field(default=0.05, gt=0)
    max_steps: int = Field(default=200, gt=0)
    relax_cell: bool = False


class MaterialWorkflowConfigRequest(MaterialWorkflowSpecRequest):
    """MLIP workflow identity plus optional relaxation execution settings."""

    relaxation: MaterialRelaxationConfig | None = None


class MaterialWorkflowValidationResponse(BaseModel):
    """Validated workflow identity and required runtime inputs."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    spec: MaterialWorkflowSpecResponse
    requirements: list[str]


class MaterialWorkflowConfigResponse(MaterialWorkflowValidationResponse):
    """Canonical workflow configuration ready for runtime execution."""

    relaxation: MaterialRelaxationConfig | None


class MaterialRelaxationExecutionRequest(MaterialWorkflowConfigRequest):
    """Execute the relaxation stage for one configured material workflow."""

    structures: list[dict[str, Any]] = Field(min_length=1, max_length=100)
    backend_options: dict[str, Any] = Field(default_factory=dict)


class MaterialRelaxationExecutionResponse(BaseModel):
    """Serializable results produced by the common MLIP relaxation runtime."""

    model_config = ConfigDict(extra="forbid")

    spec: MaterialWorkflowSpecResponse
    relaxation: MaterialRelaxationConfig
    backend_options: dict[str, Any]
    results: list[dict[str, Any]]


__all__ = [
    "MaterialRelaxationConfig",
    "MaterialRelaxationExecutionRequest",
    "MaterialRelaxationExecutionResponse",
    "MaterialWorkflowConfigRequest",
    "MaterialWorkflowConfigResponse",
    "MaterialWorkflowSpecRequest",
    "MaterialWorkflowSpecResponse",
    "MaterialWorkflowValidationResponse",
]
