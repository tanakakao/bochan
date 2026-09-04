"""Pydantic schemas for material MLIP workflow validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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


class MaterialWorkflowValidationResponse(BaseModel):
    """Validated workflow identity and required runtime inputs."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    spec: MaterialWorkflowSpecResponse
    requirements: list[str]


__all__ = [
    "MaterialWorkflowSpecRequest",
    "MaterialWorkflowSpecResponse",
    "MaterialWorkflowValidationResponse",
]
