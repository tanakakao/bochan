"""FastAPI schemas for material model-axis discovery and validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MaterialExplicitTaskRequest(BaseModel):
    """Explicit task-index contract used by long-format material observations."""

    task_feature: int = -1
    all_tasks: list[int] | None = None
    output_tasks: list[int] | None = None


class MaterialModelAxesRequest(BaseModel):
    """JSON-facing material Gaussian model-axis specification."""

    family: str
    kind: Literal["gp", "dkl"] | str = "gp"
    input_mode: Literal["continuous", "mixed"] | str = "continuous"
    output_mode: Literal["scalar", "independent", "correlated"] | str = "scalar"
    task_mode: Literal["none", "explicit"] | str = "none"
    fidelity_mode: Literal["none", "continuous"] | str = "none"
    cat_dims: list[int] | None = None
    task: MaterialExplicitTaskRequest | None = None
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class MaterialModelAxesResponse(BaseModel):
    """Normalized material model-axis contract returned by validation."""

    family: str
    domain: str
    kind: str
    input_mode: str
    output_mode: str
    task_mode: str
    fidelity_mode: str
    route: str
    implemented: bool
    cat_dims: list[int]
    task: MaterialExplicitTaskRequest | None = None


class MaterialModelAxesCapabilitiesResponse(BaseModel):
    """Capability response for one registered material family."""

    family: str
    domain: str
    axes: dict[str, list[str]]
    implemented_routes: list[str]
    fidelity_route_implemented: bool
    notes: dict[str, str]


class MaterialModelAxesCatalogResponse(BaseModel):
    """Capability catalog for all registered material families."""

    families: list[MaterialModelAxesCapabilitiesResponse]


class MaterialTaskFixedFeaturesRequest(BaseModel):
    """Request a fixed-feature mapping for explicit-task candidate optimization."""

    model: MaterialModelAxesRequest
    target_task: int
    input_dim: int = Field(gt=0)


class MaterialTaskFixedFeaturesResponse(BaseModel):
    """Fixed features suitable for bochan candidate optimization."""

    fixed_features: dict[int, float]


__all__ = [
    "MaterialExplicitTaskRequest",
    "MaterialModelAxesCapabilitiesResponse",
    "MaterialModelAxesCatalogResponse",
    "MaterialModelAxesRequest",
    "MaterialModelAxesResponse",
    "MaterialTaskFixedFeaturesRequest",
    "MaterialTaskFixedFeaturesResponse",
]
