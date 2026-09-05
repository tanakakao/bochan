"""Pydantic schemas for material MLIP capability discovery."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MaterialBackendCapabilitiesResponse(BaseModel):
    """Capability metadata for one material MLIP backend."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    direct_quantities: list[str]
    residual_quantities: list[str]
    model_modes: list[str]
    residual_input_modes: list[str]
    residual_scalar_quantities: list[str]
    residual_correlated_multioutput_quantities: list[str]
    supports_mixed_residual: bool
    supports_correlated_multioutput_residual: bool
    workflow_modes: list[str]
    supports_relaxation: bool
    supports_relax_rank: bool
    supports_relax_acquisition: bool
    force_fixed_topology: bool
    stress_components: int
    residual_requires_structure_graphs: bool


class MaterialCapabilityCatalogResponse(BaseModel):
    """Complete MLIP capability catalog for UI/API discovery."""

    model_config = ConfigDict(extra="forbid")

    backends: list[MaterialBackendCapabilitiesResponse]
    quantities: list[str]
    model_modes: list[str]
    residual_input_modes: list[str]
    workflow_modes: list[str]


__all__ = [
    "MaterialBackendCapabilitiesResponse",
    "MaterialCapabilityCatalogResponse",
]
