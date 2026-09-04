"""Material MLIP capability discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bochan.models.regression.gaussian.materials.structure.capabilities import (
    get_material_backend_capabilities,
    get_material_capability_catalog,
)

from ..schemas.material_capabilities import (
    MaterialBackendCapabilitiesResponse,
    MaterialCapabilityCatalogResponse,
)

router = APIRouter(prefix="/materials/mlip", tags=["materials"])


@router.get("/capabilities", response_model=MaterialCapabilityCatalogResponse)
def get_mlip_capability_catalog() -> MaterialCapabilityCatalogResponse:
    """Return the complete MLIP capability catalog."""

    return MaterialCapabilityCatalogResponse.model_validate(
        get_material_capability_catalog()
    )


@router.get(
    "/capabilities/{backend}",
    response_model=MaterialBackendCapabilitiesResponse,
)
def get_mlip_backend_capabilities(
    backend: str,
) -> MaterialBackendCapabilitiesResponse:
    """Return capability metadata for one MLIP backend."""

    try:
        capabilities = get_material_backend_capabilities(backend)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MaterialBackendCapabilitiesResponse.model_validate(capabilities.as_dict())


__all__ = ["router"]
