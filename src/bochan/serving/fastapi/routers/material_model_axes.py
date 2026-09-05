"""Material model-axis discovery and validation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bochan.api import MaterialAPIModelSpec, material_task_fixed_features
from bochan.models.regression.gaussian.materials import (
    list_material_families,
    material_model_axes_capabilities,
)

from ..schemas.material_model_axes import (
    MaterialExplicitTaskRequest,
    MaterialModelAxesCapabilitiesResponse,
    MaterialModelAxesCatalogResponse,
    MaterialModelAxesRequest,
    MaterialModelAxesResponse,
    MaterialTaskFixedFeaturesRequest,
    MaterialTaskFixedFeaturesResponse,
)

router = APIRouter(prefix="/materials/models", tags=["materials"])


def _to_api_spec(request: MaterialModelAxesRequest) -> MaterialAPIModelSpec:
    task = request.task
    return MaterialAPIModelSpec(
        family=request.family,
        kind=request.kind,
        input_mode=request.input_mode,
        output_mode=request.output_mode,
        task_mode=request.task_mode,
        fidelity_mode=request.fidelity_mode,
        task_feature=-1 if task is None else task.task_feature,
        all_tasks=None if task is None or task.all_tasks is None else tuple(task.all_tasks),
        output_tasks=None if task is None or task.output_tasks is None else tuple(task.output_tasks),
        backend_kwargs=dict(request.backend_kwargs),
    )


def _normalized_response(request: MaterialModelAxesRequest) -> MaterialModelAxesResponse:
    spec = _to_api_spec(request)
    payload = spec.as_dict()
    task = None
    if spec.axes.task_mode == "explicit":
        task = MaterialExplicitTaskRequest(
            task_feature=spec.task_feature,
            all_tasks=None if spec.all_tasks is None else list(spec.all_tasks),
            output_tasks=None if spec.output_tasks is None else list(spec.output_tasks),
        )
    return MaterialModelAxesResponse(
        family=payload["family"],
        domain=payload["domain"],
        kind=payload["kind"],
        input_mode=payload["input_mode"],
        output_mode=payload["output_mode"],
        task_mode=payload["task_mode"],
        fidelity_mode=payload["fidelity_mode"],
        route=payload["route"],
        implemented=payload["implemented"],
        cat_dims=list(request.cat_dims or []),
        task=task,
    )


@router.get("/capabilities", response_model=MaterialModelAxesCatalogResponse)
def get_material_model_axes_catalog() -> MaterialModelAxesCatalogResponse:
    """Return model-axis capabilities for all registered material families."""

    families = [
        MaterialModelAxesCapabilitiesResponse.model_validate(
            material_model_axes_capabilities(family)
        )
        for family in list_material_families()
    ]
    return MaterialModelAxesCatalogResponse(families=families)


@router.get(
    "/capabilities/{family}",
    response_model=MaterialModelAxesCapabilitiesResponse,
)
def get_material_model_axes_capabilities(
    family: str,
) -> MaterialModelAxesCapabilitiesResponse:
    """Return model-axis capabilities for one registered material family."""

    try:
        payload = material_model_axes_capabilities(family)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MaterialModelAxesCapabilitiesResponse.model_validate(payload)


@router.post("/validate", response_model=MaterialModelAxesResponse)
def validate_material_model_axes(
    request: MaterialModelAxesRequest,
) -> MaterialModelAxesResponse:
    """Normalize and validate a JSON material model-axis request."""

    try:
        response = _normalized_response(request)
        if response.input_mode == "mixed" and not response.cat_dims:
            raise ValueError("cat_dims is required when input_mode='mixed'.")
        if response.input_mode == "continuous" and response.cat_dims:
            raise ValueError("cat_dims must be omitted when input_mode='continuous'.")
        if response.task_mode == "none" and request.task is not None:
            raise ValueError("task is only valid when task_mode='explicit'.")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return response


@router.post(
    "/task-fixed-features",
    response_model=MaterialTaskFixedFeaturesResponse,
)
def get_material_task_fixed_features(
    request: MaterialTaskFixedFeaturesRequest,
) -> MaterialTaskFixedFeaturesResponse:
    """Resolve fixed features for BO over one explicit material task."""

    try:
        spec = _to_api_spec(request.model)
        fixed_features = material_task_fixed_features(
            spec,
            request.target_task,
            input_dim=request.input_dim,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MaterialTaskFixedFeaturesResponse(fixed_features=fixed_features)


__all__ = ["router"]
