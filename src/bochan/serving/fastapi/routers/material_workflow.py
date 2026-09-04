"""Material MLIP workflow validation, configuration, and execution endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from bochan.models.regression.gaussian.materials.structure.capabilities import (
    get_material_backend_capabilities,
)
from bochan.models.regression.gaussian.materials.structure.factory import (
    create_structure_relaxer,
)
from bochan.models.regression.gaussian.materials.structure.workflow_factory import (
    MaterialWorkflowSpec,
)

from ..schemas import (
    MaterialRelaxationConfig,
    MaterialRelaxationExecutionRequest,
    MaterialRelaxationExecutionResponse,
    MaterialWorkflowConfigRequest,
    MaterialWorkflowConfigResponse,
    MaterialWorkflowSpecRequest,
    MaterialWorkflowValidationResponse,
)

router = APIRouter(prefix="/materials/mlip/workflows", tags=["materials"])


def _resolve_workflow(request: MaterialWorkflowSpecRequest) -> tuple[MaterialWorkflowSpec, list[str]]:
    """Normalize one workflow request and resolve its runtime requirements."""

    spec = MaterialWorkflowSpec(
        backend=request.backend,
        quantity=request.quantity,
        model_mode=request.model_mode,
        workflow_mode=request.workflow_mode,
    )
    capabilities = get_material_backend_capabilities(spec.backend)
    if not capabilities.supports(
        quantity=spec.quantity,
        model_mode=spec.model_mode,
        workflow_mode=spec.workflow_mode,
    ):
        raise ValueError(
            "Unsupported material workflow combination: "
            f"{spec.backend}/{spec.quantity}/{spec.model_mode}/{spec.workflow_mode}."
        )
    requirements = capabilities.requirements(
        quantity=spec.quantity,
        model_mode=spec.model_mode,
    )
    return spec, list(requirements)


def _canonical_relaxation(
    spec: MaterialWorkflowSpec,
    relaxation: MaterialRelaxationConfig | None,
) -> MaterialRelaxationConfig | None:
    """Return canonical common relaxation settings for the selected workflow."""

    if spec.workflow_mode == "model_only":
        if relaxation is not None:
            raise ValueError("model_only workflows do not accept relaxation settings.")
        return None
    return MaterialRelaxationConfig() if relaxation is None else relaxation


@router.post("/validate", response_model=MaterialWorkflowValidationResponse)
def validate_material_workflow(
    request: MaterialWorkflowSpecRequest,
) -> MaterialWorkflowValidationResponse:
    """Normalize and validate an MLIP workflow without loading an MLIP model."""

    try:
        spec, requirements = _resolve_workflow(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MaterialWorkflowValidationResponse(
        valid=True,
        spec=spec.as_dict(),
        requirements=requirements,
    )


@router.post("/configure", response_model=MaterialWorkflowConfigResponse)
def configure_material_workflow(
    request: MaterialWorkflowConfigRequest,
) -> MaterialWorkflowConfigResponse:
    """Build a canonical dependency-light workflow execution configuration."""

    try:
        spec, requirements = _resolve_workflow(request)
        relaxation = _canonical_relaxation(spec, request.relaxation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload: dict[str, Any] = {
        "valid": True,
        "spec": spec.as_dict(),
        "requirements": requirements,
        "relaxation": relaxation,
    }
    return MaterialWorkflowConfigResponse(**payload)


@router.post(
    "/execute/relaxation",
    response_model=MaterialRelaxationExecutionResponse,
)
def execute_material_relaxation(
    request: MaterialRelaxationExecutionRequest,
) -> MaterialRelaxationExecutionResponse:
    """Execute the MLIP relaxation stage for a configured rank/acquisition workflow."""

    try:
        spec, _ = _resolve_workflow(request)
        relaxation = _canonical_relaxation(spec, request.relaxation)
        if relaxation is None:
            raise ValueError("model_only workflows do not have a relaxation execution stage.")
        relaxer = create_structure_relaxer(spec.backend, **request.backend_options)
        relax_kwargs = relaxation.model_dump()
        results = [
            relaxer.relax(structure, **relax_kwargs).as_dict()
            for structure in request.structures
        ]
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MaterialRelaxationExecutionResponse(
        spec=spec.as_dict(),
        relaxation=relaxation,
        backend_options=dict(request.backend_options),
        results=results,
    )


__all__ = [
    "configure_material_workflow",
    "execute_material_relaxation",
    "router",
    "validate_material_workflow",
]
