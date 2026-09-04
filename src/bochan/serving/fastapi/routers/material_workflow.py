"""Material MLIP workflow validation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bochan.models.regression.gaussian.materials.structure.capabilities import (
    get_material_backend_capabilities,
)
from bochan.models.regression.gaussian.materials.structure.workflow_factory import (
    MaterialWorkflowSpec,
)

from ..schemas import MaterialWorkflowSpecRequest, MaterialWorkflowValidationResponse

router = APIRouter(prefix="/materials/mlip/workflows", tags=["materials"])


@router.post("/validate", response_model=MaterialWorkflowValidationResponse)
def validate_material_workflow(
    request: MaterialWorkflowSpecRequest,
) -> MaterialWorkflowValidationResponse:
    """Normalize and validate an MLIP workflow without loading an MLIP model."""

    try:
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MaterialWorkflowValidationResponse(
        valid=True,
        spec=spec.as_dict(),
        requirements=list(requirements),
    )


__all__ = ["router", "validate_material_workflow"]
