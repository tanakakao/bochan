"""Model refit, observation, and deletion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import OptimizerStore, get_optimizer_store
from ..schemas import ModelDeleteResponse, ModelFitResponse, RefitModelRequest, TellRequest
from ..services import models as model_service

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)
router = APIRouter(prefix="/models", tags=["models"])


@router.post("/{model_id}/refit", response_model=ModelFitResponse)
def refit_model(
    model_id: str,
    request: RefitModelRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelFitResponse:
    try:
        return model_service.refit_model(model_id, request, store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
