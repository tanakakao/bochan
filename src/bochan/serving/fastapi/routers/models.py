"""Model fit / planning endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import OptimizerStore, get_optimizer_store
from ..schemas import (
    AutoCandidateRequest,
    FitModelRequest,
    LLMPlanRequest,
    ModelFitResponse,
    ModelListResponse,
)
from ..services import models as model_service

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)
router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelFitResponse)
def fit_model(
    request: FitModelRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelFitResponse:
    try:
        return model_service.fit_model(request, store)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/plan")
def plan_model_config(request: LLMPlanRequest) -> dict[str, Any]:
    """Infer model / fit / acquisition / optimize configs without fitting a model."""
    try:
        return model_service.plan_model_config(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/auto-candidates")
def auto_candidates(
    request: AutoCandidateRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> dict[str, Any]:
    """Infer configs, fit a model, and generate candidates in one request."""
    try:
        return model_service.auto_candidates(request, store)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ModelListResponse)
def list_models(
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelListResponse:
    return ModelListResponse(model_ids=store.list_ids())
