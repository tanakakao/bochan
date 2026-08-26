"""FastAPI endpoints for structure-aware ALIGNN tabular optimization."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import TabularOptimizerStore, get_tabular_optimizer_store
from ..schemas.alignn_tabular import (
    ALIGNNTabularCandidateRequest,
    ALIGNNTabularFitModelRequest,
)
from ..schemas.tabular import (
    TabularCandidateResponse,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)
from ..services.alignn_tabular import (
    alignn_candidate_response,
    alignn_predict_response,
    build_alignn_fit_response,
    fit_alignn_tabular_optimizer,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)

router = APIRouter(prefix="/tabular/alignn/models", tags=["tabular", "alignn"])


def _get_optimizer(store: TabularOptimizerStore, model_id: str) -> Any:
    try:
        optimizer = store.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    model_type = str(getattr(optimizer.model_config, "model_type", "")).lower()
    if model_type not in {"alignn_gp", "alignn_dkl"}:
        raise HTTPException(
            status_code=422,
            detail=f"model_id={model_id!r} is not an ALIGNN tabular model.",
        )
    return optimizer


@router.post("", response_model=TabularModelFitResponse)
def fit_alignn_tabular_model(
    request: ALIGNNTabularFitModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Fit and store ALIGNN-GP / ALIGNN-DKL from inline crystal structures."""

    try:
        optimizer = fit_alignn_tabular_optimizer(request)
        model_id = store.add(optimizer)
        return build_alignn_fit_response(model_id, optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/predict", response_model=TabularPredictResponse)
def predict_alignn_tabular_model(
    model_id: str,
    request: TabularPredictRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularPredictResponse:
    optimizer = _get_optimizer(store, model_id)
    try:
        return alignn_predict_response(model_id, optimizer, request)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _candidate_endpoint(
    model_id: str,
    request: ALIGNNTabularCandidateRequest,
    store: TabularOptimizerStore,
    *,
    use_ask: bool,
) -> TabularCandidateResponse:
    optimizer = _get_optimizer(store, model_id)
    try:
        return alignn_candidate_response(
            model_id,
            optimizer,
            request,
            use_ask=use_ask,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/candidates", response_model=TabularCandidateResponse)
def generate_alignn_tabular_candidates(
    model_id: str,
    request: ALIGNNTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Optimize continuous process variables while enumerating structure IDs."""

    return _candidate_endpoint(model_id, request, store, use_ask=False)


@router.post("/{model_id}/ask", response_model=TabularCandidateResponse)
def ask_alignn_tabular_candidates(
    model_id: str,
    request: ALIGNNTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Generate and register pending ALIGNN structure/process candidates."""

    return _candidate_endpoint(model_id, request, store, use_ask=True)


__all__ = ["router"]
