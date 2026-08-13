"""TabularBayesianOptimizer HTTP endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import TabularOptimizerStore, get_tabular_optimizer_store
from ..schemas import ModelDeleteResponse, ModelListResponse
from ..schemas.tabular import (
    TabularCandidateRequest,
    TabularCandidateResponse,
    TabularFeatureImportanceRequest,
    TabularFeatureImportanceResponse,
    TabularFitModelRequest,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)
from ..services.tabular import (
    build_fit_response,
    candidate_response,
    compute_feature_importance_response,
    fit_tabular_optimizer,
    predict_response,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)

router = APIRouter(prefix="/tabular/models", tags=["tabular"])


def _get_optimizer(store: TabularOptimizerStore, model_id: str) -> Any:
    """Resolve one stored optimizer and translate a missing id to HTTP 404."""

    try:
        return store.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{model_id}/feature-importance", response_model=TabularFeatureImportanceResponse)
def compute_tabular_feature_importance(
    model_id: str,
    request: TabularFeatureImportanceRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularFeatureImportanceResponse:
    """Compute feature importance for one fitted tabular model."""

    optimizer = _get_optimizer(store, model_id)
    try:
        return compute_feature_importance_response(model_id, optimizer, request)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("", response_model=TabularModelFitResponse)
def fit_tabular_model(
    request: TabularFitModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Fit and store one tabular optimizer."""

    try:
        optimizer = fit_tabular_optimizer(request)
        model_id = store.add(optimizer)
        return build_fit_response(model_id, optimizer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ModelListResponse)
def list_tabular_models(
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> ModelListResponse:
    return ModelListResponse(model_ids=store.list_ids())


@router.post("/{model_id}/predict", response_model=TabularPredictResponse)
def predict_tabular_model(
    model_id: str,
    request: TabularPredictRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularPredictResponse:
    optimizer = _get_optimizer(store, model_id)
    try:
        return predict_response(model_id, optimizer, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _candidate_endpoint(
    model_id: str,
    request: TabularCandidateRequest,
    store: TabularOptimizerStore,
    *,
    use_ask: bool = False,
) -> TabularCandidateResponse:
    optimizer = _get_optimizer(store, model_id)
    try:
        return candidate_response(
            model_id,
            optimizer,
            request,
            use_ask=use_ask,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/candidates", response_model=TabularCandidateResponse)
def generate_tabular_candidates(
    model_id: str,
    request: TabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    return _candidate_endpoint(model_id, request, store)


@router.post("/{model_id}/ask", response_model=TabularCandidateResponse)
def ask_tabular_candidates(
    model_id: str,
    request: TabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    return _candidate_endpoint(model_id, request, store, use_ask=True)


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
def delete_tabular_model(
    model_id: str,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> ModelDeleteResponse:
    try:
        store.delete(model_id)
        return ModelDeleteResponse(model_id=model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
