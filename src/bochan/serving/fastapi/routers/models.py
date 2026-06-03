"""Model fit / lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from bochan.api import BayesianOptimizer

from ..converters import model_metadata, to_data_context, to_fit_config, to_model_config, to_tensor
from ..dependencies import InMemoryOptimizerStore, get_optimizer_store
from ..schemas import FitModelRequest, ModelDeleteResponse, ModelFitResponse, ModelListResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelFitResponse)
def fit_model(
    request: FitModelRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> ModelFitResponse:
    try:
        train_X = to_tensor(request.train_X)
        train_Y = to_tensor(request.train_Y)
        bounds = to_tensor(request.bounds) if request.bounds is not None else None
        model_config = to_model_config(request.bo_model_config)
        fit_config = to_fit_config(request.fit_config)
        data_context = to_data_context(request.data_context) if request.data_context is not None else None

        optimizer = BayesianOptimizer(
            model_config=model_config,
            fit_config=fit_config,
            bounds=bounds,
            data_context=data_context,
        )
        optimizer.fit(train_X, train_Y)
        model_id = store.add(optimizer)
        n_train = int(train_X.shape[-2]) if hasattr(train_X, "shape") else None
        return ModelFitResponse(
            model_id=model_id,
            task_type=str(optimizer.bundle.task_type),
            model_type=str(optimizer.bundle.model_type),
            n_train=n_train,
            metadata=model_metadata(optimizer),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ModelListResponse)
def list_models(store: InMemoryOptimizerStore = Depends(get_optimizer_store)) -> ModelListResponse:
    return ModelListResponse(model_ids=store.list_ids())


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
def delete_model(
    model_id: str,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> ModelDeleteResponse:
    try:
        store.delete(model_id)
        return ModelDeleteResponse(model_id=model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
