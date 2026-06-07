"""Model fit / lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from bochan.api import BayesianOptimizer

from ..converters import model_metadata, to_fit_config, to_data_context, to_model_config, to_tensor
from ..dependencies import InMemoryOptimizerStore, get_optimizer_store
from ..schemas import FitModelRequest, ModelDeleteResponse, ModelFitResponse, ModelListResponse, RefitModelRequest, TellRequest

router = APIRouter(prefix="/models", tags=["models"])


def _model_fit_response(model_id: str, optimizer: BayesianOptimizer) -> ModelFitResponse:
    train_X = getattr(optimizer, "train_X", None)
    n_train = int(train_X.shape[-2]) if hasattr(train_X, "shape") else None
    bundle = optimizer.bundle
    if bundle is None:
        raise RuntimeError("Optimizer has no fitted bundle.")
    return ModelFitResponse(
        model_id=model_id,
        task_type=str(bundle.task_type),
        model_type=str(bundle.model_type),
        n_train=n_train,
        metadata=model_metadata(optimizer),
    )


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
        return _model_fit_response(model_id, optimizer)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ModelListResponse)
def list_models(store: InMemoryOptimizerStore = Depends(get_optimizer_store)) -> ModelListResponse:
    return ModelListResponse(model_ids=store.list_ids())


@router.post("/{model_id}/refit", response_model=ModelFitResponse)
def refit_model(
    model_id: str,
    request: RefitModelRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> ModelFitResponse:
    try:
        optimizer = store.get(model_id)
        fit_config = to_fit_config(request.fit_config) if request.fit_config is not None else None
        optimizer.refit(fit_config=fit_config)
        return _model_fit_response(model_id, optimizer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/tell", response_model=ModelFitResponse)
def tell_model(
    model_id: str,
    request: TellRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> ModelFitResponse:
    try:
        optimizer = store.get(model_id)
        new_X = to_tensor(request.new_X)
        new_Y = to_tensor(request.new_Y)
        fit_config = to_fit_config(request.fit_config) if request.fit_config is not None else None
        optimizer.tell(new_X, new_Y, refit=request.refit, fit_config=fit_config)
        return _model_fit_response(model_id, optimizer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
