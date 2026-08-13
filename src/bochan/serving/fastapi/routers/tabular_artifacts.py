"""Tabular model update and common ``.bochan.pt`` persistence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    FileOptimizerStore,
    TabularOptimizerStore,
    get_file_optimizer_store,
    get_tabular_optimizer_store,
)
from ..schemas import (
    LoadModelRequest,
    SaveModelRequest,
    SaveModelResponse,
    TabularModelFitResponse,
    TabularModelLoadResponse,
    TabularTellRequest,
)
from ..services.tabular_artifacts import (
    build_tabular_load_response,
    load_tabular_optimizer,
    save_tabular_optimizer,
    tell_tabular_optimizer,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)
FILE_STORE_DEP = Depends(get_file_optimizer_store)

router = APIRouter(prefix="/tabular/models", tags=["tabular", "artifacts"])


@router.post("/{model_id}/tell", response_model=TabularModelFitResponse)
def tell_tabular_model(
    model_id: str,
    request: TabularTellRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Append tabular observations and optionally refit the model."""

    try:
        optimizer = store.get(model_id)
        return tell_tabular_optimizer(model_id, optimizer, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/save", response_model=SaveModelResponse)
def save_tabular_model(
    model_id: str,
    request: SaveModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> SaveModelResponse:
    """Save a tabular optimizer using the common artifact format."""

    try:
        optimizer = store.get(model_id)
        return save_tabular_optimizer(model_id, optimizer, request, file_store)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load", response_model=TabularModelLoadResponse)
def load_tabular_model(
    request: LoadModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> TabularModelLoadResponse:
    """Load a trusted common artifact containing a tabular optimizer."""

    try:
        optimizer, path = load_tabular_optimizer(request, file_store)
        model_id = store.add(optimizer)
        return build_tabular_load_response(model_id, optimizer, path, file_store)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
