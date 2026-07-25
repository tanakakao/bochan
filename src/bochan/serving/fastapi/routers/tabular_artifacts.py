"""Tabular model update and common ``.bochan.pt`` persistence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from bochan.tabular import TabularBayesianOptimizer

from ..converters import model_metadata, to_fit_config
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
from .tabular import _fit_response, _to_dataframe

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
        frame = _to_dataframe(request.data)
        fit_config = to_fit_config(request.fit_config) if request.fit_config is not None else None
        optimizer.tell(frame, refit=request.refit, fit_config=fit_config)
        return _fit_response(model_id, optimizer)
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
    """Save a tabular optimizer using the same artifact format as tensor models."""

    try:
        optimizer = store.get(model_id)
        path = file_store.save(
            optimizer,
            request.filename,
            default_stem=model_id,
            overwrite=request.overwrite,
            backend="tabular",
            metadata={"surface": "fastapi", "source_model_id": model_id},
        )
        return SaveModelResponse(
            model_id=model_id,
            filename=file_store.relative_name(path),
            path=str(path),
            metadata={
                **model_metadata(optimizer.bo),
                "artifact_backend": "tabular",
                "artifact_format": "bochan-model-artifact",
            },
        )
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
        optimizer, path = file_store.load(
            request.filename,
            map_location=request.map_location,
            trust_pickle=request.trust_pickle,
            expected_backend="tabular",
        )
        if not isinstance(optimizer, TabularBayesianOptimizer):
            raise TypeError("The selected artifact does not contain a tabular optimizer.")
        model_id = store.add(optimizer)
        response = _fit_response(model_id, optimizer).model_dump()
        return TabularModelLoadResponse(
            **response,
            filename=file_store.relative_name(path),
            path=str(path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
