"""Optimizer artifact persistence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..converters import model_metadata
from ..dependencies import FileOptimizerStore, OptimizerStore, get_file_optimizer_store, get_optimizer_store
from ..schemas import LoadModelRequest, LoadModelResponse, SavedModelsResponse, SaveModelRequest, SaveModelResponse
from .models import _model_fit_response

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)
FILE_STORE_DEP = Depends(get_file_optimizer_store)

router = APIRouter(tags=["artifacts"])


@router.get("/artifacts", response_model=SavedModelsResponse)
def list_artifacts(file_store: FileOptimizerStore = FILE_STORE_DEP) -> SavedModelsResponse:
    """List saved optimizer artifacts.

    Args:
        file_store: Injected file artifact store.

    Returns:
        Artifact root directory and relative filenames.
    """
    return SavedModelsResponse(root_dir=str(file_store.root_dir), filenames=file_store.list())


@router.post("/models/{model_id}/save", response_model=SaveModelResponse)
def save_model(
    model_id: str,
    request: SaveModelRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> SaveModelResponse:
    """Save an in-memory optimizer as a file artifact.

    Args:
        model_id: Model id in the injected optimizer store.
        request: Save options including relative filename and overwrite flag.
        store: Injected optimizer store.
        file_store: Injected file artifact store.

    Returns:
        Saved artifact metadata.
    """
    try:
        optimizer = store.get(model_id)
        path = file_store.save(optimizer, request.filename, default_stem=model_id, overwrite=request.overwrite)
        return SaveModelResponse(
            model_id=model_id,
            filename=file_store.relative_name(path),
            path=str(path),
            metadata=model_metadata(optimizer),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/load", response_model=LoadModelResponse)
def load_model(
    request: LoadModelRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> LoadModelResponse:
    """Load a trusted optimizer artifact and register it as a new model.

    Args:
        request: Load request. ``trust_pickle`` must be true because loading uses
            pickle through ``torch.load``.
        store: Injected optimizer store.
        file_store: Injected file artifact store.

    Returns:
        New model id and loaded artifact metadata.
    """
    try:
        optimizer, path = file_store.load(
            request.filename,
            map_location=request.map_location,
            trust_pickle=request.trust_pickle,
        )
        model_id = store.add(optimizer)
        response = _model_fit_response(model_id, optimizer).model_dump()
        return LoadModelResponse(
            **response,
            filename=file_store.relative_name(path),
            path=str(path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
