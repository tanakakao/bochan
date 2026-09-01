"""FastAPI endpoints for structure-aware CHGNet tabular optimization."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

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
    TabularModelLoadResponse,
    TabularTellRequest,
)
from ..schemas.chgnet_tabular import (
    CHGNetTabularCandidateRequest,
    CHGNetTabularFitModelRequest,
)
from ..schemas.tabular import (
    TabularCandidateResponse,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)
from ..services.chgnet_tabular import (
    _normalize_structure_column,
    build_chgnet_fit_response,
    chgnet_candidate_response,
    chgnet_predict_response,
    fit_chgnet_tabular_optimizer,
)
from ..services.tabular import to_dataframe
from ..services.tabular_artifacts import (
    append_tabular_data,
    load_tabular_artifact,
    save_tabular_artifact,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)
FILE_STORE_DEP = Depends(get_file_optimizer_store)
_CHGNET_MODEL_TYPES = frozenset(
    {"chgnet_gp", "chgnet_dkl", "chgnet_multitask", "chgnet_multitask_dkl"}
)

router = APIRouter(prefix="/tabular/chgnet/models", tags=["tabular", "chgnet"])


def _validate_chgnet_optimizer(optimizer: Any, *, model_id: str | None = None) -> Any:
    """Validate that one tabular optimizer carries the CHGNet structure contract."""

    model_type = str(getattr(optimizer.model_config, "model_type", "")).lower()
    if model_type not in _CHGNET_MODEL_TYPES:
        prefix = f"model_id={model_id!r} " if model_id is not None else "Artifact "
        raise TypeError(f"{prefix}is not a CHGNet tabular model.")
    structure = getattr(optimizer, "structure", None)
    if structure is None or not bool(getattr(structure, "enabled", False)):
        raise TypeError("CHGNet tabular model is missing its fitted structure contract.")
    if getattr(structure, "graph_builder", None) is not None:
        raise TypeError("CHGNet tabular model must not contain an ALIGNN graph builder.")
    if getattr(optimizer, "dataset", None) is None or optimizer.bo.bundle is None:
        raise TypeError("CHGNet tabular model is not fitted.")
    return optimizer


def _get_optimizer(store: TabularOptimizerStore, model_id: str) -> Any:
    try:
        optimizer = store.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return _validate_chgnet_optimizer(optimizer, model_id=model_id)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", response_model=TabularModelFitResponse)
def fit_chgnet_tabular_model(
    request: CHGNetTabularFitModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Fit/store CHGNet GP, DKL, or correlated multitask variants."""

    try:
        optimizer = fit_chgnet_tabular_optimizer(request)
        model_id = store.add(optimizer)
        return build_chgnet_fit_response(model_id, optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/tell", response_model=TabularModelFitResponse)
def tell_chgnet_tabular_model(
    model_id: str,
    request: TabularTellRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Append structure/process observations using the fitted categorical mappings."""

    optimizer = _get_optimizer(store, model_id)
    try:
        frame = _normalize_structure_column(
            to_dataframe(request.data),
            str(optimizer.structure.column),
        )
        append_tabular_data(optimizer, frame)
        if request.refit:
            fit_config = (
                to_fit_config(request.fit_config)
                if request.fit_config is not None
                else optimizer.fit_config
            )
            optimizer.bo.refit(fit_config=fit_config)
            optimizer._sync_visualization_metadata()  # noqa: SLF001
        return build_chgnet_fit_response(model_id, optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/predict", response_model=TabularPredictResponse)
def predict_chgnet_tabular_model(
    model_id: str,
    request: TabularPredictRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularPredictResponse:
    """Predict known structures using the fitted process-category mapping."""

    optimizer = _get_optimizer(store, model_id)
    try:
        return chgnet_predict_response(model_id, optimizer, request)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _candidate_endpoint(
    model_id: str,
    request: CHGNetTabularCandidateRequest,
    store: TabularOptimizerStore,
    *,
    use_ask: bool,
) -> TabularCandidateResponse:
    optimizer = _get_optimizer(store, model_id)
    try:
        return chgnet_candidate_response(model_id, optimizer, request, use_ask=use_ask)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/candidates", response_model=TabularCandidateResponse)
def generate_chgnet_tabular_candidates(
    model_id: str,
    request: CHGNetTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Enumerate structure/category choices and optimize continuous process inputs."""

    return _candidate_endpoint(model_id, request, store, use_ask=False)


@router.post("/{model_id}/ask", response_model=TabularCandidateResponse)
def ask_chgnet_tabular_candidates(
    model_id: str,
    request: CHGNetTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Generate and register pending structure/mixed-process candidates."""

    return _candidate_endpoint(model_id, request, store, use_ask=True)


@router.post("/{model_id}/save", response_model=SaveModelResponse)
def save_chgnet_tabular_model(
    model_id: str,
    request: SaveModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> SaveModelResponse:
    """Save a CHGNet tabular optimizer in the common trusted artifact format."""

    optimizer = _get_optimizer(store, model_id)
    try:
        path = save_tabular_artifact(
            optimizer,
            file_store,
            filename=request.filename,
            default_stem=model_id,
            overwrite=request.overwrite,
            metadata={
                "surface": "fastapi-chgnet",
                "source_model_id": model_id,
                "model_family": "chgnet",
            },
        )
        chgnet_metadata = build_chgnet_fit_response(model_id, optimizer).metadata.get(
            "chgnet", {}
        )
        return SaveModelResponse(
            model_id=model_id,
            filename=file_store.relative_name(path),
            path=str(path),
            metadata={
                **model_metadata(optimizer.bo),
                "artifact_backend": "tabular",
                "artifact_format": "bochan-model-artifact",
                "chgnet": chgnet_metadata,
            },
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load", response_model=TabularModelLoadResponse)
def load_chgnet_tabular_model(
    request: LoadModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> TabularModelLoadResponse:
    """Load a trusted CHGNet tabular artifact and restore its structure contract."""

    try:
        optimizer, path = load_tabular_artifact(
            file_store,
            filename=request.filename,
            map_location=request.map_location,
            trust_pickle=request.trust_pickle,
        )
        _validate_chgnet_optimizer(optimizer)
        model_id = store.add(optimizer)
        response = build_chgnet_fit_response(model_id, optimizer).model_dump()
        return TabularModelLoadResponse(
            **response,
            filename=file_store.relative_name(path),
            path=str(path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
