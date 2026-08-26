"""FastAPI endpoints for structure-aware ALIGNN tabular optimization."""

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
    _normalize_structure_column,
    alignn_candidate_response,
    alignn_predict_response,
    build_alignn_fit_response,
    fit_alignn_tabular_optimizer,
)
from ..services.tabular import to_dataframe
from ..services.tabular_artifacts import (
    append_tabular_data,
    load_tabular_artifact,
    save_tabular_artifact,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)
FILE_STORE_DEP = Depends(get_file_optimizer_store)
_ALIGNN_MODEL_TYPES = frozenset(
    {"alignn_gp", "alignn_dkl", "alignn_multitask", "alignn_multitask_dkl"}
)

router = APIRouter(prefix="/tabular/alignn/models", tags=["tabular", "alignn"])


def _validate_alignn_optimizer(optimizer: Any, *, model_id: str | None = None) -> Any:
    """Validate that one tabular optimizer carries the ALIGNN structure contract."""

    model_type = str(getattr(optimizer.model_config, "model_type", "")).lower()
    if model_type not in _ALIGNN_MODEL_TYPES:
        prefix = f"model_id={model_id!r} " if model_id is not None else "Artifact "
        raise TypeError(f"{prefix}is not an ALIGNN tabular model.")
    structure = getattr(optimizer, "structure", None)
    if structure is None or not bool(getattr(structure, "enabled", False)):
        raise TypeError("ALIGNN tabular model is missing its fitted structure contract.")
    if getattr(optimizer, "dataset", None) is None or optimizer.bo.bundle is None:
        raise TypeError("ALIGNN tabular model is not fitted.")
    return optimizer


def _get_optimizer(store: TabularOptimizerStore, model_id: str) -> Any:
    try:
        optimizer = store.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return _validate_alignn_optimizer(optimizer, model_id=model_id)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", response_model=TabularModelFitResponse)
def fit_alignn_tabular_model(
    request: ALIGNNTabularFitModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Fit/store ALIGNN GP, DKL, or correlated multitask variants."""

    try:
        optimizer = fit_alignn_tabular_optimizer(request)
        model_id = store.add(optimizer)
        return build_alignn_fit_response(model_id, optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/tell", response_model=TabularModelFitResponse)
def tell_alignn_tabular_model(
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
    """Predict known structures using the fitted process-category mapping."""

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
    """Enumerate structure/category choices and optimize continuous process inputs."""

    return _candidate_endpoint(model_id, request, store, use_ask=False)


@router.post("/{model_id}/ask", response_model=TabularCandidateResponse)
def ask_alignn_tabular_candidates(
    model_id: str,
    request: ALIGNNTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Generate and register pending structure/mixed-process candidates."""

    return _candidate_endpoint(model_id, request, store, use_ask=True)


@router.post("/{model_id}/save", response_model=SaveModelResponse)
def save_alignn_tabular_model(
    model_id: str,
    request: SaveModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> SaveModelResponse:
    """Save an ALIGNN tabular optimizer in the common trusted artifact format."""

    optimizer = _get_optimizer(store, model_id)
    try:
        path = save_tabular_artifact(
            optimizer,
            file_store,
            filename=request.filename,
            default_stem=model_id,
            overwrite=request.overwrite,
            metadata={
                "surface": "fastapi-alignn",
                "source_model_id": model_id,
                "model_family": "alignn",
            },
        )
        alignn_metadata = build_alignn_fit_response(model_id, optimizer).metadata.get(
            "alignn", {}
        )
        return SaveModelResponse(
            model_id=model_id,
            filename=file_store.relative_name(path),
            path=str(path),
            metadata={
                **model_metadata(optimizer.bo),
                "artifact_backend": "tabular",
                "artifact_format": "bochan-model-artifact",
                "alignn": alignn_metadata,
            },
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load", response_model=TabularModelLoadResponse)
def load_alignn_tabular_model(
    request: LoadModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> TabularModelLoadResponse:
    """Load a trusted ALIGNN tabular artifact and restore its structure contract."""

    try:
        optimizer, path = load_tabular_artifact(
            file_store,
            filename=request.filename,
            map_location=request.map_location,
            trust_pickle=request.trust_pickle,
        )
        _validate_alignn_optimizer(optimizer)
        model_id = store.add(optimizer)
        response = build_alignn_fit_response(model_id, optimizer).model_dump()
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
