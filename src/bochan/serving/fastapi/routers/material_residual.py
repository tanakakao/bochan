"""FastAPI endpoints for CHGNet/M3GNet/MACE residual GP optimization."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from bochan.tabular.structure.material_residual import material_residual_model_types

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
from ..schemas.material_residual import (
    MaterialResidualTabularCandidateRequest,
    MaterialResidualTabularFitModelRequest,
)
from ..schemas.tabular import (
    TabularCandidateResponse,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)
from ..services.chgnet_tabular import _normalize_structure_column
from ..services.material_residual import (
    build_material_residual_fit_response,
    fit_material_residual_tabular_optimizer,
    material_residual_candidate_response,
    material_residual_predict_response,
)
from ..services.tabular import to_dataframe
from ..services.tabular_artifacts import (
    append_tabular_data,
    load_tabular_artifact,
    save_tabular_artifact,
)

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)
FILE_STORE_DEP = Depends(get_file_optimizer_store)
_RESIDUAL_MODEL_TYPES = frozenset(material_residual_model_types())

router = APIRouter(
    prefix="/tabular/material-residual/models",
    tags=["tabular", "materials", "residual-gp"],
)


def _validate_material_residual_optimizer(
    optimizer: Any,
    *,
    model_id: str | None = None,
) -> Any:
    """Validate one fitted residual structure optimizer."""

    model_type = str(getattr(optimizer.model_config, "model_type", "")).lower()
    if model_type not in _RESIDUAL_MODEL_TYPES:
        prefix = f"model_id={model_id!r} " if model_id is not None else "Artifact "
        raise TypeError(f"{prefix}is not a material residual GP model.")
    structure = getattr(optimizer, "structure", None)
    if structure is None or not bool(getattr(structure, "enabled", False)):
        raise TypeError("Material residual model is missing its fitted structure contract.")
    if getattr(structure, "graph_builder", None) is not None:
        raise TypeError("Material residual models must not contain an ALIGNN graph builder.")
    if getattr(optimizer, "dataset", None) is None or optimizer.bo.bundle is None:
        raise TypeError("Material residual tabular model is not fitted.")
    return optimizer


def _get_optimizer(store: TabularOptimizerStore, model_id: str) -> Any:
    try:
        optimizer = store.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return _validate_material_residual_optimizer(optimizer, model_id=model_id)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", response_model=TabularModelFitResponse)
def fit_material_residual_model(
    request: MaterialResidualTabularFitModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Fit/store a pretrained-baseline + GP residual workflow."""

    try:
        optimizer = fit_material_residual_tabular_optimizer(request)
        model_id = store.add(optimizer)
        return build_material_residual_fit_response(model_id, optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/tell", response_model=TabularModelFitResponse)
def tell_material_residual_model(
    model_id: str,
    request: TabularTellRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Append structure/process observations and optionally refit."""

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
        return build_material_residual_fit_response(model_id, optimizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/predict", response_model=TabularPredictResponse)
def predict_material_residual_model(
    model_id: str,
    request: TabularPredictRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularPredictResponse:
    """Predict corrected properties for known structures."""

    optimizer = _get_optimizer(store, model_id)
    try:
        return material_residual_predict_response(model_id, optimizer, request)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _candidate_endpoint(
    model_id: str,
    request: MaterialResidualTabularCandidateRequest,
    store: TabularOptimizerStore,
    *,
    use_ask: bool,
) -> TabularCandidateResponse:
    optimizer = _get_optimizer(store, model_id)
    try:
        return material_residual_candidate_response(
            model_id,
            optimizer,
            request,
            use_ask=use_ask,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/candidates", response_model=TabularCandidateResponse)
def generate_material_residual_candidates(
    model_id: str,
    request: MaterialResidualTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Enumerate structures/categories and optimize continuous process inputs."""

    return _candidate_endpoint(model_id, request, store, use_ask=False)


@router.post("/{model_id}/ask", response_model=TabularCandidateResponse)
def ask_material_residual_candidates(
    model_id: str,
    request: MaterialResidualTabularCandidateRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularCandidateResponse:
    """Generate and register pending residual-GP candidates."""

    return _candidate_endpoint(model_id, request, store, use_ask=True)


@router.post("/{model_id}/save", response_model=SaveModelResponse)
def save_material_residual_model(
    model_id: str,
    request: SaveModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> SaveModelResponse:
    """Save a residual tabular optimizer in the common trusted artifact format."""

    optimizer = _get_optimizer(store, model_id)
    try:
        path = save_tabular_artifact(
            optimizer,
            file_store,
            filename=request.filename,
            default_stem=model_id,
            overwrite=request.overwrite,
            metadata={
                "surface": "fastapi-material-residual",
                "source_model_id": model_id,
                "model_family": "material_residual",
            },
        )
        residual_metadata = build_material_residual_fit_response(
            model_id,
            optimizer,
        ).metadata.get("material_residual", {})
        return SaveModelResponse(
            model_id=model_id,
            filename=file_store.relative_name(path),
            path=str(path),
            metadata={
                **model_metadata(optimizer.bo),
                "artifact_backend": "tabular",
                "artifact_format": "bochan-model-artifact",
                "material_residual": residual_metadata,
            },
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load", response_model=TabularModelLoadResponse)
def load_material_residual_model(
    request: LoadModelRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
    file_store: FileOptimizerStore = FILE_STORE_DEP,
) -> TabularModelLoadResponse:
    """Load a trusted residual-GP artifact and restore its structure contract."""

    try:
        optimizer, path = load_tabular_artifact(
            file_store,
            filename=request.filename,
            map_location=request.map_location,
            trust_pickle=request.trust_pickle,
        )
        _validate_material_residual_optimizer(optimizer)
        model_id = store.add(optimizer)
        response = build_material_residual_fit_response(model_id, optimizer).model_dump()
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
