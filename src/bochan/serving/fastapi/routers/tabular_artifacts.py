"""Tabular model update and common ``.bochan.pt`` persistence endpoints."""

from __future__ import annotations

from dataclasses import replace

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
from ..services.tabular import build_fit_response, to_dataframe

TABULAR_STORE_DEP = Depends(get_tabular_optimizer_store)
FILE_STORE_DEP = Depends(get_file_optimizer_store)

router = APIRouter(prefix="/tabular/models", tags=["tabular", "artifacts"])


def _synchronize_tabular_dataset(optimizer: TabularBayesianOptimizer) -> None:
    """Keep tabular metadata tensors aligned after the underlying optimizer grows."""

    if optimizer.dataset is None:
        return
    train_x = getattr(optimizer.bo, "train_X", None)
    train_y = getattr(optimizer.bo, "train_Y", None)
    if train_x is not None:
        optimizer.dataset.X = train_x
    if train_y is not None:
        optimizer.dataset.Y = train_y


def _append_tabular_data(optimizer: TabularBayesianOptimizer, frame: object) -> None:
    """Encode new rows with the fitted category maps and append their tensors."""

    if optimizer.dataset is None:
        raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
    data_config = replace(
        optimizer.data_config,
        input_cols=list(optimizer.dataset.feature_names),
        target_cols=list(optimizer.dataset.target_names),
        category_maps=optimizer.dataset.category_maps,
        target_category_maps=optimizer.dataset.target_category_maps,
    )
    new_dataset = optimizer._to_dataset(  # noqa: SLF001
        frame,
        data_config=data_config,
        feature_names=optimizer.dataset.feature_names,
        target_names=optimizer.dataset.target_names,
    )
    if new_dataset.Y is None:
        raise ValueError("Target values are required for tabular tell().")
    optimizer.bo.update_data(new_dataset.X, new_dataset.Y)
    _synchronize_tabular_dataset(optimizer)


@router.post("/{model_id}/tell", response_model=TabularModelFitResponse)
def tell_tabular_model(
    model_id: str,
    request: TabularTellRequest,
    store: TabularOptimizerStore = TABULAR_STORE_DEP,
) -> TabularModelFitResponse:
    """Append tabular observations and optionally refit the model."""

    try:
        optimizer = store.get(model_id)
        frame = to_dataframe(request.data)
        _append_tabular_data(optimizer, frame)
        if request.refit:
            fit_config = (
                to_fit_config(request.fit_config)
                if request.fit_config is not None
                else optimizer.fit_config
            )
            optimizer.bo.refit(fit_config=fit_config)
            optimizer._sync_visualization_metadata()  # noqa: SLF001
        return build_fit_response(model_id, optimizer)
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
        _synchronize_tabular_dataset(optimizer)
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
        _synchronize_tabular_dataset(optimizer)
        model_id = store.add(optimizer)
        response = build_fit_response(model_id, optimizer).model_dump()
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
