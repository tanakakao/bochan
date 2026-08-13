"""Application services for tabular model updates and persistence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bochan.tabular import TabularBayesianOptimizer

from ..converters import model_metadata, to_fit_config
from ..schemas import (
    LoadModelRequest,
    SaveModelRequest,
    SaveModelResponse,
    TabularModelFitResponse,
    TabularModelLoadResponse,
    TabularTellRequest,
)
from .tabular import build_fit_response, to_dataframe


def synchronize_tabular_dataset(optimizer: TabularBayesianOptimizer) -> None:
    """Keep tabular metadata tensors aligned after the core optimizer grows."""

    if optimizer.dataset is None:
        return
    train_x = getattr(optimizer.bo, "train_X", None)
    train_y = getattr(optimizer.bo, "train_Y", None)
    if train_x is not None:
        optimizer.dataset.X = train_x
    if train_y is not None:
        optimizer.dataset.Y = train_y


def append_tabular_data(
    optimizer: TabularBayesianOptimizer,
    frame: object,
) -> None:
    """Encode new rows with fitted category maps and append their tensors."""

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
    synchronize_tabular_dataset(optimizer)


def tell_tabular_optimizer(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: TabularTellRequest,
) -> TabularModelFitResponse:
    """Append observations, optionally refit, and serialize the fitted state."""

    frame = to_dataframe(request.data)
    append_tabular_data(optimizer, frame)
    if request.refit:
        fit_config = (
            to_fit_config(request.fit_config)
            if request.fit_config is not None
            else optimizer.fit_config
        )
        optimizer.bo.refit(fit_config=fit_config)
        optimizer._sync_visualization_metadata()  # noqa: SLF001
    return build_fit_response(model_id, optimizer)


def save_tabular_optimizer(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: SaveModelRequest,
    file_store: Any,
) -> SaveModelResponse:
    """Persist one tabular optimizer using the common artifact format."""

    synchronize_tabular_dataset(optimizer)
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


def load_tabular_optimizer(
    request: LoadModelRequest,
    file_store: Any,
) -> tuple[TabularBayesianOptimizer, Any]:
    """Load and validate one trusted tabular optimizer artifact."""

    optimizer, path = file_store.load(
        request.filename,
        map_location=request.map_location,
        trust_pickle=request.trust_pickle,
        expected_backend="tabular",
    )
    if not isinstance(optimizer, TabularBayesianOptimizer):
        raise TypeError("The selected artifact does not contain a tabular optimizer.")
    synchronize_tabular_dataset(optimizer)
    return optimizer, path


def build_tabular_load_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    path: Any,
    file_store: Any,
) -> TabularModelLoadResponse:
    """Serialize a loaded tabular optimizer with artifact path metadata."""

    response = build_fit_response(model_id, optimizer).model_dump()
    return TabularModelLoadResponse(
        **response,
        filename=file_store.relative_name(path),
        path=str(path),
    )


__all__ = [
    "append_tabular_data",
    "build_tabular_load_response",
    "load_tabular_optimizer",
    "save_tabular_optimizer",
    "synchronize_tabular_dataset",
    "tell_tabular_optimizer",
]
