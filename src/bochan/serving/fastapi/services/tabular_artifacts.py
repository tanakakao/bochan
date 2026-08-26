"""Reusable tabular tell and artifact persistence helpers for FastAPI."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bochan.tabular import TabularBayesianOptimizer


def synchronize_tabular_dataset(optimizer: TabularBayesianOptimizer) -> None:
    """Keep facade dataset tensors aligned with the underlying optimizer state."""

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
    frame: Any,
) -> None:
    """Encode rows with fitted maps and append them to the optimizer training data."""

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


def save_tabular_artifact(
    optimizer: TabularBayesianOptimizer,
    file_store: Any,
    *,
    filename: str | None,
    default_stem: str,
    overwrite: bool,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Save a synchronized tabular optimizer using the common artifact envelope."""

    synchronize_tabular_dataset(optimizer)
    return file_store.save(
        optimizer,
        filename,
        default_stem=default_stem,
        overwrite=overwrite,
        backend="tabular",
        metadata=dict(metadata or {}),
    )


def load_tabular_artifact(
    file_store: Any,
    *,
    filename: str,
    map_location: str | None,
    trust_pickle: bool,
) -> tuple[TabularBayesianOptimizer, Any]:
    """Load and validate a trusted tabular optimizer artifact."""

    optimizer, path = file_store.load(
        filename,
        map_location=map_location,
        trust_pickle=trust_pickle,
        expected_backend="tabular",
    )
    if not isinstance(optimizer, TabularBayesianOptimizer):
        raise TypeError("The selected artifact does not contain a tabular optimizer.")
    synchronize_tabular_dataset(optimizer)
    return optimizer, path


__all__ = [
    "append_tabular_data",
    "load_tabular_artifact",
    "save_tabular_artifact",
    "synchronize_tabular_dataset",
]
