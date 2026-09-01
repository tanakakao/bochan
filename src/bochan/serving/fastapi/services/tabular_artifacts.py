"""Reusable tabular tell and artifact persistence helpers for FastAPI."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bochan.api import ObservationData
from bochan.tabular import TabularBayesianOptimizer


def synchronize_tabular_dataset(optimizer: TabularBayesianOptimizer) -> None:
    """Keep facade dataset tensors aligned with the underlying optimizer state."""

    if optimizer.dataset is None:
        return

    observations = getattr(optimizer.bo, "observations", None)
    observation_aware_dataset = all(
        hasattr(optimizer.dataset, name)
        for name in ("observed_mask", "failed_mask", "pending_mask")
    )
    if observations is not None and observation_aware_dataset:
        optimizer.dataset.X = observations.X
        optimizer.dataset.Y = observations.Y
        optimizer.dataset.observed_mask = observations.observed_mask
        optimizer.dataset.failed_mask = observations.failed_mask
        optimizer.dataset.pending_mask = observations.pending_mask
        return

    train_x = getattr(optimizer.bo, "train_X", None)
    train_y = getattr(optimizer.bo, "train_Y", None)
    if train_x is not None:
        optimizer.dataset.X = train_x
    if train_y is not None:
        optimizer.dataset.Y = train_y


def _new_tabular_observations(optimizer: TabularBayesianOptimizer, dataset: Any) -> Any:
    """Preserve explicit experiment state and canonical success defaults for tell()."""

    observation_factory = getattr(dataset, "observation_data", None)
    if callable(observation_factory):
        return observation_factory()
    if getattr(optimizer.bo, "observations", None) is None:
        return None
    if dataset.Y is None:
        raise ValueError("Target values are required for tabular tell().")
    return ObservationData.from_status(
        dataset.X,
        dataset.Y,
        status=["success"] * int(dataset.X.shape[0]),
    )


def append_tabular_data(
    optimizer: TabularBayesianOptimizer,
    frame: Any,
) -> None:
    """Encode and append tell rows while preserving physical experiment state."""

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

    new_observations = _new_tabular_observations(optimizer, new_dataset)
    if new_observations is None:
        optimizer.bo.update_data(new_dataset.X, new_dataset.Y)
    else:
        current = optimizer.bo.observations
        if current is None:
            raise RuntimeError("Observation-aware tell requires fitted observation state.")
        optimizer.bo.observations = current.resolve_pending(new_observations)
        optimizer.bo.train_X, optimizer.bo.train_Y = (
            optimizer.bo.observations.objective_training_data()
        )
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
