"""Dataset metadata for the tabular API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import ColumnKey


@dataclass
class TabularDataset:
    """Data and metadata needed to convert candidates back to tables."""

    X: Any
    Y: Any | None
    feature_names: list[ColumnKey]
    target_names: list[ColumnKey]
    cat_dims: list[int]
    Yvar: Any | None = None
    bounds: Any | None = None
    category_maps: dict[ColumnKey, dict[Any, int]] | None = None
    inverse_category_maps: dict[ColumnKey, dict[int, Any]] | None = None
    target_category_maps: dict[ColumnKey, dict[Any, int]] | None = None
    inverse_target_category_maps: dict[ColumnKey, dict[int, Any]] | None = None
    impute_values: dict[ColumnKey, Any] | None = None
    target_impute_values: dict[ColumnKey, Any] | None = None
    source_index: Any | None = None


__all__ = ["TabularDataset"]
