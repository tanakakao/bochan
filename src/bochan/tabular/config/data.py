"""Tabular configuration objects for pandas / numpy friendly APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ColumnKey = str | int


@dataclass(frozen=True)
class TabularFeatureGroup:
    """A named raw-space feature group addressed by table columns.

    Args:
        name: Display name used in feature-importance results.
        columns: Non-empty, unique source column names.
        role: Semantic role attached to the group.
    """

    name: str
    columns: tuple[ColumnKey, ...]
    role: str = "group"

    def __post_init__(self) -> None:
        """Validate group structure before column resolution."""
        if not self.name:
            raise ValueError("TabularFeatureGroup.name must not be empty.")
        if not self.columns:
            raise ValueError(f"Feature group {self.name!r} must not be empty.")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(f"Feature group {self.name!r} contains duplicate columns.")


@dataclass
class TabularDataConfig:
    """Configuration for converting tabular data into bochan tensors.

    ``missing_strategy`` applies to explanatory variables only. Target missingness
    is controlled independently by ``target_missing_strategy`` so a missing
    objective can be retained as an unobserved cell without imputing it.

    When ``experiment_status_col`` is configured, its values must be ``success``,
    ``failed``, or ``pending`` (case-insensitive). Failed and pending rows are
    retained as experiment-state observations even when every target is missing.
    """

    input_cols: Sequence[ColumnKey] | None = None
    target_cols: Sequence[ColumnKey] | ColumnKey | None = None
    categorical_cols: Sequence[ColumnKey] = field(default_factory=list)
    target_categorical_cols: Sequence[ColumnKey] | None = None
    experiment_status_col: ColumnKey | None = None

    bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None
    dtype: Any | None = None
    device: Any | None = None

    dropna: bool = True
    missing_strategy: str | None = None
    target_missing_strategy: str = "drop"
    continuous_impute_strategy: str = "mean"
    categorical_impute_strategy: str = "mode"
    impute_targets: bool = False
    impute_random_state: int | None = None
    impute_max_iter: int = 10
    multiple_impute_sample_posterior: bool = False

    encode_categories: bool = True
    category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None
    target_category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None
    return_original_categories: bool = True
