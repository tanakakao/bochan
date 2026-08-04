"""Complete missing bounds for composition-enabled tabular workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .element_constraint_composition_optimizer import (
    TabularBayesianOptimizer as _ElementConstraintTabularBayesianOptimizer,
)


class TabularBayesianOptimizer(_ElementConstraintTabularBayesianOptimizer):
    """Infer bounds for ordinary columns beside composition-derived features.

    Composition wrappers must create explicit bounds for ILR/CLR/ALR coordinates
    and descriptor columns. Once a bounds mapping exists, the core tabular
    converter requires an entry for every input feature. This layer fills any
    missing passthrough numeric or categorical columns from the transformed
    training frame while preserving user-supplied bounds.
    """

    @staticmethod
    def _mapping_contains_column(bounds: Mapping[Any, Any], column: Any) -> bool:
        return column in bounds or str(column) in bounds

    @staticmethod
    def _infer_column_bound(series: Any, column: Any) -> list[float]:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(
                    f"Cannot infer bounds for column {column!r}: no finite values."
                )
            return [float(finite.min()), float(finite.max())]

        unique_count = int(series.dropna().nunique())
        if unique_count == 0:
            raise ValueError(
                f"Cannot infer categorical bounds for column {column!r}: "
                "no non-missing values."
            )
        return [0.0, float(unique_count - 1)]

    @classmethod
    def _complete_transformed_bounds(
        cls,
        bounds: Any,
        transformed: Any,
    ) -> Any:
        if not isinstance(bounds, Mapping):
            return bounds
        completed = dict(bounds)
        for column in transformed.columns:
            if cls._mapping_contains_column(completed, column):
                continue
            completed[column] = cls._infer_column_bound(
                transformed.loc[:, column],
                column,
            )
        return completed

    def _expanded_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = super()._expanded_bounds(bounds, transformed)
        return self._complete_transformed_bounds(expanded, transformed)

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        return self._complete_transformed_bounds(expanded, transformed)


__all__ = ["TabularBayesianOptimizer"]
