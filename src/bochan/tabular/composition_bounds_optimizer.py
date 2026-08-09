"""Complete missing bounds for composition-enabled tabular workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import element_constraint_composition_optimizer as _element_constraint_module
from .observation_optimizer import ObservationTabularMixin

_ElementConstraintTabularBayesianOptimizer = (
    _element_constraint_module.TabularBayesianOptimizer
)


class TabularBayesianOptimizer(
    ObservationTabularMixin,
    _ElementConstraintTabularBayesianOptimizer,
):
    """Canonical composition- and observation-aware tabular optimizer.

    The class keeps the existing composition preprocessing / repair chain while
    adding explicit partial-observation and experiment-state handling through a
    source-level mixin. This remains the single public tabular optimizer class.
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


_element_constraint_module.TabularBayesianOptimizer = TabularBayesianOptimizer

__all__ = ["TabularBayesianOptimizer"]
