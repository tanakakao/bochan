"""Composition-aware transformed-bound completion helpers.

The bounds logic is a stateless component. It no longer defines another
``TabularBayesianOptimizer`` subclass in the public adapter inheritance chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class CompositionBoundsResolver:
    """Complete missing transformed bounds from observed transformed columns."""

    @staticmethod
    def mapping_contains_column(bounds: Mapping[Any, Any], column: Any) -> bool:
        """Return whether a bound exists using either native or string keys."""

        return column in bounds or str(column) in bounds

    @staticmethod
    def infer_column_bound(series: Any, column: Any) -> list[float]:
        """Infer one numeric or categorical transformed-column bound."""

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

    def complete(self, bounds: Any, transformed: Any) -> Any:
        """Fill only transformed columns that are absent from mapping bounds."""

        if not isinstance(bounds, Mapping):
            return bounds
        completed = dict(bounds)
        for column in transformed.columns:
            if self.mapping_contains_column(completed, column):
                continue
            completed[column] = self.infer_column_bound(
                transformed.loc[:, column],
                column,
            )
        return completed


__all__ = ["CompositionBoundsResolver"]
