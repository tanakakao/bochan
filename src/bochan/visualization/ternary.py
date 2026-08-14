"""Ternary plotting helpers for row-major and fixed-total coordinate grids.

The public ternary renderer accepts both the native ``(3, N)`` representation
used by :func:`bochan.visualization.data.tri_grid` and row-major ``(N, 3)``
grids used by composition visualizations.  Plotly's contour helper internally
expects a unit simplex, so fixed-total slices are normalized locally and only
the generated contour traces are restored to the caller's original total.

Keeping this behavior in the visualization layer avoids mutating Plotly or
``bochan.visualization.plots`` at Web application import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from plotly.graph_objs._figure import Figure

from .plots import show_triscatter_with_acqf as _show_triscatter_with_acqf


def _ternary_coordinates(grid: Any) -> np.ndarray:
    """Return numeric ternary coordinates with shape ``(2|3, n_points)``."""

    try:
        if isinstance(grid, pd.DataFrame):
            coordinates = grid.to_numpy(dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] not in {2, 3}:
                raise ValueError(
                    "Ternary grid DataFrame must contain two or three columns."
                )
            return coordinates.T
        coordinates = np.asarray(grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ternary grid values must be numeric.") from exc

    if coordinates.ndim != 2:
        raise ValueError("Ternary grid must be a two-dimensional array.")
    if coordinates.shape[0] in {2, 3}:
        return coordinates
    if coordinates.shape[1] in {2, 3}:
        return coordinates.T
    raise ValueError(
        "Ternary grid must have two or three coordinate rows or columns."
    )


def _normalized_ternary_coordinates(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize one fixed-total ternary slice to Plotly's unit simplex."""

    values = np.asarray(coordinates, dtype=float).copy()
    if values.ndim != 2 or values.shape[0] not in {2, 3}:
        raise ValueError("Ternary coordinates must have shape (2|3, n_points).")
    values[np.abs(values) < 1e-12] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("Ternary coordinates must be finite.")
    if (values < -1e-9).any():
        raise ValueError("Ternary coordinates must be non-negative.")
    values = np.clip(values, 0.0, None)

    totals = values.sum(axis=0)
    if (totals <= 0.0).any():
        raise ValueError("Each ternary point must have a positive coordinate total.")
    total = float(totals[0])
    if not np.allclose(totals, total, rtol=1e-7, atol=1e-10):
        raise ValueError("All ternary contour points must share the same total.")
    return values / totals[None, :], total


def _scale_contour_prefix(figure: Figure, total: float) -> Figure:
    """Restore only the contour traces that precede the renderer's marker traces."""

    if np.isclose(total, 1.0):
        return figure

    for trace in figure.data:
        if getattr(trace, "type", "") != "scatterternary":
            break
        for coordinate in ("a", "b", "c"):
            values = getattr(trace, coordinate, None)
            if values is None:
                continue
            setattr(
                trace,
                coordinate,
                (np.asarray(values, dtype=float) * float(total)).tolist(),
            )
    return figure


def show_triscatter_with_acqf(
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    data_tri_plot: tuple[np.ndarray, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    df_cand: pd.DataFrame | None = None,
    *,
    show_type: str = "acqf",
    cycle: str | Sequence[Any] | pd.Series | None = None,
    ncontours: int = 25,
) -> Figure:
    """Render a ternary contour without mutating Plotly or module globals.

    Row-major DataFrames and arrays are oriented explicitly.  Fixed-total
    composition slices such as ``Fe + Co + Ni = 0.8`` are normalized only for
    Plotly contour construction and then restored to their original coordinate
    total in the generated contour traces.
    """

    if not (isinstance(data_tri_plot, (tuple, list)) and len(data_tri_plot) == 2):
        raise ValueError("`data_tri_plot` は (ac, grid) のタプルで指定してください。")

    contour_values, grid = data_tri_plot
    coordinates = _ternary_coordinates(grid)
    flattened_values = np.asarray(contour_values).reshape(-1)
    if flattened_values.shape[0] != coordinates.shape[1]:
        raise ValueError(
            "Ternary contour values and coordinate points must have the same length."
        )

    normalized, total = _normalized_ternary_coordinates(coordinates)
    figure = _show_triscatter_with_acqf(
        feature_col1,
        feature_col2,
        feature_col3,
        target_col,
        (contour_values, normalized),
        X,
        y,
        df_cand,
        show_type=show_type,
        cycle=cycle,
        ncontours=ncontours,
    )
    return _scale_contour_prefix(figure, total)


__all__ = ["show_triscatter_with_acqf"]
