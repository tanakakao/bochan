"""Normalize ternary contour coordinates before calling Plotly."""

from __future__ import annotations

from typing import Any

import numpy as np


def _ternary_coordinates(grid: Any) -> np.ndarray:
    """Return ternary coordinates with shape ``(2|3, n_points)``."""

    try:
        import pandas as pd

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
    """Normalize a fixed-total ternary slice to Plotly's unit simplex."""

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


def _scale_ternary_contour_traces(figure: Any, total: float) -> Any:
    """Restore normalized contour traces to the requested slice total."""

    if np.isclose(total, 1.0):
        return figure
    for trace in figure.data:
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


def install_ternary_plot_grid_compat() -> None:
    """Allow row-major and non-unit-total grids in ternary plotting helpers."""

    import plotly.figure_factory as ff

    import bochan.visualization as visualization
    from bochan.visualization import plots

    if getattr(visualization, "_ternary_plot_grid_compat_installed", False):
        return

    original_plot = plots.show_triscatter_with_acqf
    original_contour = ff.create_ternary_contour

    def contour_adapter(
        coordinates: Any,
        values: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        oriented = _ternary_coordinates(coordinates)
        normalized, total = _normalized_ternary_coordinates(oriented)
        figure = original_contour(normalized, values, *args, **kwargs)
        return _scale_ternary_contour_traces(figure, total)

    def plot_adapter(
        feature_col1: str,
        feature_col2: str,
        feature_col3: str,
        target_col: str,
        data_tri_plot: tuple[np.ndarray, Any],
        X: Any,
        y: Any,
        df_cand: Any = None,
        **kwargs: Any,
    ) -> Any:
        if not (
            isinstance(data_tri_plot, (tuple, list))
            and len(data_tri_plot) == 2
        ):
            return original_plot(
                feature_col1,
                feature_col2,
                feature_col3,
                target_col,
                data_tri_plot,
                X,
                y,
                df_cand,
                **kwargs,
            )
        contour_values, grid = data_tri_plot
        coordinates = _ternary_coordinates(grid)
        n_points = coordinates.shape[1]
        if np.asarray(contour_values).reshape(-1).shape[0] != n_points:
            raise ValueError(
                "Ternary contour values and coordinate points must have the same length."
            )
        return original_plot(
            feature_col1,
            feature_col2,
            feature_col3,
            target_col,
            (contour_values, coordinates),
            X,
            y,
            df_cand,
            **kwargs,
        )

    ff.create_ternary_contour = contour_adapter
    plots.show_triscatter_with_acqf = plot_adapter
    visualization.show_triscatter_with_acqf = plot_adapter
    visualization._ternary_plot_grid_compat_installed = True


__all__ = [
    "_normalized_ternary_coordinates",
    "_ternary_coordinates",
    "install_ternary_plot_grid_compat",
]
