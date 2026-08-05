"""Normalize ternary contour coordinates before calling Plotly."""

from __future__ import annotations

from typing import Any

import numpy as np


def _ternary_coordinates(grid: Any) -> np.ndarray:
    """Return Plotly ternary coordinates with shape ``(2|3, n_points)``."""

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


def install_ternary_plot_grid_compat() -> None:
    """Allow DataFrame and row-major arrays in the ternary plotting helper."""

    import bochan.visualization as visualization
    from bochan.visualization import plots

    if getattr(visualization, "_ternary_plot_grid_compat_installed", False):
        return

    original = plots.show_triscatter_with_acqf

    def adapter(
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
            return original(
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
        values, grid = data_tri_plot
        coordinates = _ternary_coordinates(grid)
        n_points = coordinates.shape[1]
        if np.asarray(values).reshape(-1).shape[0] != n_points:
            raise ValueError(
                "Ternary contour values and coordinate points must have the same length."
            )
        return original(
            feature_col1,
            feature_col2,
            feature_col3,
            target_col,
            (values, coordinates),
            X,
            y,
            df_cand,
            **kwargs,
        )

    plots.show_triscatter_with_acqf = adapter
    visualization.show_triscatter_with_acqf = adapter
    visualization._ternary_plot_grid_compat_installed = True


__all__ = ["_ternary_coordinates", "install_ternary_plot_grid_compat"]
