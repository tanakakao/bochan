from __future__ import annotations

import numpy as np
import pandas as pd

from bochan.visualization import show_triscatter_with_acqf


def _simplex_grid(levels: int = 5) -> np.ndarray:
    """Return barycentric coordinates with shape ``(3, n_points)``."""

    points: list[tuple[float, float, float]] = []
    for i in range(levels + 1):
        for j in range(levels + 1 - i):
            k = levels - i - j
            points.append((i / levels, j / levels, k / levels))
    return np.asarray(points, dtype=float).T


def test_ternary_contour_can_be_rendered_with_visualization_extra() -> None:
    """The visualization extra must include Plotly's ternary contour dependency."""

    grid = _simplex_grid()
    values = grid[0] + 2.0 * grid[1] + 3.0 * grid[2]
    features = ["a", "b", "c"]
    x = pd.DataFrame(grid.T, columns=features)
    y = pd.DataFrame({"target": values})

    figure = show_triscatter_with_acqf(
        "a",
        "b",
        "c",
        "target",
        (values, grid),
        x,
        y,
        ncontours=5,
        show_type="pred",
    )

    assert figure.data
    assert any(trace.type == "scatterternary" for trace in figure.data)
