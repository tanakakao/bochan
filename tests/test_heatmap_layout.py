from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from bochan.visualization.heatmap_layout import (
    _numeric_axis_range,
    apply_multiclass_heatmap_layout,
)


def test_numeric_axis_range_uses_heatmap_cell_edges() -> None:
    result = _numeric_axis_range([0.0, 0.5, 1.0])

    assert result == [-0.25, 1.25]


def test_heatmap_layout_matches_axes_and_separates_colorbar() -> None:
    figure = go.Figure(
        data=[
            go.Heatmap(
                x=np.linspace(-1.0, 2.0, 4),
                y=np.linspace(-1.5, 1.5, 4),
                z=np.arange(16).reshape(4, 4),
                colorbar=dict(title="predicted class"),
            ),
            go.Scatter(
                x=[-2.0, 3.0],
                y=[-2.0, 2.0],
                mode="markers",
                name="observed: 0",
            ),
        ]
    )

    result = apply_multiclass_heatmap_layout(figure)

    assert list(result.layout.xaxis.range) == [-1.5, 2.5]
    assert list(result.layout.yaxis.range) == [-2.0, 2.0]
    assert result.layout.xaxis.autorange is False
    assert result.layout.yaxis.autorange is False
    assert result.layout.legend.x == 1.03
    assert result.layout.legend.y == 1.0
    assert result.layout.margin.r == 230

    colorbar = result.data[0].colorbar
    assert colorbar.x == 1.04
    assert colorbar.y == 0.24
    assert colorbar.len == 0.32
    assert colorbar.thickness == 18
