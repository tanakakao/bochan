from __future__ import annotations

import plotly.graph_objects as go

from bochan.visualization.heatmap_layout import apply_multiclass_heatmap_layout


def test_heatmap_layout_matches_visible_range_and_separates_controls() -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=[0.0, 1.0, 2.0],
            y=[-1.0, 0.0, 1.0],
            z=[[0, 0, 1], [0, 1, 1], [2, 2, 1]],
            colorbar=dict(title="predicted class"),
        )
    )
    # These points would normally widen Plotly autorange beyond the heatmap.
    fig.add_trace(
        go.Scatter(
            x=[-2.0, 3.0],
            y=[-2.0, 2.0],
            mode="markers",
            name="observed: 0",
        )
    )
    fig.update_layout(width=850)

    result = apply_multiclass_heatmap_layout(fig)

    assert list(result.layout.xaxis.range) == [-0.5, 2.5]
    assert list(result.layout.yaxis.range) == [-1.5, 1.5]
    assert result.layout.xaxis.autorange is False
    assert result.layout.yaxis.autorange is False

    colorbar = result.data[0].colorbar
    assert colorbar.x == 1.055
    assert colorbar.y == 0.20
    assert colorbar.len == 0.28
    assert colorbar.thickness == 16
    assert colorbar.title.text == "class"

    assert result.layout.legend.x == 1.02
    assert result.layout.legend.y == 1.0
    assert result.layout.margin.r == 235
    assert result.layout.width == 950


def test_heatmap_layout_preserves_non_class_colorbar_meaning() -> None:
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                z=[[0.1, 0.2], [0.3, 0.4]],
                colorbar=dict(title="normalized entropy"),
            )
        ]
    )

    result = apply_multiclass_heatmap_layout(fig)

    assert result.data[0].colorbar.title.text == "entropy"
