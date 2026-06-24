from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from bochan.visualization import ordinal_display
from bochan.visualization._heatmap_layout import apply_probability_heatmap_layout


def _heatmap_figure() -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            x=[0.0, 0.5, 1.0],
            y=[10.0, 20.0],
            z=[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
            colorbar=dict(title="predicted class"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0.25],
            y=[15.0],
            mode="markers",
            name="observed: alpha",
        )
    )
    return figure


def test_probability_heatmap_layout_matches_cell_extent() -> None:
    figure = apply_probability_heatmap_layout(_heatmap_figure())

    np.testing.assert_allclose(figure.layout.xaxis.range, [-0.25, 1.25])
    np.testing.assert_allclose(figure.layout.yaxis.range, [5.0, 25.0])
    assert figure.layout.xaxis.autorange is False
    assert figure.layout.yaxis.autorange is False


def test_probability_heatmap_layout_separates_legend_and_colorbar() -> None:
    figure = apply_probability_heatmap_layout(_heatmap_figure())
    colorbar = figure.data[0].colorbar

    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y > 1.0
    assert colorbar.len == 0.5
    assert colorbar.lenmode == "fraction"
    assert colorbar.thickness == 16
    assert colorbar.y == 0.5


def test_probability_heatmap_layout_handles_category_axes() -> None:
    figure = go.Figure(
        go.Heatmap(
            x=["low", "medium", "high"],
            y=["A", "B"],
            z=[[0, 1, 2], [2, 1, 0]],
        )
    )

    result = apply_probability_heatmap_layout(figure)

    assert result.layout.xaxis.type == "category"
    assert list(result.layout.xaxis.categoryarray) == ["low", "medium", "high"]
    assert list(result.layout.xaxis.range) == [-0.5, 2.5]
    assert list(result.layout.yaxis.range) == [-0.5, 1.5]


def test_public_dispatch_applies_layout_only_to_probability_heatmaps(
    monkeypatch,
) -> None:
    multiclass_obj = SimpleNamespace(
        model_config=SimpleNamespace(task_type="multiclass"),
        model=SimpleNamespace(),
    )

    def return_heatmap(*_args, **_kwargs):
        return _heatmap_figure()

    monkeypatch.setattr(
        ordinal_display,
        "_show_scatter_from_optimizer",
        return_heatmap,
    )

    result = ordinal_display.show_scatter_with_acqf_from_optimizer(
        multiclass_obj,
        "x0",
        "x1",
        "class",
        show_type="pred",
    )

    assert result.layout.legend.orientation == "h"
    assert result.data[0].colorbar.len == 0.5
