"""Shared layout adjustments for class-probability heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np
from plotly.graph_objs._figure import Figure


def _numeric_axis_range(values: np.ndarray) -> list[float] | None:
    """Return the full heatmap cell extent for a numeric axis."""

    numeric = values.astype(float)
    if not np.isfinite(numeric).all():
        return None
    if numeric.size == 1:
        half_step = max(abs(float(numeric[0])) * 0.05, 0.5)
        return [float(numeric[0] - half_step), float(numeric[0] + half_step)]

    differences = np.diff(numeric)
    nonzero = differences[np.abs(differences) > 0.0]
    if nonzero.size == 0:
        half_step = max(abs(float(numeric[0])) * 0.05, 0.5)
        return [float(numeric[0] - half_step), float(numeric[0] + half_step)]

    fallback_step = float(nonzero[0])
    first_step = float(differences[0]) if differences[0] != 0.0 else fallback_step
    last_step = float(differences[-1]) if differences[-1] != 0.0 else fallback_step
    return [
        float(numeric[0] - first_step / 2.0),
        float(numeric[-1] + last_step / 2.0),
    ]


def _axis_layout(values: Any) -> dict[str, Any]:
    """Build an axis layout whose visible extent matches the heatmap cells."""

    array = np.asarray(values).ravel()
    if array.size == 0:
        return {}

    if np.issubdtype(array.dtype, np.number):
        axis_range = _numeric_axis_range(array)
        if axis_range is None:
            return {}
        return {"autorange": False, "range": axis_range}

    categories = array.tolist()
    return {
        "autorange": False,
        "categoryarray": categories,
        "categoryorder": "array",
        "range": [-0.5, len(categories) - 0.5],
        "type": "category",
    }


def _trace_axis_values(trace: Any, axis: str) -> np.ndarray:
    """Return explicit heatmap axis values or positional defaults."""

    values = getattr(trace, axis, None)
    if values is not None and len(values) > 0:
        return np.asarray(values)

    z = np.asarray(trace.z)
    size = z.shape[1] if axis == "x" else z.shape[0]
    return np.arange(size, dtype=float)


def apply_probability_heatmap_layout(figure: Figure) -> Figure:
    """Align axes and separate the legend from a compact colorbar.

    The axis range is fixed to the outer cell edges rather than Plotly's
    auto-ranged scatter extent. The observed-class legend is placed above the
    plot, while the colorbar is shortened and kept on the right.
    """

    heatmap = next(
        (trace for trace in figure.data if getattr(trace, "type", None) == "heatmap"),
        None,
    )
    if heatmap is None:
        return figure

    heatmap.colorbar.update(
        len=0.5,
        lenmode="fraction",
        thickness=16,
        thicknessmode="pixels",
        x=1.02,
        xanchor="left",
        y=0.5,
        yanchor="middle",
    )

    figure.update_xaxes(**_axis_layout(_trace_axis_values(heatmap, "x")))
    figure.update_yaxes(**_axis_layout(_trace_axis_values(heatmap, "y")))
    figure.update_layout(
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.03,
            yanchor="bottom",
        ),
        margin=dict(l=80, r=125, t=95, b=75),
    )
    return figure


__all__ = ["apply_probability_heatmap_layout"]
