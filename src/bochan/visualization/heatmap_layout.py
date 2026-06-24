"""Layout corrections for multiclass / ordinal Plotly heatmaps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from plotly.graph_objs._figure import Figure


def _numeric_axis_range(values: Sequence[Any]) -> list[float] | None:
    """Return heatmap cell-edge limits for a numeric center-coordinate axis."""

    array = np.asarray(values)
    if array.size == 0:
        return None
    try:
        numeric = array.astype(float).ravel()
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric).all():
        return None
    if numeric.size == 1:
        return [float(numeric[0] - 0.5), float(numeric[0] + 0.5)]

    ordered = np.unique(np.sort(numeric))
    if ordered.size == 1:
        return [float(ordered[0] - 0.5), float(ordered[0] + 0.5)]

    lower_step = ordered[1] - ordered[0]
    upper_step = ordered[-1] - ordered[-2]
    return [
        float(ordered[0] - lower_step / 2.0),
        float(ordered[-1] + upper_step / 2.0),
    ]


def _compact_colorbar_title(title: Any) -> str:
    """Shorten verbose colorbar titles while preserving their meaning."""

    text = str(title or "")
    replacements = {
        "predicted class": "class",
        "normalized entropy": "entropy",
        "top-2 probability margin": "margin",
    }
    return replacements.get(text, text)


def apply_multiclass_heatmap_layout(fig: Figure) -> Figure:
    """Align axes with the heatmap and separate legend from a compact colorbar."""

    heatmap = next((trace for trace in fig.data if trace.type == "heatmap"), None)
    if heatmap is None:
        return fig

    x_range = _numeric_axis_range(heatmap.x)
    y_range = _numeric_axis_range(heatmap.y)
    if x_range is not None:
        fig.update_xaxes(range=x_range, autorange=False)
    if y_range is not None:
        fig.update_yaxes(range=y_range, autorange=False)

    heatmap.update(
        colorbar=dict(
            x=1.055,
            xanchor="left",
            y=0.20,
            yanchor="middle",
            len=0.28,
            lenmode="fraction",
            thickness=16,
            thicknessmode="pixels",
            outlinewidth=0,
            title=dict(
                text=_compact_colorbar_title(heatmap.colorbar.title.text),
                side="top",
            ),
        )
    )
    fig.update_layout(
        width=max(int(fig.layout.width or 850), 950),
        margin=dict(l=70, r=235, t=55, b=70),
        legend=dict(
            title=dict(text="observed class"),
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.82)",
        ),
    )
    return fig


def install_multiclass_heatmap_layout_patch(multiclass_module: Any) -> None:
    """Install the layout correction without changing the public API."""

    original = multiclass_module.show_multiclass_heatmap
    if getattr(original, "_bochan_layout_patched", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Figure:
        return apply_multiclass_heatmap_layout(original(*args, **kwargs))

    wrapped._bochan_layout_patched = True  # type: ignore[attr-defined]
    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    multiclass_module.show_multiclass_heatmap = wrapped


__all__ = [
    "apply_multiclass_heatmap_layout",
    "install_multiclass_heatmap_layout_patch",
]
