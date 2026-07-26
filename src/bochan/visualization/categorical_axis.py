"""Categorical-axis normalization for optimizer-backed Plotly figures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .utils import decode_values, labels_from


def _ordered_category_labels(mapping: Mapping[Any, Any]) -> list[Any]:
    """Return original labels in the fitted encoder's category-code order."""

    try:
        return [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    except TypeError:
        return list(mapping)


def apply_categorical_xaxis_labels(figure: Any, obj: Any, feature: str) -> Any:
    """Decode marker x-values and configure a categorical Plotly x-axis.

    Optimizer-backed one-dimensional plots evaluate the model in the numeric
    encoded feature space. Their prediction-grid traces are already decoded by
    the grid builders, while observed and candidate marker traces still contain
    fitted category codes. Decode only marker-only traces so numeric category
    labels that overlap with internal codes are not decoded twice.
    """

    mapping = labels_from(obj, feature)
    if not isinstance(mapping, Mapping) or not mapping:
        return figure

    for trace in getattr(figure, "data", ()):
        mode = str(getattr(trace, "mode", "") or "")
        if "markers" not in mode or "lines" in mode:
            continue
        values = getattr(trace, "x", None)
        if values is None:
            continue
        trace.x = decode_values(list(values), mapping)

    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=_ordered_category_labels(mapping),
    )
    return figure


__all__ = ["apply_categorical_xaxis_labels"]
