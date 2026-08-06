"""Public ordinal display selectors aligned with multiclass terminology.

One-dimensional ordinal plots default to ordered-category probabilities because
they stay on a bounded and directly interpretable scale.  The latent-score view
remains available with ``ordinal_display="latent"`` for model diagnostics.
"""

from __future__ import annotations

from typing import Any, Literal

from ._heatmap_layout import apply_probability_heatmap_layout
from .categorical_axis import apply_categorical_xaxis_labels
from .multiclass import MulticlassHeatmapMode, is_multiclass_object
from .ordinal import (
    is_ordinal_object,
)
from .ordinal import (
    show_1dplot_from_optimizer as _show_1dplot_from_optimizer,
)
from .ordinal import (
    show_scatter_with_acqf_from_optimizer as _show_scatter_from_optimizer,
)
from .ordinal import (
    show_triscatter_with_acqf_from_optimizer as _show_triscatter_from_optimizer,
)

OrdinalDisplayMode = Literal["latent", "probability"]
OrdinalProbabilityMode = MulticlassHeatmapMode


def _to_internal_display(display: OrdinalDisplayMode) -> str:
    """Translate the public selector to the old internal implementation."""

    if display == "latent":
        return "current"
    if display == "probability":
        return "probability"
    raise ValueError("ordinal_display must be 'latent' or 'probability'.")


def show_1dplot_from_optimizer(
    obj: Any,
    feature: str,
    target: str,
    *,
    ordinal_display: OrdinalDisplayMode = "probability",
    **kwargs: Any,
) -> Any:
    """Plot a 1D response using original labels for categorical feature axes."""

    figure = _show_1dplot_from_optimizer(
        obj,
        feature,
        target,
        ordinal_display=_to_internal_display(ordinal_display),
        **kwargs,
    )
    return apply_categorical_xaxis_labels(figure, obj, feature)


def show_scatter_with_acqf_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    target_col: str,
    *,
    ordinal_display: OrdinalDisplayMode = "latent",
    ordinal_mode: OrdinalProbabilityMode = "class_confidence",
    **kwargs: Any,
) -> Any:
    """Plot a 2D ordinal latent surface or category-probability diagnostic."""

    figure = _show_scatter_from_optimizer(
        obj,
        feature_col1,
        feature_col2,
        target_col,
        ordinal_display=_to_internal_display(ordinal_display),
        ordinal_mode=ordinal_mode,
        **kwargs,
    )
    is_probability_heatmap = kwargs.get("show_type", "acqf") == "pred" and (
        is_multiclass_object(obj)
        or (is_ordinal_object(obj) and ordinal_display == "probability")
    )
    if is_probability_heatmap:
        return apply_probability_heatmap_layout(figure)
    return figure


def show_triscatter_with_acqf_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    *,
    ordinal_display: OrdinalDisplayMode = "latent",
    ordinal_mode: OrdinalProbabilityMode = "class_confidence",
    **kwargs: Any,
) -> Any:
    """Plot a ternary ordinal latent surface or probability diagnostic."""

    return _show_triscatter_from_optimizer(
        obj,
        feature_col1,
        feature_col2,
        feature_col3,
        target_col,
        ordinal_display=_to_internal_display(ordinal_display),
        ordinal_mode=ordinal_mode,
        **kwargs,
    )


__all__ = [
    "OrdinalDisplayMode",
    "OrdinalProbabilityMode",
    "show_1dplot_from_optimizer",
    "show_scatter_with_acqf_from_optimizer",
    "show_triscatter_with_acqf_from_optimizer",
]
