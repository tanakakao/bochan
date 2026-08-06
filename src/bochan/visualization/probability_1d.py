"""Bound probability uncertainty shown in one-dimensional classification plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .heteroscedastic_1d import (
    show_1dplot_from_optimizer as _show_1dplot_from_optimizer,
)
from .utils import get_model


def _target_index(
    target: str,
    target_cols: Sequence[str] | None,
    model: Any,
) -> int:
    """Resolve the selected output index from explicit columns or model specs."""

    columns = list(target_cols or [])
    if target in columns:
        return columns.index(target)

    specs = list(getattr(model, "specs", []) or [])
    for index, spec in enumerate(specs):
        if str(getattr(spec, "name", "")) == target:
            return index
    return 0


def _target_task_type(
    obj: Any,
    target: str,
    target_cols: Sequence[str] | None,
) -> str:
    """Return the task type for a selected output, including hybrid wrappers."""

    model = get_model(obj)
    index = _target_index(target, target_cols, model)

    specs = list(getattr(model, "specs", []) or [])
    if index < len(specs):
        task_type = str(getattr(specs[index], "task_type", "")).lower()
        if task_type:
            return task_type

    bundle = getattr(obj, "bundle", None)
    metadata = getattr(bundle, "metadata", None)
    if isinstance(metadata, dict):
        sub_bundles = list(metadata.get("sub_bundles", []) or [])
        if index < len(sub_bundles):
            task_type = str(
                getattr(sub_bundles[index], "task_type", "")
            ).lower()
            if task_type:
                return task_type

    for candidate in (bundle, getattr(obj, "model_config", None)):
        task_type = str(getattr(candidate, "task_type", "")).lower()
        if task_type and task_type != "hybrid":
            return task_type

    selected_model = model
    models = list(getattr(model, "models", []) or [])
    if index < len(models):
        selected_model = models[index]
    qualified_name = (
        f"{type(selected_model).__module__}.{type(selected_model).__name__}"
        .lower()
    )
    if "binary" in qualified_name:
        return "binary"
    if "multiclass" in qualified_name:
        return "multiclass"
    if "ordinal" in qualified_name:
        return "ordinal"
    return ""


def _numeric_y(values: Any) -> np.ndarray | None:
    """Convert one Plotly y array to float while tolerating non-numeric traces."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _bound_probability_figure(figure: Any) -> Any:
    """Clip binary probability bands and error bars to the valid [0, 1] domain."""

    for trace in figure.data:
        y = _numeric_y(getattr(trace, "y", None))
        if y is None:
            continue
        bounded_y = np.clip(y, 0.0, 1.0)
        trace.y = bounded_y

        error_y = getattr(trace, "error_y", None)
        error_array = getattr(error_y, "array", None) if error_y is not None else None
        if error_array is not None:
            plus = np.abs(np.asarray(error_array, dtype=float))
            minus_source = getattr(error_y, "arrayminus", None)
            minus = plus if minus_source is None else np.abs(
                np.asarray(minus_source, dtype=float)
            )
            error_y.array = np.minimum(plus, np.maximum(1.0 - bounded_y, 0.0))
            error_y.arrayminus = np.minimum(minus, np.maximum(bounded_y, 0.0))
            error_y.symmetric = False

        if getattr(trace, "fill", None) == "tonexty" and "±1σ" in str(
            getattr(trace, "name", "")
        ):
            trace.name = "モデル不確実性 ±1σ（確率範囲内）"

    figure.update_yaxes(range=[0.0, 1.0], autorange=False)
    return figure


def show_1dplot_from_optimizer(
    obj: Any,
    feature: str,
    target: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    n: int = 50,
    cycle: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Render a 1D plot and keep binary uncertainty inside probability bounds."""

    figure = _show_1dplot_from_optimizer(
        obj,
        feature,
        target,
        feature_cols=feature_cols,
        target_cols=target_cols,
        value_dict=value_dict,
        candidate_result=candidate_result,
        n=n,
        cycle=cycle,
        **kwargs,
    )
    if _target_task_type(obj, target, target_cols) == "binary":
        return _bound_probability_figure(figure)
    return figure


__all__ = ["show_1dplot_from_optimizer"]
