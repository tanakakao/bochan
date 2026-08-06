"""Bounded probability views for one-dimensional discrete-output plots."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

from .heteroscedastic_1d import (
    show_1dplot_from_optimizer as _show_1dplot_from_optimizer,
)
from .utils import get_model

_DISCRETE_TASKS = {"binary", "multiclass", "ordinal"}


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


def _sub_bundles(obj: Any) -> list[Any]:
    """Return target-specific bundles stored by a hybrid model build."""

    bundle = getattr(obj, "bundle", None)
    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    return list(metadata.get("sub_bundles", []) or [])


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

    sub_bundles = _sub_bundles(obj)
    if index < len(sub_bundles):
        task_type = str(getattr(sub_bundles[index], "task_type", "")).lower()
        if task_type:
            return task_type

    bundle = getattr(obj, "bundle", None)
    for candidate in (bundle, getattr(obj, "model_config", None)):
        task_type = str(getattr(candidate, "task_type", "")).lower()
        if task_type and task_type != "hybrid":
            return task_type

    selected_model = _selected_model(model, index)
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


def _selected_model(model: Any, index: int) -> Any:
    """Return one target model from a hybrid wrapper."""

    specs = list(getattr(model, "specs", []) or [])
    if index < len(specs):
        selected = getattr(specs[index], "model", None)
        if selected is not None:
            return selected
    models = list(getattr(model, "models", []) or [])
    if index < len(models):
        return models[index]
    return model


def _selected_train_y(obj: Any, index: int) -> Any:
    """Select one target column while preserving a two-dimensional output shape."""

    train_y = getattr(obj, "train_Y", None)
    if train_y is None:
        return None
    ndim = int(getattr(train_y, "ndim", np.asarray(train_y).ndim))
    if ndim <= 1:
        try:
            return train_y.reshape(-1, 1)
        except AttributeError:
            return np.asarray(train_y).reshape(-1, 1)
    return train_y[..., index : index + 1]


def _discrete_target_proxy(
    obj: Any,
    *,
    target: str,
    target_cols: Sequence[str] | None,
    task_type: str,
) -> Any:
    """Expose one hybrid discrete output as a regular single-output optimizer."""

    model = get_model(obj)
    index = _target_index(target, target_cols, model)
    selected_model = _selected_model(model, index)
    sub_bundles = _sub_bundles(obj)
    selected_bundle = sub_bundles[index] if index < len(sub_bundles) else None

    original_bundle = getattr(obj, "bundle", None)
    metadata: dict[str, Any] = {}
    original_metadata = getattr(original_bundle, "metadata", None)
    if isinstance(original_metadata, dict):
        metadata.update(original_metadata)
    selected_metadata = getattr(selected_bundle, "metadata", None)
    if isinstance(selected_metadata, dict):
        metadata.update(selected_metadata)
    metadata["target_cols"] = [target]

    cat_dims = getattr(selected_bundle, "cat_dims", None)
    if cat_dims is None:
        cat_dims = getattr(original_bundle, "cat_dims", [])
    model_config = getattr(selected_bundle, "model_config", None)
    if model_config is None:
        model_config = SimpleNamespace(task_type=task_type)

    bundle = SimpleNamespace(
        model=selected_model,
        task_type=task_type,
        metadata=metadata,
        cat_dims=list(cat_dims or []),
        model_config=model_config,
    )
    return SimpleNamespace(
        model=selected_model,
        bundle=bundle,
        model_config=model_config,
        train_X=getattr(obj, "train_X", None),
        train_Y=_selected_train_y(obj, index),
        bounds=getattr(obj, "bounds", None),
        data_context=getattr(obj, "data_context", None),
        labels=getattr(obj, "labels", None),
    )


def _is_hybrid_discrete_target(
    obj: Any,
    task_type: str,
) -> bool:
    """Return whether a discrete target is wrapped in a hybrid output model."""

    if task_type not in _DISCRETE_TASKS:
        return False
    model = get_model(obj)
    if list(getattr(model, "specs", []) or []):
        return True
    bundle_task = str(getattr(getattr(obj, "bundle", None), "task_type", "")).lower()
    return bundle_task == "hybrid"


def _numeric_y(values: Any) -> np.ndarray | None:
    """Convert one Plotly y array to float while tolerating non-numeric traces."""

    if values is None:
        return None
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
    """Render discrete targets in probability space with bounded uncertainty."""

    task_type = _target_task_type(obj, target, target_cols)
    plot_obj = obj
    plot_target_cols = target_cols
    if _is_hybrid_discrete_target(obj, task_type):
        plot_obj = _discrete_target_proxy(
            obj,
            target=target,
            target_cols=target_cols,
            task_type=task_type,
        )
        plot_target_cols = [target]

    figure = _show_1dplot_from_optimizer(
        plot_obj,
        feature,
        target,
        feature_cols=feature_cols,
        target_cols=plot_target_cols,
        value_dict=value_dict,
        candidate_result=candidate_result,
        n=n,
        cycle=cycle,
        **kwargs,
    )
    if task_type == "binary":
        return _bound_probability_figure(figure)
    return figure


__all__ = ["show_1dplot_from_optimizer"]
