"""Optional ordinal class-probability visualizations.

The existing latent / prediction-value plots remain the default.  Set
``ordinal_display='probability'`` on the public optimizer plotting helpers to
switch to ordered-category probability views.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from plotly.graph_objs._figure import Figure

from .data import training_dataframe
from .multiclass import (
    MulticlassHeatmapMode,
    _as_probability_matrix,
    _grid_1d_inputs,
    infer_class_labels,
    show_1dplot_from_optimizer as _show_1dplot_from_optimizer,
    show_multiclass_1dplot,
    show_multiclass_heatmap,
    show_scatter_with_acqf_from_optimizer as _show_scatter_from_optimizer,
)
from .multiclass_ternary import (
    _normalize_ternary,
    _resolve_sum_value,
    _simplex_grid,
    show_multiclass_triscatter,
    show_triscatter_with_acqf_from_optimizer as _show_triscatter_from_optimizer,
)
from .utils import (
    axis_values,
    candidate_result_from,
    decode_values,
    ensure_2d,
    fixed_row_from,
    get_bounds,
    get_model,
    get_train_X,
    infer_feature_cols,
    labels_from,
    to_numpy,
    to_tensor_like,
)

OrdinalDisplayMode = Literal["current", "probability"]
OrdinalProbabilityMode = MulticlassHeatmapMode


def is_ordinal_object(obj: Any) -> bool:
    """Return whether ``obj`` represents an ordinal model."""

    bundle = getattr(obj, "bundle", None)
    task_type = getattr(bundle, "task_type", None)
    if task_type is None:
        config = getattr(obj, "model_config", None)
        task_type = getattr(config, "task_type", None)
    if str(task_type).lower() == "ordinal":
        return True

    model = get_model(obj)
    name = f"{type(model).__module__}.{type(model).__name__}".lower()
    return "ordinal" in name


def _call_probability_function(func: Any, X: Any) -> Any:
    try:
        return func(X=X)
    except TypeError:
        return func(X)


def _select_output(values: Any, output_index: int) -> Any:
    if isinstance(values, (list, tuple)):
        if output_index < 0 or output_index >= len(values):
            raise IndexError(
                f"output_index={output_index} is out of range for "
                f"{len(values)} ordinal outputs."
            )
        return values[output_index]
    return values


def _select_submodel(model: Any, output_index: int) -> Any:
    models = getattr(model, "models", None)
    if models is None:
        return model
    if output_index < 0 or output_index >= len(models):
        raise IndexError(
            f"output_index={output_index} is out of range for "
            f"{len(models)} ordinal models."
        )
    return models[output_index]


def _ordinal_probability_tensor(obj: Any, X: Any, *, output_index: int) -> Any:
    """Evaluate ordered-category probabilities for common ordinal model APIs."""

    import torch

    X_t = to_tensor_like(X, obj)
    model = get_model(obj)
    with torch.no_grad():
        class_probs_list = getattr(model, "class_probs_list", None)
        if callable(class_probs_list):
            return _select_output(
                _call_probability_function(class_probs_list, X_t),
                output_index,
            )

        submodel = _select_submodel(model, output_index)
        class_probs = getattr(submodel, "class_probs", None)
        if callable(class_probs):
            return _call_probability_function(class_probs, X_t)

        posterior = submodel.posterior(X_t)
        likelihood = getattr(submodel, "ordinal_likelihood", None)
        if likelihood is None:
            likelihood = getattr(submodel, "likelihood", None)
        if likelihood is None:
            raise AttributeError(
                "Ordinal probability visualization requires class_probs(), "
                "class_probs_list(), or an ordinal likelihood."
            )

        distribution = getattr(posterior, "distribution", None)
        marginal = getattr(likelihood, "marginal_class_probs", None)
        if callable(marginal) and distribution is not None:
            return marginal(distribution)

        latent_mean = posterior.mean
        from_latent = getattr(likelihood, "class_probs_from_f", None)
        if from_latent is None:
            from_latent = getattr(likelihood, "probs_from_latent", None)
        if callable(from_latent):
            return from_latent(latent_mean)

    raise RuntimeError("Could not evaluate ordinal class probabilities.")


def ordinal_probabilities(
    obj: Any,
    X: Any,
    *,
    output_index: int = 0,
) -> np.ndarray:
    """Return ordinal category probabilities with shape ``[n, K]``."""

    X_arr = ensure_2d(X)
    values = _ordinal_probability_tensor(obj, X, output_index=output_index)
    return _as_probability_matrix(values, n_points=len(X_arr))


def ordinal_prediction_dataframe(
    obj: Any,
    X: Any,
    *,
    output_index: int = 0,
    class_labels: Sequence[Any] | None = None,
    observed_labels: Any | None = None,
) -> pd.DataFrame:
    """Return one ordered-category probability column per class."""

    probabilities = ordinal_probabilities(obj, X, output_index=output_index)
    labels = infer_class_labels(
        obj,
        probabilities.shape[-1],
        class_labels=class_labels,
        output_index=output_index,
        observed_labels=observed_labels,
    )
    return pd.DataFrame(
        probabilities,
        columns=[str(label) for label in labels],
    )


def ordinal_grid_1d(
    obj: Any,
    select_col: str,
    value_dict: Mapping[str, Any] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    output_index: int = 0,
    class_labels: Sequence[Any] | None = None,
    observed_labels: Any | None = None,
    n: int = 100,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a 1D grid and evaluate every ordinal-category probability."""

    grid, x = _grid_1d_inputs(
        obj,
        select_col,
        value_dict,
        feature_cols=feature_cols,
        n=n,
    )
    probabilities = ordinal_prediction_dataframe(
        obj,
        grid,
        output_index=output_index,
        class_labels=class_labels,
        observed_labels=observed_labels,
    )
    return probabilities, x


def _probability_diagnostics(
    probabilities: np.ndarray,
    *,
    shape: tuple[int, ...],
    class_labels: Sequence[Any],
) -> dict[str, Any]:
    predicted_class = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1)
    entropy = entropy / np.log(probabilities.shape[-1])
    sorted_probabilities = np.sort(probabilities, axis=-1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return {
        "class_index": predicted_class.reshape(shape),
        "confidence": confidence.reshape(shape),
        "entropy": entropy.reshape(shape),
        "margin": margin.reshape(shape),
        "probabilities": probabilities.reshape(*shape, probabilities.shape[-1]),
        "class_labels": list(class_labels),
    }


def ordinal_grid_2d(
    obj: Any,
    select_cols: Sequence[str],
    value_dict: Mapping[str, Any] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    output_index: int = 0,
    class_labels: Sequence[Any] | None = None,
    observed_labels: Any | None = None,
    n: int = 50,
) -> dict[str, Any]:
    """Evaluate ordinal probability diagnostics on a 2D grid."""

    if len(select_cols) != 2:
        raise ValueError("select_cols must contain exactly two columns.")

    train_X = get_train_X(obj)
    X_arr = ensure_2d(train_X)
    columns = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    indices = [columns.index(col) for col in select_cols]
    bounds = get_bounds(obj, train_X)
    axes = [
        axis_values(
            obj,
            col=col,
            col_index=index,
            feature_cols=columns,
            n=n,
            train_X=train_X,
            bounds=bounds,
        )
        for col, index in zip(select_cols, indices, strict=False)
    ]
    xx, yy = np.meshgrid(axes[0], axes[1])
    row = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = np.repeat(row, repeats=xx.size, axis=0)
    grid[:, indices] = np.column_stack([xx.ravel(), yy.ravel()])

    probabilities = ordinal_probabilities(
        obj,
        to_tensor_like(grid, obj),
        output_index=output_index,
    )
    labels = infer_class_labels(
        obj,
        probabilities.shape[-1],
        class_labels=class_labels,
        output_index=output_index,
        observed_labels=observed_labels,
    )
    result = _probability_diagnostics(
        probabilities,
        shape=(len(axes[1]), len(axes[0])),
        class_labels=labels,
    )

    decoded_axes = []
    for axis_name, values in zip(select_cols, axes, strict=False):
        mapping = labels_from(obj, axis_name)
        decoded_axes.append(
            np.asarray(
                decode_values(values.tolist(), mapping)
                if mapping is not None
                else values,
                dtype=object if mapping is not None else None,
            )
        )
    result["x"] = decoded_axes[0]
    result["y"] = decoded_axes[1]
    return result


def ordinal_tri_grid(
    obj: Any,
    select_cols: Sequence[str],
    value_dict: Mapping[str, Any] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    output_index: int = 0,
    class_labels: Sequence[Any] | None = None,
    observed_labels: Any | None = None,
    sum_value: float | None = None,
    n: int = 50,
) -> dict[str, Any]:
    """Evaluate ordinal probability diagnostics on a ternary simplex."""

    if len(select_cols) != 3:
        raise ValueError("select_cols must contain exactly three columns.")

    train_X = get_train_X(obj)
    train_array = ensure_2d(train_X)
    columns = [
        column
        for column in infer_feature_cols(obj, feature_cols, train_array.shape[1])
        if column != "task"
    ]
    indices = [columns.index(column) for column in select_cols]
    total = _resolve_sum_value(obj, indices, sum_value)
    simplex_values = _simplex_grid(total, n)

    base = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = np.repeat(base, repeats=len(simplex_values), axis=0)
    grid[:, indices] = simplex_values
    probabilities = ordinal_probabilities(
        obj,
        to_tensor_like(grid, obj),
        output_index=output_index,
    )
    labels = infer_class_labels(
        obj,
        probabilities.shape[-1],
        class_labels=class_labels,
        output_index=output_index,
        observed_labels=observed_labels,
    )
    result = _probability_diagnostics(
        probabilities,
        shape=(len(simplex_values),),
        class_labels=labels,
    )
    result.update(
        {
            "grid": _normalize_ternary(simplex_values),
            "raw_grid": simplex_values,
            "sum_value": total,
        }
    )
    return result


def _resolve_output_index(
    y: pd.DataFrame,
    target: str,
    output_index: int | None,
) -> int:
    if output_index is not None:
        return int(output_index)
    if target not in y.columns:
        raise ValueError(f"target must be one of {list(y.columns)}.")
    return list(y.columns).index(target)


def _candidate_inputs(obj: Any, candidate_result: Any | None) -> Any | None:
    result = candidate_result or candidate_result_from(obj)
    if result is None:
        return None
    return getattr(result, "candidates", None)


def show_ordinal_1dplot_from_optimizer(
    obj: Any,
    feature: str,
    target: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    n: int = 100,
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
) -> Figure:
    """Plot one ordered-category probability line per ordinal class."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    output_index = _resolve_output_index(y_df, target, output_index)
    observed = y_df[target]
    probabilities, x_grid = ordinal_grid_1d(
        obj,
        feature,
        value_dict,
        feature_cols=list(X_df.columns),
        output_index=output_index,
        class_labels=class_labels,
        observed_labels=observed,
        n=n,
    )
    labels = infer_class_labels(
        obj,
        probabilities.shape[-1],
        class_labels=class_labels,
        output_index=output_index,
        observed_labels=observed,
    )

    candidate_X = _candidate_inputs(obj, candidate_result)
    candidate_probabilities = None
    if candidate_X is not None:
        candidate_probabilities = ordinal_probabilities(
            obj,
            candidate_X,
            output_index=output_index,
        )
    return show_multiclass_1dplot(
        feature,
        target,
        (probabilities, x_grid),
        X_df,
        y_df,
        candidate_X=candidate_X,
        candidate_probabilities=candidate_probabilities,
        class_labels=labels,
    )


def show_ordinal_heatmap_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    n: int = 50,
    mode: OrdinalProbabilityMode = "class_confidence",
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
) -> Figure:
    """Create an ordinal ordered-category decision heatmap."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    output_index = _resolve_output_index(y_df, target_col, output_index)
    data = ordinal_grid_2d(
        obj,
        [feature_col1, feature_col2],
        value_dict,
        feature_cols=list(X_df.columns),
        output_index=output_index,
        class_labels=class_labels,
        observed_labels=y_df[target_col],
        n=n,
    )
    return show_multiclass_heatmap(
        feature_col1,
        feature_col2,
        target_col,
        data,
        X_df,
        y_df,
        candidate_X=_candidate_inputs(obj, candidate_result),
        mode=mode,
    )


def show_ordinal_triscatter_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    sum_value: float | None = None,
    n: int = 50,
    mode: OrdinalProbabilityMode = "class_confidence",
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
    marker_size: float = 8.0,
    boundary_margin: float | None = 0.08,
) -> Figure:
    """Create an ordinal ordered-category ternary decision map."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    output_index = _resolve_output_index(y_df, target_col, output_index)
    data = ordinal_tri_grid(
        obj,
        [feature_col1, feature_col2, feature_col3],
        value_dict,
        feature_cols=list(X_df.columns),
        output_index=output_index,
        class_labels=class_labels,
        observed_labels=y_df[target_col],
        sum_value=sum_value,
        n=n,
    )
    return show_multiclass_triscatter(
        feature_col1,
        feature_col2,
        feature_col3,
        target_col,
        data,
        X_df,
        y_df,
        candidate_X=_candidate_inputs(obj, candidate_result),
        mode=mode,
        marker_size=marker_size,
        boundary_margin=boundary_margin,
    )


def _validate_display(display: OrdinalDisplayMode) -> None:
    if display not in {"current", "probability"}:
        raise ValueError("ordinal_display must be 'current' or 'probability'.")


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
    cycle: str | Sequence[Any] | pd.Series | None = None,
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
    ordinal_display: OrdinalDisplayMode = "current",
) -> Figure:
    """Select the current ordinal plot or ordered-category probabilities."""

    _validate_display(ordinal_display)
    if is_ordinal_object(obj) and ordinal_display == "probability":
        return show_ordinal_1dplot_from_optimizer(
            obj,
            feature,
            target,
            feature_cols=feature_cols,
            target_cols=target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            n=n,
            class_labels=class_labels,
            output_index=output_index,
        )
    return _show_1dplot_from_optimizer(
        obj,
        feature,
        target,
        feature_cols=feature_cols,
        target_cols=target_cols,
        value_dict=value_dict,
        candidate_result=candidate_result,
        n=n,
        cycle=cycle,
        class_labels=class_labels,
        output_index=output_index,
    )


def show_scatter_with_acqf_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    n: int = 25,
    show_type: str = "acqf",
    cycle: str | Sequence[Any] | pd.Series | None = None,
    multiclass_mode: MulticlassHeatmapMode = "class_confidence",
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
    ordinal_display: OrdinalDisplayMode = "current",
    ordinal_mode: OrdinalProbabilityMode = "class_confidence",
) -> Figure:
    """Select the current ordinal surface or category-probability map."""

    _validate_display(ordinal_display)
    if (
        is_ordinal_object(obj)
        and show_type == "pred"
        and ordinal_display == "probability"
    ):
        return show_ordinal_heatmap_from_optimizer(
            obj,
            feature_col1,
            feature_col2,
            target_col,
            feature_cols=feature_cols,
            target_cols=target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            n=n,
            mode=ordinal_mode,
            class_labels=class_labels,
            output_index=output_index,
        )
    return _show_scatter_from_optimizer(
        obj,
        feature_col1,
        feature_col2,
        target_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        value_dict=value_dict,
        candidate_result=candidate_result,
        n=n,
        show_type=show_type,
        cycle=cycle,
        multiclass_mode=multiclass_mode,
        class_labels=class_labels,
        output_index=output_index,
    )


def show_triscatter_with_acqf_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    sum_value: float | None = None,
    n: int = 50,
    show_type: str = "acqf",
    cycle: str | Sequence[Any] | pd.Series | None = None,
    ncontours: int = 25,
    multiclass_mode: MulticlassHeatmapMode = "class_confidence",
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
    marker_size: float = 8.0,
    boundary_margin: float | None = 0.08,
    ordinal_display: OrdinalDisplayMode = "current",
    ordinal_mode: OrdinalProbabilityMode = "class_confidence",
) -> Figure:
    """Select the current ordinal ternary plot or category probabilities."""

    _validate_display(ordinal_display)
    if (
        is_ordinal_object(obj)
        and show_type == "pred"
        and ordinal_display == "probability"
    ):
        return show_ordinal_triscatter_from_optimizer(
            obj,
            feature_col1,
            feature_col2,
            feature_col3,
            target_col,
            feature_cols=feature_cols,
            target_cols=target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            sum_value=sum_value,
            n=n,
            mode=ordinal_mode,
            class_labels=class_labels,
            output_index=output_index,
            marker_size=marker_size,
            boundary_margin=boundary_margin,
        )
    return _show_triscatter_from_optimizer(
        obj,
        feature_col1,
        feature_col2,
        feature_col3,
        target_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        value_dict=value_dict,
        candidate_result=candidate_result,
        sum_value=sum_value,
        n=n,
        show_type=show_type,
        cycle=cycle,
        ncontours=ncontours,
        multiclass_mode=multiclass_mode,
        class_labels=class_labels,
        output_index=output_index,
        marker_size=marker_size,
        boundary_margin=boundary_margin,
    )


__all__ = [
    "OrdinalDisplayMode",
    "OrdinalProbabilityMode",
    "is_ordinal_object",
    "ordinal_grid_1d",
    "ordinal_grid_2d",
    "ordinal_prediction_dataframe",
    "ordinal_probabilities",
    "ordinal_tri_grid",
    "show_1dplot_from_optimizer",
    "show_ordinal_1dplot_from_optimizer",
    "show_ordinal_heatmap_from_optimizer",
    "show_ordinal_triscatter_from_optimizer",
    "show_scatter_with_acqf_from_optimizer",
    "show_triscatter_with_acqf_from_optimizer",
]
