"""Multiclass probability visualizations for :mod:`bochan.visualization`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.graph_objs._figure import Figure

from .data import training_dataframe
from .plots import (
    show_1dplot_from_optimizer as _show_1dplot_from_optimizer,
    show_scatter_with_acqf_from_optimizer as _show_scatter_with_acqf_from_optimizer,
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

MulticlassHeatmapMode = Literal[
    "class_confidence",
    "class",
    "entropy",
    "margin",
]


def is_multiclass_object(obj: Any) -> bool:
    """Return whether ``obj`` represents a multiclass classification model."""

    bundle = getattr(obj, "bundle", None)
    task_type = getattr(bundle, "task_type", None)
    if task_type is None:
        config = getattr(obj, "model_config", None)
        task_type = getattr(config, "task_type", None)
    if str(task_type).lower() == "multiclass":
        return True

    model = get_model(obj)
    name = f"{type(model).__module__}.{type(model).__name__}".lower()
    return "multiclass" in name


def _is_multioutput_multiclass(model: Any) -> bool:
    return callable(getattr(model, "class_probs_list", None)) or (
        "multioutput" in type(model).__name__.lower()
        and "multiclass" in type(model).__name__.lower()
    )


def _call_probability_function(func: Any, X: Any) -> Any:
    try:
        return func(X=X)
    except TypeError:
        return func(X)


def _probability_tensor(obj: Any, X: Any, *, output_index: int = 0) -> Any:
    """Evaluate class probabilities while supporting single and multi-output models."""

    import torch

    X_t = to_tensor_like(X, obj)
    model = get_model(obj)
    with torch.no_grad():
        class_probs_list = getattr(model, "class_probs_list", None)
        if callable(class_probs_list):
            values = _call_probability_function(class_probs_list, X_t)
            if not isinstance(values, (list, tuple)):
                raise TypeError("class_probs_list() must return a list or tuple.")
            if output_index < 0 or output_index >= len(values):
                raise IndexError(
                    f"output_index={output_index} is out of range for "
                    f"{len(values)} multiclass outputs."
                )
            return values[output_index]

        class_probs = getattr(model, "class_probs", None)
        if callable(class_probs):
            values = _call_probability_function(class_probs, X_t)
        else:
            posterior = model.posterior(X_t)
            values = posterior.mean

        if _is_multioutput_multiclass(model):
            values = torch.as_tensor(values)
            if values.ndim < 3:
                raise RuntimeError(
                    "A multi-output multiclass model must return probabilities "
                    "with shape [..., n, m, C]."
                )
            if output_index < 0 or output_index >= values.shape[-2]:
                raise IndexError(
                    f"output_index={output_index} is out of range for "
                    f"probability shape={tuple(values.shape)}."
                )
            values = values[..., output_index, :]
        return values


def _as_probability_matrix(values: Any, *, n_points: int) -> np.ndarray:
    """Normalize probability-like values to an ``n_points x n_classes`` array."""

    arr = np.asarray(to_numpy(values), dtype=float)
    if arr.ndim == 0:
        raise RuntimeError("Multiclass probabilities must have at least one dimension.")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    while arr.ndim > 2 and arr.shape[-2] == 1:
        arr = np.squeeze(arr, axis=-2)

    if arr.ndim > 2:
        if arr.size % (n_points * arr.shape[-1]) != 0:
            raise RuntimeError(
                "Could not normalize multiclass probability shape. "
                f"Got shape={arr.shape}, n_points={n_points}."
            )
        arr = arr.reshape(-1, n_points, arr.shape[-1]).mean(axis=0)

    if arr.ndim != 2:
        raise RuntimeError(
            "Multiclass probabilities must normalize to shape [n, C]. "
            f"Got shape={arr.shape}."
        )
    if arr.shape[0] != n_points and arr.shape[1] == n_points:
        arr = arr.T
    if arr.shape[0] != n_points:
        raise RuntimeError(
            "The number of probability rows does not match the number of inputs. "
            f"Got probabilities={arr.shape[0]}, inputs={n_points}."
        )
    if arr.shape[1] < 2:
        raise RuntimeError(
            "Multiclass visualization requires at least two probability columns."
        )

    finite = np.isfinite(arr).all()
    row_sums = arr.sum(axis=-1, keepdims=True)
    on_simplex = (
        finite
        and np.nanmin(arr) >= -1e-6
        and np.nanmax(arr) <= 1.0 + 1e-6
        and np.allclose(row_sums, 1.0, atol=1e-4, rtol=1e-4)
    )
    if not on_simplex:
        shifted = arr - np.nanmax(arr, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        arr = exp_values / np.clip(exp_values.sum(axis=-1, keepdims=True), 1e-12, None)
    else:
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr / np.clip(arr.sum(axis=-1, keepdims=True), 1e-12, None)
    return arr


def multiclass_probabilities(
    obj: Any,
    X: Any,
    *,
    output_index: int = 0,
) -> np.ndarray:
    """Return class probabilities with shape ``[n, C]``."""

    X_arr = ensure_2d(X)
    values = _probability_tensor(obj, X, output_index=output_index)
    return _as_probability_matrix(values, n_points=len(X_arr))


def _metadata_values(obj: Any, keys: Sequence[str]) -> Any | None:
    for candidate in (obj, getattr(obj, "bundle", None), get_model(obj)):
        if candidate is None:
            continue
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in keys:
                if metadata.get(key) is not None:
                    return metadata[key]
        for key in keys:
            value = getattr(candidate, key, None)
            if value is not None:
                return value
    return None


def infer_class_labels(
    obj: Any,
    n_classes: int,
    *,
    class_labels: Sequence[Any] | None = None,
    output_index: int = 0,
    observed_labels: Any | None = None,
) -> list[Any]:
    """Infer display labels for probability columns."""

    raw = class_labels
    if raw is None:
        raw = _metadata_values(
            obj,
            ("class_labels", "class_names", "classes", "classes_"),
        )

    if isinstance(raw, Mapping):
        raw = raw.get(output_index, raw.get(str(output_index)))
    if raw is not None:
        values = list(raw)
        if values and isinstance(values[0], (list, tuple, np.ndarray)):
            if output_index >= len(values):
                raise IndexError(
                    f"output_index={output_index} is out of range for class_labels."
                )
            values = list(values[output_index])
        if len(values) != n_classes:
            raise ValueError(
                f"class_labels must contain {n_classes} labels. Got {len(values)}."
            )
        return values

    if observed_labels is not None:
        observed = np.ravel(to_numpy(observed_labels))
        unique = [value for value in pd.unique(observed) if not pd.isna(value)]
        try:
            unique = sorted(unique)
        except TypeError:
            pass
        if len(unique) == n_classes:
            return unique

    return list(range(n_classes))


def multiclass_prediction_dataframe(
    obj: Any,
    X: Any,
    *,
    output_index: int = 0,
    class_labels: Sequence[Any] | None = None,
    observed_labels: Any | None = None,
) -> pd.DataFrame:
    """Return one probability column per class."""

    probabilities = multiclass_probabilities(obj, X, output_index=output_index)
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


def _grid_1d_inputs(
    obj: Any,
    select_col: str,
    value_dict: Mapping[str, Any] | None,
    *,
    feature_cols: Sequence[str] | None,
    n: int,
) -> tuple[Any, np.ndarray]:
    train_X = get_train_X(obj)
    X_arr = ensure_2d(train_X)
    columns = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    if select_col not in columns:
        raise ValueError(f"select_col must be one of {columns}.")
    index = columns.index(select_col)
    x = axis_values(
        obj,
        col=select_col,
        col_index=index,
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=get_bounds(obj, train_X),
    )
    row = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = np.repeat(row, repeats=len(x), axis=0)
    grid[:, index] = x
    mapping = labels_from(obj, select_col)
    display_x = np.asarray(
        decode_values(x.tolist(), mapping) if mapping is not None else x,
        dtype=object if mapping is not None else None,
    )
    return to_tensor_like(grid, obj), display_x


def multiclass_grid_1d(
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
    """Build a 1D grid and evaluate every class probability."""

    grid, x = _grid_1d_inputs(
        obj,
        select_col,
        value_dict,
        feature_cols=feature_cols,
        n=n,
    )
    probabilities = multiclass_prediction_dataframe(
        obj,
        grid,
        output_index=output_index,
        class_labels=class_labels,
        observed_labels=observed_labels,
    )
    return probabilities, x


def multiclass_grid_2d(
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
    """Evaluate class, confidence, entropy, margin, and probabilities on a 2D grid."""

    if len(select_cols) != 2:
        raise ValueError("select_cols must contain exactly two columns.")

    train_X = get_train_X(obj)
    X_arr = ensure_2d(train_X)
    columns = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    indices = [columns.index(col) for col in select_cols]
    bounds = get_bounds(obj, train_X)
    x1 = axis_values(
        obj,
        col=select_cols[0],
        col_index=indices[0],
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=bounds,
    )
    x2 = axis_values(
        obj,
        col=select_cols[1],
        col_index=indices[1],
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=bounds,
    )
    xx, yy = np.meshgrid(x1, x2)
    row = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = np.repeat(row, repeats=xx.size, axis=0)
    grid[:, indices] = np.column_stack([xx.ravel(), yy.ravel()])

    probabilities = multiclass_probabilities(
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
    predicted_class = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1)
    if probabilities.shape[-1] > 1:
        entropy = entropy / np.log(probabilities.shape[-1])
        sorted_probabilities = np.sort(probabilities, axis=-1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    else:  # pragma: no cover - protected by probability validation
        margin = np.ones_like(confidence)

    shape = (len(x2), len(x1))
    decoded_axes = []
    for axis_name, values in zip(select_cols, (x1, x2), strict=False):
        mapping = labels_from(obj, axis_name)
        decoded_axes.append(
            np.asarray(
                decode_values(values.tolist(), mapping)
                if mapping is not None
                else values,
                dtype=object if mapping is not None else None,
            )
        )

    return {
        "class_index": predicted_class.reshape(shape),
        "confidence": confidence.reshape(shape),
        "entropy": entropy.reshape(shape),
        "margin": margin.reshape(shape),
        "probabilities": probabilities.reshape(*shape, probabilities.shape[-1]),
        "x": decoded_axes[0],
        "y": decoded_axes[1],
        "class_labels": labels,
    }


def _class_colors(n_classes: int) -> list[str]:
    palette = list(qualitative.Plotly) + list(qualitative.Dark24)
    return [palette[index % len(palette)] for index in range(n_classes)]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a hexadecimal color, got {color!r}.")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _blend_with_white(color: str, strength: float) -> str:
    red, green, blue = _hex_to_rgb(color)
    strength = float(np.clip(strength, 0.0, 1.0))
    mixed = [round(255 - (255 - channel) * strength) for channel in (red, green, blue)]
    return f"rgb({mixed[0]},{mixed[1]},{mixed[2]})"


def _class_colorscale(colors: Sequence[str], *, shade_confidence: bool) -> list[list[Any]]:
    scale: list[list[Any]] = []
    n_classes = len(colors)
    for index, color in enumerate(colors):
        start = index / n_classes
        end = (index + 1) / n_classes
        low = _blend_with_white(color, 0.2) if shade_confidence else color
        high = color
        scale.append([start, low])
        scale.append([end, high])
    return scale


def _class_indices(values: Any, class_labels: Sequence[Any]) -> np.ndarray:
    raw = np.ravel(to_numpy(values))
    mapping = {label: index for index, label in enumerate(class_labels)}
    string_mapping = {str(label): index for index, label in enumerate(class_labels)}
    result = np.full(raw.shape, -1, dtype=int)
    for index, value in enumerate(raw):
        if value in mapping:
            result[index] = mapping[value]
        elif str(value) in string_mapping:
            result[index] = string_mapping[str(value)]
        else:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= numeric < len(class_labels):
                result[index] = numeric
    return result


def show_multiclass_1dplot(
    feature: str,
    target: str,
    data_1d_plot: tuple[pd.DataFrame, np.ndarray],
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    candidate_X: Any | None = None,
    candidate_probabilities: np.ndarray | None = None,
    class_labels: Sequence[Any] | None = None,
) -> Figure:
    """Plot one predicted-probability line per class."""

    probability_df, x_grid = data_1d_plot
    labels = list(class_labels or probability_df.columns)
    if len(labels) != probability_df.shape[1]:
        raise ValueError("class_labels length must match probability columns.")
    colors = _class_colors(len(labels))

    x_values = np.asarray(x_grid).ravel()
    order = np.arange(len(x_values))
    try:
        order = np.argsort(x_values.astype(float))
    except (TypeError, ValueError):
        pass

    fig = go.Figure()
    for class_index, (label, column, color) in enumerate(
        zip(labels, probability_df.columns, colors, strict=False)
    ):
        probability = pd.to_numeric(probability_df[column], errors="coerce").to_numpy()
        fig.add_trace(
            go.Scatter(
                x=x_values[order],
                y=probability[order],
                mode="lines",
                name=f"P({target}={label})",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"{feature}: %{{x}}<br>"
                    f"class: {label}<br>"
                    "probability: %{y:.3f}<extra></extra>"
                ),
            )
        )

    observed_indices = _class_indices(y[target], labels)
    for class_index, (label, color) in enumerate(zip(labels, colors, strict=False)):
        mask = observed_indices == class_index
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=X.loc[mask, feature],
                    y=np.full(int(mask.sum()), 0.015),
                    mode="markers",
                    name=f"observed: {label}",
                    marker=dict(
                        color=color,
                        size=9,
                        symbol="line-ns-open",
                        line=dict(width=2, color=color),
                    ),
                    hovertemplate=(
                        f"{feature}: %{{x}}<br>observed class: {label}"
                        "<extra></extra>"
                    ),
                )
            )

    if candidate_X is not None and candidate_probabilities is not None:
        candidate_arr = ensure_2d(candidate_X)
        if feature in X.columns:
            feature_index = list(X.columns).index(feature)
            candidate_class = candidate_probabilities.argmax(axis=-1)
            candidate_confidence = candidate_probabilities.max(axis=-1)
            fig.add_trace(
                go.Scatter(
                    x=candidate_arr[:, feature_index],
                    y=candidate_confidence,
                    mode="markers",
                    name="candidates",
                    marker=dict(
                        color=[colors[index] for index in candidate_class],
                        size=12,
                        symbol="diamond",
                        line=dict(width=0.8, color="black"),
                    ),
                    customdata=np.column_stack(
                        [
                            np.asarray(labels, dtype=object)[candidate_class],
                            candidate_confidence,
                        ]
                    ),
                    hovertemplate=(
                        f"{feature}: %{{x}}<br>"
                        "predicted class: %{customdata[0]}<br>"
                        "confidence: %{customdata[1]:.3f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=600,
        width=850,
        xaxis_title=feature,
        yaxis_title="predicted probability",
        legend_title_text="class",
        font_size=16,
    )
    fig.update_yaxes(range=[0.0, 1.0])
    return fig


def _heatmap_customdata(data: Mapping[str, Any]) -> np.ndarray:
    class_index = np.asarray(data["class_index"], dtype=int)
    labels = np.asarray(data["class_labels"], dtype=object)
    customdata = np.empty((*class_index.shape, 4), dtype=object)
    customdata[..., 0] = labels[class_index]
    customdata[..., 1] = np.asarray(data["confidence"], dtype=float)
    customdata[..., 2] = np.asarray(data["entropy"], dtype=float)
    customdata[..., 3] = np.asarray(data["margin"], dtype=float)
    return customdata


def show_multiclass_heatmap(
    feature_col1: str,
    feature_col2: str,
    target_col: str,
    data_2d_plot: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    candidate_X: Any | None = None,
    mode: MulticlassHeatmapMode = "class_confidence",
) -> Figure:
    """Plot a multiclass decision map.

    ``class_confidence`` uses hue for the predicted class and saturation for the
    maximum class probability. ``entropy`` and ``margin`` provide continuous
    uncertainty views that are useful around decision boundaries.
    """

    valid_modes = {"class_confidence", "class", "entropy", "margin"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}.")

    class_index = np.asarray(data_2d_plot["class_index"], dtype=int)
    confidence = np.asarray(data_2d_plot["confidence"], dtype=float)
    labels = list(data_2d_plot["class_labels"])
    x_values = np.asarray(data_2d_plot["x"]).ravel()
    y_values = np.asarray(data_2d_plot["y"]).ravel()
    colors = _class_colors(len(labels))
    customdata = _heatmap_customdata(data_2d_plot)

    if mode == "class_confidence":
        chance = 1.0 / len(labels)
        strength = (confidence - chance) / max(1.0 - chance, 1e-12)
        strength = np.clip(strength, 0.0, 1.0 - 1e-9)
        z = class_index + strength
        colorscale = _class_colorscale(colors, shade_confidence=True)
        colorbar = dict(
            title="predicted class",
            tickmode="array",
            tickvals=[index + 0.5 for index in range(len(labels))],
            ticktext=[str(label) for label in labels],
        )
        zmin, zmax = 0.0, float(len(labels))
    elif mode == "class":
        z = class_index + 0.5
        colorscale = _class_colorscale(colors, shade_confidence=False)
        colorbar = dict(
            title="predicted class",
            tickmode="array",
            tickvals=[index + 0.5 for index in range(len(labels))],
            ticktext=[str(label) for label in labels],
        )
        zmin, zmax = 0.0, float(len(labels))
    elif mode == "entropy":
        z = np.asarray(data_2d_plot["entropy"], dtype=float)
        colorscale = "Viridis"
        colorbar = dict(title="normalized entropy")
        zmin, zmax = 0.0, 1.0
    else:
        z = np.asarray(data_2d_plot["margin"], dtype=float)
        colorscale = "Blues"
        colorbar = dict(title="top-2 probability margin")
        zmin, zmax = 0.0, 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_values,
            y=y_values,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar=colorbar,
            customdata=customdata,
            hovertemplate=(
                f"{feature_col1}: %{{x}}<br>"
                f"{feature_col2}: %{{y}}<br>"
                "predicted class: %{customdata[0]}<br>"
                "max probability: %{customdata[1]:.3f}<br>"
                "normalized entropy: %{customdata[2]:.3f}<br>"
                "top-2 margin: %{customdata[3]:.3f}<extra></extra>"
            ),
        )
    )

    if len(labels) > 1:
        fig.add_trace(
            go.Contour(
                z=class_index,
                x=x_values,
                y=y_values,
                contours=dict(
                    start=0.5,
                    end=len(labels) - 1.5,
                    size=1.0,
                    coloring="none",
                ),
                line=dict(color="rgba(40,40,40,0.75)", width=1.1),
                showscale=False,
                hoverinfo="skip",
            )
        )

    observed_indices = _class_indices(y[target_col], labels)
    for class_number, (label, color) in enumerate(zip(labels, colors, strict=False)):
        mask = observed_indices == class_number
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=X.loc[mask, feature_col1],
                    y=X.loc[mask, feature_col2],
                    mode="markers",
                    name=f"observed: {label}",
                    marker=dict(
                        color=color,
                        size=9,
                        line=dict(width=0.8, color="black"),
                    ),
                    hovertemplate=(
                        f"{feature_col1}: %{{x}}<br>"
                        f"{feature_col2}: %{{y}}<br>"
                        f"observed class: {label}<extra></extra>"
                    ),
                )
            )

    if candidate_X is not None:
        candidate_arr = ensure_2d(candidate_X)
        feature_columns = list(X.columns)
        fig.add_trace(
            go.Scatter(
                x=candidate_arr[:, feature_columns.index(feature_col1)],
                y=candidate_arr[:, feature_columns.index(feature_col2)],
                mode="markers",
                name="candidates",
                marker=dict(
                    color="white",
                    size=12,
                    symbol="diamond",
                    line=dict(width=1.5, color="black"),
                ),
            )
        )

    fig.update_layout(
        height=650,
        width=850,
        xaxis_title=feature_col1,
        yaxis_title=feature_col2,
        legend_title_text="observed class",
        font_size=15,
    )
    return fig


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


def show_multiclass_1dplot_from_optimizer(
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
    """Create a 1D all-class probability plot from an optimizer."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    output_index = _resolve_output_index(y_df, target, output_index)
    observed = y_df[target]
    probabilities, x_grid = multiclass_grid_1d(
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
        candidate_probabilities = multiclass_probabilities(
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


def show_multiclass_heatmap_from_optimizer(
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
    mode: MulticlassHeatmapMode = "class_confidence",
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
) -> Figure:
    """Create a multiclass decision heatmap from an optimizer."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    output_index = _resolve_output_index(y_df, target_col, output_index)
    observed = y_df[target_col]
    data = multiclass_grid_2d(
        obj,
        [feature_col1, feature_col2],
        value_dict,
        feature_cols=list(X_df.columns),
        output_index=output_index,
        class_labels=class_labels,
        observed_labels=observed,
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
) -> Figure:
    """Dispatch to multiclass probability lines or the existing plot."""

    if is_multiclass_object(obj):
        return show_multiclass_1dplot_from_optimizer(
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
) -> Figure:
    """Dispatch multiclass prediction maps while preserving acquisition plots."""

    if is_multiclass_object(obj) and show_type == "pred":
        return show_multiclass_heatmap_from_optimizer(
            obj,
            feature_col1,
            feature_col2,
            target_col,
            feature_cols=feature_cols,
            target_cols=target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            n=n,
            mode=multiclass_mode,
            class_labels=class_labels,
            output_index=output_index,
        )
    return _show_scatter_with_acqf_from_optimizer(
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
    )


__all__ = [
    "MulticlassHeatmapMode",
    "infer_class_labels",
    "is_multiclass_object",
    "multiclass_grid_1d",
    "multiclass_grid_2d",
    "multiclass_prediction_dataframe",
    "multiclass_probabilities",
    "show_1dplot_from_optimizer",
    "show_multiclass_1dplot",
    "show_multiclass_1dplot_from_optimizer",
    "show_multiclass_heatmap",
    "show_multiclass_heatmap_from_optimizer",
    "show_scatter_with_acqf_from_optimizer",
]
