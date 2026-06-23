"""Multiclass ternary probability visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.graph_objs._figure import Figure

from .data import training_dataframe
from .multiclass import (
    MulticlassHeatmapMode,
    infer_class_labels,
    is_multiclass_object,
    multiclass_probabilities,
)
from .plots import show_triscatter_with_acqf_from_optimizer as _show_triscatter_with_acqf_from_optimizer
from .utils import (
    candidate_result_from,
    ensure_2d,
    fixed_row_from,
    get_train_X,
    infer_feature_cols,
    to_numpy,
    to_tensor_like,
)


def _attribute_from(obj: Any, name: str) -> Any | None:
    for candidate in (
        obj,
        getattr(obj, "bundle", None),
        getattr(obj, "data_context", None),
    ):
        if candidate is not None and getattr(candidate, name, None) is not None:
            return getattr(candidate, name)
    return None


def _resolve_sum_value(
    obj: Any,
    select_indices: Sequence[int],
    sum_value: float | None,
) -> float:
    if sum_value is not None:
        return float(sum_value)

    constraint_idx = _attribute_from(obj, "constraint_idx")
    constraint_values = _attribute_from(obj, "constraint_values")
    if constraint_idx is None or constraint_values is None:
        raise ValueError(
            "sum_value を指定するか、constraint_idx / constraint_values を設定してください。"
        )

    constraints = list(constraint_idx)
    hits = [
        index
        for index, constraint in enumerate(constraints)
        if all(selected in constraint for selected in select_indices)
    ]
    if not hits:
        raise ValueError(
            "select_cols の3列を含む制約が constraint_idx に見つかりません。"
        )
    values = np.ravel(to_numpy(constraint_values)).astype(float)
    return float(values[hits[0]])


def _simplex_grid(sum_value: float, n: int) -> np.ndarray:
    axis = np.linspace(0.0, float(sum_value), int(n))
    first, second = np.meshgrid(axis, axis)
    valid = first.ravel() + second.ravel() <= float(sum_value) + 1e-12
    first = first.ravel()[valid]
    second = second.ravel()[valid]
    third = float(sum_value) - first - second
    third[third < 0.0] = 0.0
    return np.column_stack([first, second, third])


def _normalize_ternary(values: Any) -> np.ndarray:
    array = ensure_2d(values).astype(float)
    denominator = array.sum(axis=1, keepdims=True)
    denominator[np.isclose(denominator, 0.0)] = 1.0
    return array / denominator


def multiclass_tri_grid(
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
    """Evaluate multiclass probabilities on a three-component simplex."""

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
    entropy = entropy / np.log(probabilities.shape[-1])
    sorted_probabilities = np.sort(probabilities, axis=-1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    return {
        "grid": _normalize_ternary(simplex_values),
        "raw_grid": simplex_values,
        "probabilities": probabilities,
        "class_index": predicted_class,
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "class_labels": labels,
        "sum_value": total,
    }


def _class_colors(n_classes: int) -> list[str]:
    palette = list(qualitative.Plotly) + list(qualitative.Dark24)
    return [palette[index % len(palette)] for index in range(n_classes)]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _blend_with_white(color: str, strength: float) -> str:
    red, green, blue = _hex_to_rgb(color)
    strength = float(np.clip(strength, 0.0, 1.0))
    channels = [
        round(255 - (255 - channel) * strength)
        for channel in (red, green, blue)
    ]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


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


def _customdata(data: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(data["class_labels"], dtype=object)
    indices = np.asarray(data["class_index"], dtype=int)
    return np.column_stack(
        [
            labels[indices],
            np.asarray(data["confidence"], dtype=float),
            np.asarray(data["entropy"], dtype=float),
            np.asarray(data["margin"], dtype=float),
        ]
    )


def _continuous_metric_trace(
    grid: np.ndarray,
    values: np.ndarray,
    customdata: np.ndarray,
    *,
    mode: MulticlassHeatmapMode,
    marker_size: float,
) -> go.Scatterternary:
    if mode == "entropy":
        colorscale = "Viridis"
        colorbar_title = "normalized entropy"
    else:
        colorscale = "Blues"
        colorbar_title = "top-2 probability margin"
    return go.Scatterternary(
        a=grid[:, 0],
        b=grid[:, 1],
        c=grid[:, 2],
        mode="markers",
        name=colorbar_title,
        marker=dict(
            size=marker_size,
            color=values,
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            showscale=True,
            colorbar=dict(title=colorbar_title, lenmode="pixels", len=220),
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=(
            "predicted class: %{customdata[0]}<br>"
            "max probability: %{customdata[1]:.3f}<br>"
            "normalized entropy: %{customdata[2]:.3f}<br>"
            "top-2 margin: %{customdata[3]:.3f}<extra></extra>"
        ),
        showlegend=False,
    )


def show_multiclass_triscatter(
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    data_tri_plot: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    candidate_X: Any | None = None,
    mode: MulticlassHeatmapMode = "class_confidence",
    marker_size: float = 8.0,
    boundary_margin: float | None = 0.08,
) -> Figure:
    """Plot a multiclass probability map on a ternary simplex.

    For ``class_confidence``, hue indicates the winning class and lightness
    indicates its maximum predicted probability. Low top-2 margins can be
    overlaid as black rings to make decision boundaries visible.
    """

    valid_modes = {"class_confidence", "class", "entropy", "margin"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}.")

    grid = np.asarray(data_tri_plot["grid"], dtype=float)
    class_index = np.asarray(data_tri_plot["class_index"], dtype=int)
    confidence = np.asarray(data_tri_plot["confidence"], dtype=float)
    labels = list(data_tri_plot["class_labels"])
    colors = _class_colors(len(labels))
    customdata = _customdata(data_tri_plot)

    fig = go.Figure()
    if mode in {"entropy", "margin"}:
        metric = np.asarray(data_tri_plot[mode], dtype=float)
        fig.add_trace(
            _continuous_metric_trace(
                grid,
                metric,
                customdata,
                mode=mode,
                marker_size=marker_size,
            )
        )
    else:
        chance = 1.0 / len(labels)
        normalized_confidence = np.clip(
            (confidence - chance) / max(1.0 - chance, 1e-12),
            0.0,
            1.0,
        )
        for class_number, (label, color) in enumerate(
            zip(labels, colors, strict=False)
        ):
            mask = class_index == class_number
            if not mask.any():
                continue
            if mode == "class_confidence":
                marker_colors = [
                    _blend_with_white(color, 0.2 + 0.8 * strength)
                    for strength in normalized_confidence[mask]
                ]
            else:
                marker_colors = [color] * int(mask.sum())
            fig.add_trace(
                go.Scatterternary(
                    a=grid[mask, 0],
                    b=grid[mask, 1],
                    c=grid[mask, 2],
                    mode="markers",
                    name=f"predicted: {label}",
                    marker=dict(
                        size=marker_size,
                        color=marker_colors,
                        line=dict(width=0),
                    ),
                    customdata=customdata[mask],
                    hovertemplate=(
                        "predicted class: %{customdata[0]}<br>"
                        "max probability: %{customdata[1]:.3f}<br>"
                        "normalized entropy: %{customdata[2]:.3f}<br>"
                        "top-2 margin: %{customdata[3]:.3f}<extra></extra>"
                    ),
                )
            )

    if boundary_margin is not None:
        boundary_mask = np.asarray(data_tri_plot["margin"], dtype=float) <= float(
            boundary_margin
        )
        if boundary_mask.any():
            fig.add_trace(
                go.Scatterternary(
                    a=grid[boundary_mask, 0],
                    b=grid[boundary_mask, 1],
                    c=grid[boundary_mask, 2],
                    mode="markers",
                    name="low-margin boundary",
                    marker=dict(
                        size=marker_size + 1.5,
                        color="rgba(0,0,0,0)",
                        line=dict(width=1.0, color="black"),
                    ),
                    hoverinfo="skip",
                )
            )

    observed = _normalize_ternary(
        X[[feature_col1, feature_col2, feature_col3]].to_numpy()
    )
    observed_index = _class_indices(y[target_col], labels)
    for class_number, (label, color) in enumerate(zip(labels, colors, strict=False)):
        mask = observed_index == class_number
        if mask.any():
            fig.add_trace(
                go.Scatterternary(
                    a=observed[mask, 0],
                    b=observed[mask, 1],
                    c=observed[mask, 2],
                    mode="markers",
                    name=f"observed: {label}",
                    marker=dict(
                        size=9,
                        color=color,
                        symbol="diamond",
                        line=dict(width=0.9, color="black"),
                    ),
                    hovertemplate=f"observed class: {label}<extra></extra>",
                )
            )

    if candidate_X is not None:
        candidate_array = ensure_2d(candidate_X)
        columns = list(X.columns)
        selected = candidate_array[
            :,
            [
                columns.index(feature_col1),
                columns.index(feature_col2),
                columns.index(feature_col3),
            ],
        ]
        selected = _normalize_ternary(selected)
        fig.add_trace(
            go.Scatterternary(
                a=selected[:, 0],
                b=selected[:, 1],
                c=selected[:, 2],
                mode="markers",
                name="candidates",
                marker=dict(
                    size=12,
                    color="white",
                    symbol="diamond",
                    line=dict(width=1.5, color="black"),
                ),
            )
        )

    annotation = (
        "Hue = predicted class; darker = higher maximum probability"
        if mode == "class_confidence"
        else None
    )
    fig.update_layout(
        height=680,
        width=850,
        font_size=15,
        legend_title_text="class",
        ternary=dict(
            sum=1,
            aaxis=dict(title=feature_col1),
            baxis=dict(title=feature_col2),
            caxis=dict(title=feature_col3),
        ),
        annotations=(
            [
                dict(
                    text=annotation,
                    x=0.5,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ]
            if annotation is not None
            else []
        ),
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


def show_multiclass_triscatter_from_optimizer(
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
    mode: MulticlassHeatmapMode = "class_confidence",
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
    marker_size: float = 8.0,
    boundary_margin: float | None = 0.08,
) -> Figure:
    """Create a multiclass ternary decision map from an optimizer."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    output_index = _resolve_output_index(y_df, target_col, output_index)
    data = multiclass_tri_grid(
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
) -> Figure:
    """Dispatch multiclass ternary prediction maps or the existing plot."""

    if is_multiclass_object(obj) and show_type == "pred":
        return show_multiclass_triscatter_from_optimizer(
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
            mode=multiclass_mode,
            class_labels=class_labels,
            output_index=output_index,
            marker_size=marker_size,
            boundary_margin=boundary_margin,
        )
    return _show_triscatter_with_acqf_from_optimizer(
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
    )


__all__ = [
    "multiclass_tri_grid",
    "show_multiclass_triscatter",
    "show_multiclass_triscatter_from_optimizer",
    "show_triscatter_with_acqf_from_optimizer",
]
