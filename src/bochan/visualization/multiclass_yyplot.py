"""Multiclass probability YY plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from .data import training_dataframe
from .multiclass import (
    _class_colors,
    _class_indices,
    infer_class_labels,
    is_multiclass_object,
    multiclass_probabilities,
)
from .plots import show_yyplot_from_optimizer as _show_yyplot_from_optimizer
from .utils import cycle_series


def show_multiclass_yyplot(
    y: pd.DataFrame,
    target: str,
    probabilities: np.ndarray,
    *,
    class_labels: Sequence[Any],
    cycle: Sequence[Any] | pd.Series | None = None,
) -> Figure:
    """Plot every predicted class probability against the observed class.

    The x-axis contains the ground-truth class. Each colored series represents
    the probability assigned to one class, so all class probabilities remain
    visible for every training sample.
    """

    if target not in y.columns:
        raise ValueError(f"y に列 {target!r} が存在しません。")

    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 2:
        raise ValueError(
            "probabilities must have shape [n_samples, n_classes]. "
            f"Got shape={probability_array.shape}."
        )

    labels = list(class_labels)
    if probability_array.shape != (len(y), len(labels)):
        raise ValueError(
            "probabilities shape must match y rows and class_labels. "
            f"Got shape={probability_array.shape}, rows={len(y)}, "
            f"classes={len(labels)}."
        )

    observed_indices = _class_indices(y[target], labels)
    unknown_mask = observed_indices < 0
    if unknown_mask.any():
        unknown_values = pd.unique(y.loc[unknown_mask, target]).tolist()
        raise ValueError(
            "正解ラベルを class_labels に対応付けられませんでした。"
            f"unknown={unknown_values}, class_labels={labels}"
        )

    predicted_indices = probability_array.argmax(axis=-1)
    predicted_labels = np.asarray(labels, dtype=object)[predicted_indices]
    observed_labels = np.asarray(labels, dtype=object)[observed_indices]
    colors = _class_colors(len(labels))
    offsets = (
        np.asarray([0.0])
        if len(labels) == 1
        else np.linspace(-0.3, 0.3, len(labels))
    )

    cycle_values = None
    if cycle is not None:
        cycle_values = pd.Series(cycle).reset_index(drop=True)
        if len(cycle_values) != len(y):
            raise ValueError(
                "cycle の長さが一致しません。"
                f"expected={len(y)}, got={len(cycle_values)}"
            )

    figure = go.Figure()
    sample_indices = np.arange(len(y), dtype=int)
    for class_index, (label, color, offset) in enumerate(
        zip(labels, colors, offsets, strict=False)
    ):
        valid = np.isfinite(probability_array[:, class_index])
        x_positions = observed_indices[valid].astype(float) + float(offset)
        if cycle_values is None:
            customdata = np.column_stack(
                [
                    sample_indices[valid],
                    observed_labels[valid],
                    predicted_labels[valid],
                ]
            )
            hovertemplate = (
                "sample: %{customdata[0]}<br>"
                "正解ラベル: %{customdata[1]}<br>"
                f"確率ラベル: {label}<br>"
                "確率: %{y:.3f}<br>"
                "予測ラベル: %{customdata[2]}<extra></extra>"
            )
        else:
            customdata = np.column_stack(
                [
                    sample_indices[valid],
                    observed_labels[valid],
                    predicted_labels[valid],
                    cycle_values.to_numpy()[valid],
                ]
            )
            hovertemplate = (
                "sample: %{customdata[0]}<br>"
                "正解ラベル: %{customdata[1]}<br>"
                f"確率ラベル: {label}<br>"
                "確率: %{y:.3f}<br>"
                "予測ラベル: %{customdata[2]}<br>"
                "cycle: %{customdata[3]}<extra></extra>"
            )

        figure.add_trace(
            go.Scatter(
                x=x_positions,
                y=probability_array[valid, class_index],
                mode="markers",
                name=f"P({target}={label})",
                marker=dict(
                    color=color,
                    size=9,
                    opacity=0.8,
                    line=dict(width=0.5, color="black"),
                ),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    figure.update_layout(
        height=600,
        width=850,
        xaxis_title="正解ラベル",
        yaxis_title="予測確率",
        legend_title_text="確率ラベル",
        font_size=16,
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=[str(label) for label in labels],
        range=[-0.55, len(labels) - 0.45],
    )
    figure.update_yaxes(range=[0.0, 1.0])
    return figure


def show_yyplot_from_optimizer(
    obj: Any,
    target: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    candidate_result: Any | None = None,
    cycle: str | Sequence[Any] | pd.Series | None = None,
    class_labels: Sequence[Any] | None = None,
    output_index: int | None = None,
) -> Figure:
    """Dispatch multiclass models to a ground-truth-versus-probability plot."""

    if not is_multiclass_object(obj):
        return _show_yyplot_from_optimizer(
            obj,
            target,
            feature_cols=feature_cols,
            target_cols=target_cols,
            candidate_result=candidate_result,
            cycle=cycle,
        )

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    if target not in y_df.columns:
        raise ValueError(f"target must be one of {list(y_df.columns)}.")

    resolved_output_index = (
        list(y_df.columns).index(target)
        if output_index is None
        else int(output_index)
    )
    probabilities = multiclass_probabilities(
        obj,
        X_df.to_numpy(),
        output_index=resolved_output_index,
    )
    labels = infer_class_labels(
        obj,
        probabilities.shape[-1],
        class_labels=class_labels,
        output_index=resolved_output_index,
        observed_labels=y_df[target],
    )
    resolved_cycle = cycle_series(
        cycle,
        X=X_df,
        y=y_df,
        length=len(y_df),
    )
    return show_multiclass_yyplot(
        y_df,
        target,
        probabilities,
        class_labels=labels,
        cycle=resolved_cycle,
    )


__all__ = ["show_multiclass_yyplot", "show_yyplot_from_optimizer"]
