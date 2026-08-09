"""Multiclass-specific YY-style probability visualization."""

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


def _aggregate_expanded_probability_rows(
    probabilities: np.ndarray,
    *,
    n_samples: int,
) -> np.ndarray:
    """Average one-to-many evaluation rows back to nominal samples.

    BoTorch input transforms such as ``InputPerturbation`` evaluate every
    nominal input at multiple consecutive perturbed locations. YY plots are
    defined against the nominal labels, so probability rows are averaged over
    that expansion before plotting.
    """

    if probabilities.shape[0] == n_samples:
        return probabilities
    if n_samples <= 0 or probabilities.shape[0] % n_samples != 0:
        return probabilities
    expansion = probabilities.shape[0] // n_samples
    return probabilities.reshape(n_samples, expansion, probabilities.shape[1]).mean(axis=1)


def show_multiclass_yyplot(
    true_labels: pd.Series,
    probabilities: np.ndarray,
    *,
    target: str,
    class_labels: Sequence[Any],
    cycle: str | Sequence[Any] | pd.Series | None = None,
) -> Figure:
    """Plot the probability assigned to each sample's correct class.

    The x-axis contains the observed class label. The y-axis contains
    ``P(y=true_label)`` for the corresponding sample. One trace is created per
    observed class so class-wise calibration and difficult labels are visible.
    """

    labels = list(class_labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError(
            "probabilities must have shape [n_samples, n_classes] matching "
            "class_labels."
        )
    probabilities = _aggregate_expanded_probability_rows(
        probabilities,
        n_samples=len(true_labels),
    )
    if probabilities.shape[0] != len(true_labels):
        raise ValueError(
            "The number of probability rows must match the number of labels."
        )

    true_indices = _class_indices(true_labels, labels)
    predicted_indices = probabilities.argmax(axis=-1)
    predicted_confidence = probabilities.max(axis=-1)
    row_indices = np.arange(len(true_labels))
    valid = (true_indices >= 0) & (true_indices < len(labels))
    correct_probability = np.full(len(true_labels), np.nan, dtype=float)
    correct_probability[valid] = probabilities[
        row_indices[valid],
        true_indices[valid],
    ]

    cycles = (
        cycle_series(cycle, y=true_labels.to_frame(name=target), length=len(true_labels))
        if cycle is not None
        else pd.Series([None] * len(true_labels), index=true_labels.index)
    )
    colors = _class_colors(len(labels))
    predicted_labels = np.asarray(labels, dtype=object)[predicted_indices]

    fig = go.Figure()
    for class_index, (label, color) in enumerate(zip(labels, colors, strict=False)):
        mask = valid & (true_indices == class_index)
        if not mask.any():
            continue
        customdata = np.column_stack(
            [
                predicted_labels[mask],
                predicted_confidence[mask],
                cycles.to_numpy(dtype=object)[mask],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=np.full(int(mask.sum()), str(label), dtype=object),
                y=correct_probability[mask],
                mode="markers",
                name=str(label),
                marker=dict(
                    color=color,
                    size=10,
                    line=dict(width=0.7, color="black"),
                ),
                customdata=customdata,
                hovertemplate=(
                    f"正解ラベル: {label}<br>"
                    "正解ラベル確率: %{y:.3f}<br>"
                    "予測ラベル: %{customdata[0]}<br>"
                    "最大確率: %{customdata[1]:.3f}<br>"
                    "cycle: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    chance = 1.0 / len(labels)
    fig.add_hline(
        y=chance,
        line=dict(color="gray", dash="dash"),
        annotation_text=f"chance = {chance:.3f}",
        annotation_position="top left",
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[str(label) for label in labels],
        title_text="正解ラベル",
    )
    fig.update_yaxes(range=[0.0, 1.0], title_text="正解ラベルに対する予測確率")
    fig.update_layout(
        height=600,
        width=800,
        legend_title_text="正解ラベル",
        font_size=16,
    )
    return fig


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
    """Dispatch multiclass models to a correct-label probability plot."""

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
    return show_multiclass_yyplot(
        y_df[target],
        probabilities,
        target=target,
        class_labels=labels,
        cycle=cycle,
    )


__all__ = ["show_multiclass_yyplot", "show_yyplot_from_optimizer"]