"""Study result tables and Optuna-like Plotly visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from bochan.api.study_results import (
    _resolve_direction,
    _row_values,
    _trial_value,
)


def _target_names(
    target_cols: Sequence[str] | None,
    output_indices: Sequence[int],
) -> list[str]:
    """Resolve display names for selected outputs."""
    if target_cols is None:
        return [f"y{index}" for index in output_indices]
    names = [str(name) for name in target_cols]
    if not output_indices:
        return []
    if max(output_indices) >= len(names):
        raise ValueError(
            "target_cols does not contain every selected output index. "
            f"Got {len(names)} names for indices {list(output_indices)}."
        )
    return [names[index] for index in output_indices]


def study_history_dataframe(
    study: Any,
    *,
    output_index: int = 0,
    direction: Any | None = None,
    target_name: str | None = None,
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """Build finite observation and best-so-far history for one output."""
    resolved_direction = _resolve_direction(study, output_index, direction)
    value_col = str(target_name or f"y{output_index}")
    rows: list[dict[str, Any]] = []
    best_value: float | None = None

    completed = sorted(study.completed_trials(), key=lambda trial: int(trial.trial_id))
    for order, trial in enumerate(completed):
        value = _trial_value(trial, output_index)
        if value is None:
            continue
        improved = (
            best_value is None
            or (resolved_direction == "maximize" and value > best_value)
            or (resolved_direction == "minimize" and value < best_value)
        )
        if improved:
            best_value = value
        rows.append(
            {
                "trial_id": int(trial.trial_id),
                "order": int(order),
                cycle_col: trial.metadata.get(cycle_col, order),
                value_col: value,
                "best_value": best_value,
                "is_best": bool(improved),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["trial_id", "order", cycle_col, value_col, "best_value", "is_best"],
    )


def study_pareto_dataframe(
    study: Any,
    *,
    output_indices: Sequence[int] | None = None,
    directions: Sequence[Any] | None = None,
    target_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a completed-trial table with a Pareto-front flag."""
    completed = sorted(study.completed_trials(), key=lambda trial: int(trial.trial_id))
    if not completed:
        indices = list(output_indices or [])
        names = _target_names(target_cols, indices)
        return pd.DataFrame(columns=["trial_id", *names, "is_pareto"])

    output_count = max(len(_row_values(trial.y)) for trial in completed if trial.y is not None)
    indices = list(range(output_count)) if output_indices is None else [int(index) for index in output_indices]
    names = _target_names(target_cols, indices)
    pareto_ids = {
        int(trial.trial_id)
        for trial in study.pareto_trials(
            output_indices=indices,
            directions=directions,
        )
    }

    rows: list[dict[str, Any]] = []
    for trial in completed:
        values: list[float] = []
        valid = True
        for index in indices:
            value = _trial_value(trial, index)
            if value is None:
                valid = False
                break
            values.append(value)
        if not valid:
            continue
        row = {"trial_id": int(trial.trial_id)}
        row.update(dict(zip(names, values, strict=True)))
        row["is_pareto"] = int(trial.trial_id) in pareto_ids
        rows.append(row)
    return pd.DataFrame(rows, columns=["trial_id", *names, "is_pareto"])


def show_optimization_history_study(
    study: Any,
    *,
    output_index: int = 0,
    direction: Any | None = None,
    target_name: str | None = None,
    x_axis: str = "trial_id",
    cycle_col: str = "cycle",
):
    """Plot observed values and the cumulative best value for one output."""
    import plotly.graph_objects as go

    df = study_history_dataframe(
        study,
        output_index=output_index,
        direction=direction,
        target_name=target_name,
        cycle_col=cycle_col,
    )
    value_col = str(target_name or f"y{output_index}")
    if x_axis not in {"trial_id", "order", cycle_col}:
        raise ValueError(
            f"x_axis must be 'trial_id', 'order', or {cycle_col!r}. Got {x_axis!r}."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_axis],
            y=df[value_col],
            mode="markers",
            name="observed",
            customdata=df[["trial_id"]],
            hovertemplate=(
                f"{x_axis}=%{{x}}<br>{value_col}=%{{y}}"
                "<br>trial_id=%{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[x_axis],
            y=df["best_value"],
            mode="lines+markers",
            name="best so far",
        )
    )
    improved = df[df["is_best"]]
    if not improved.empty:
        fig.add_trace(
            go.Scatter(
                x=improved[x_axis],
                y=improved[value_col],
                mode="markers",
                name="new best",
                marker={"symbol": "diamond", "size": 10},
            )
        )
    fig.update_layout(
        height=450,
        width=760,
        xaxis_title=x_axis,
        yaxis_title=value_col,
        legend_title_text="series",
        font_size=14,
    )
    return fig


def show_pareto_front_study(
    study: Any,
    *,
    output_indices: Sequence[int] = (0, 1),
    directions: Sequence[Any] | None = None,
    target_cols: Sequence[str] | None = None,
):
    """Plot all finite completed trials and highlight a two-output Pareto front."""
    import plotly.graph_objects as go

    indices = [int(index) for index in output_indices]
    if len(indices) != 2:
        raise ValueError(
            "show_pareto_front_study currently requires exactly two output indices."
        )
    names = _target_names(target_cols, indices)
    df = study_pareto_dataframe(
        study,
        output_indices=indices,
        directions=directions,
        target_cols=target_cols,
    )
    dominated = df[~df["is_pareto"]]
    pareto = df[df["is_pareto"]].copy()

    fig = go.Figure()
    if not dominated.empty:
        fig.add_trace(
            go.Scatter(
                x=dominated[names[0]],
                y=dominated[names[1]],
                mode="markers",
                name="completed",
                customdata=dominated[["trial_id"]],
                hovertemplate=(
                    f"{names[0]}=%{{x}}<br>{names[1]}=%{{y}}"
                    "<br>trial_id=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    if not pareto.empty:
        sort_ascending = True
        if directions is not None and len(directions) > 0:
            sort_ascending = _resolve_direction(study, indices[0], directions[0]) == "maximize"
        pareto = pareto.sort_values(names[0], ascending=sort_ascending)
        fig.add_trace(
            go.Scatter(
                x=pareto[names[0]],
                y=pareto[names[1]],
                mode="lines+markers",
                name="Pareto front",
                marker={"symbol": "diamond", "size": 10},
                customdata=pareto[["trial_id"]],
                hovertemplate=(
                    f"{names[0]}=%{{x}}<br>{names[1]}=%{{y}}"
                    "<br>trial_id=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=560,
        width=700,
        xaxis_title=names[0],
        yaxis_title=names[1],
        legend_title_text="trials",
        font_size=14,
    )
    return fig


__all__ = [
    "show_optimization_history_study",
    "show_pareto_front_study",
    "study_history_dataframe",
    "study_pareto_dataframe",
]
