"""Heteroscedastic uncertainty decomposition for 1D prediction plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .categorical_axis import apply_categorical_xaxis_labels
from .data import candidates_dataframe, training_dataframe
from .input_perturbation import (
    aggregate_input_perturbation_moments,
    input_perturbation_n_w,
)
from .ordinal_display import show_1dplot_from_optimizer as _default_show_1dplot
from .utils import (
    axis_values,
    cycle_color_map,
    cycle_series,
    decode_values,
    ensure_2d,
    fixed_row_from,
    get_bounds,
    get_model,
    get_train_X,
    infer_feature_cols,
    infer_target_cols,
    labels_from,
    to_numpy,
    to_tensor_like,
)


def _target_model(model: Any, target: str, target_index: int) -> Any:
    """Return the target-specific submodel when a wrapper exposes one."""

    specs = list(getattr(model, "specs", []) or [])
    if specs:
        for spec in specs:
            if str(getattr(spec, "name", "")) == target:
                return getattr(spec, "model", model)
        if target_index < len(specs):
            return getattr(specs[target_index], "model", model)

    models = list(getattr(model, "models", []) or [])
    if target_index < len(models):
        return models[target_index]
    return model


def _is_heteroscedastic_model(model: Any) -> bool:
    """Return whether a model exposes an input-dependent observation-noise GP."""

    if callable(getattr(model, "predict_noise_var", None)):
        return True
    module_name = type(model).__module__.lower()
    class_name = type(model).__name__.lower()
    return hasattr(model, "noise_model") and (
        "hetero" in module_name or "hetero" in class_name
    )


def _target_is_regression(obj: Any, target: str, target_index: int) -> bool:
    """Resolve the selected target task without changing discrete plot behavior."""

    model = get_model(obj)
    specs = list(getattr(model, "specs", []) or [])
    if specs:
        selected = None
        for spec in specs:
            if str(getattr(spec, "name", "")) == target:
                selected = spec
                break
        if selected is None and target_index < len(specs):
            selected = specs[target_index]
        if selected is not None:
            return str(getattr(selected, "task_type", "")).lower() == "regression"

    bundle = getattr(obj, "bundle", None)
    metadata = getattr(bundle, "metadata", None)
    if isinstance(metadata, dict):
        sub_bundles = list(metadata.get("sub_bundles", []) or [])
        if target_index < len(sub_bundles):
            task_type = str(getattr(sub_bundles[target_index], "task_type", "")).lower()
            if task_type:
                return task_type == "regression"

    for candidate in (bundle, getattr(obj, "model_config", None)):
        task_type = str(getattr(candidate, "task_type", "")).lower()
        if task_type:
            return task_type in {"regression", "multi_objective"}

    target_model = _target_model(model, target, target_index)
    return ".regression." in type(target_model).__module__.lower()


def _raw_posterior_moments(
    obj: Any,
    X: Any,
    *,
    observation_noise: bool,
) -> tuple[Any, Any]:
    """Evaluate mean and variance with explicit observation-noise semantics."""

    predictor = getattr(obj, "predict", None)
    if callable(predictor):
        try:
            return predictor(
                X,
                return_type="mean_variance",
                posterior_kwargs={"observation_noise": observation_noise},
            )
        except (TypeError, ValueError, NotImplementedError):
            if not observation_noise:
                try:
                    return predictor(X, return_type="mean_variance")
                except TypeError:
                    pass

    posterior = get_model(obj).posterior(
        X,
        observation_noise=observation_noise,
    )
    return posterior.mean, posterior.variance


def _target_prediction_moments(
    obj: Any,
    X: Any,
    *,
    target: str,
    target_index: int,
    observation_noise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return perturbation-aggregated mean and standard deviation for one target."""

    n_points = int(ensure_2d(to_numpy(X)).shape[0])
    try:
        mean, variance = _raw_posterior_moments(
            obj,
            X,
            observation_noise=observation_noise,
        )
        index = target_index
    except (TypeError, ValueError, RuntimeError, NotImplementedError):
        model = _target_model(get_model(obj), target, target_index)
        posterior = model.posterior(X, observation_noise=observation_noise)
        mean, variance = posterior.mean, posterior.variance
        index = 0

    mean_arr = ensure_2d(mean)
    std_arr = np.sqrt(np.clip(ensure_2d(variance), 0.0, None))
    mean_arr, std_arr = aggregate_input_perturbation_moments(
        mean_arr,
        std_arr,
        n_points=n_points,
        n_w=input_perturbation_n_w(obj),
    )
    if index >= mean_arr.shape[1]:
        if mean_arr.shape[1] != 1:
            raise ValueError(
                f"Target {target!r} index {index} is outside prediction shape "
                f"{mean_arr.shape}."
            )
        index = 0
    return mean_arr[:, index], std_arr[:, index]


def _grid_points(
    obj: Any,
    feature: str,
    *,
    feature_cols: Sequence[str] | None,
    value_dict: dict[str, Any] | None,
    n: int,
) -> tuple[Any, np.ndarray, list[str]]:
    """Build the raw model inputs and display-axis values for a 1D slice."""

    train_X = get_train_X(obj)
    X_arr = ensure_2d(train_X)
    columns = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    if feature not in columns:
        raise ValueError(f"feature must be one of {columns!r}.")
    index = columns.index(feature)
    x_values = axis_values(
        obj,
        col=feature,
        col_index=index,
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=get_bounds(obj, train_X),
    )
    row = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = np.repeat(row, repeats=len(x_values), axis=0)
    grid[:, index] = x_values
    display_x = np.asarray(x_values)
    mapping = labels_from(obj, feature)
    if mapping is not None:
        display_x = np.asarray(
            decode_values(display_x.tolist(), mapping),
            dtype=object,
        )
    return to_tensor_like(grid, obj), display_x, columns


def _add_band(
    figure: go.Figure,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    name: str,
    fillcolor: str,
    linecolor: str,
    legendgroup: str,
) -> None:
    """Add a two-trace Plotly uncertainty band."""

    figure.add_trace(
        go.Scatter(
            x=x,
            y=mean + std,
            mode="lines",
            line=dict(width=0, color=linecolor),
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=mean - std,
            mode="lines",
            line=dict(width=0, color=linecolor),
            fill="tonexty",
            fillcolor=fillcolor,
            name=name,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )


def _heteroscedastic_figure(
    obj: Any,
    feature: str,
    target: str,
    *,
    feature_cols: Sequence[str] | None,
    target_cols: Sequence[str] | None,
    value_dict: dict[str, Any] | None,
    candidate_result: Any | None,
    n: int,
    cycle: str | Sequence[Any] | pd.Series | None,
) -> go.Figure:
    """Plot epistemic, aleatoric, and total heteroscedastic uncertainty."""

    X_df, y_df = training_dataframe(
        obj,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    targets = infer_target_cols(obj, target_cols, y_df.shape[1])
    if target not in targets:
        raise ValueError(f"target must be one of {targets!r}.")
    target_index = targets.index(target)
    grid, x_grid, resolved_feature_cols = _grid_points(
        obj,
        feature,
        feature_cols=feature_cols,
        value_dict=value_dict,
        n=n,
    )

    mean, epistemic_std = _target_prediction_moments(
        obj,
        grid,
        target=target,
        target_index=target_index,
        observation_noise=False,
    )
    total_mean, total_std = _target_prediction_moments(
        obj,
        grid,
        target=target,
        target_index=target_index,
        observation_noise=True,
    )
    if total_mean.shape == mean.shape:
        mean = total_mean
    aleatoric_std = np.sqrt(
        np.clip(np.square(total_std) - np.square(epistemic_std), 0.0, None)
    )

    try:
        order = np.argsort(np.asarray(x_grid, dtype=float))
        x_grid = x_grid[order]
        mean = mean[order]
        epistemic_std = epistemic_std[order]
        aleatoric_std = aleatoric_std[order]
        total_std = total_std[order]
    except (TypeError, ValueError):
        pass

    figure = go.Figure()
    _add_band(
        figure,
        x_grid,
        mean,
        total_std,
        name="総予測誤差 ±1σ（モデル + 観測ノイズ）",
        fillcolor="rgba(120,120,120,0.16)",
        linecolor="rgba(120,120,120,0.35)",
        legendgroup="total-uncertainty",
    )
    figure.add_trace(
        go.Scatter(
            x=x_grid,
            y=mean + aleatoric_std,
            mode="lines",
            name="観測ノイズ ±1σ",
            line=dict(color="darkorange", width=1.5, dash="dash"),
            legendgroup="aleatoric-uncertainty",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_grid,
            y=mean - aleatoric_std,
            mode="lines",
            line=dict(color="darkorange", width=1.5, dash="dash"),
            legendgroup="aleatoric-uncertainty",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    _add_band(
        figure,
        x_grid,
        mean,
        epistemic_std,
        name="モデル不確実性 ±1σ",
        fillcolor="rgba(31,119,180,0.22)",
        linecolor="rgba(31,119,180,0.45)",
        legendgroup="epistemic-uncertainty",
    )
    uncertainty_hover = np.column_stack(
        [epistemic_std, aleatoric_std, total_std]
    )
    figure.add_trace(
        go.Scatter(
            x=x_grid,
            y=mean,
            mode="lines",
            name=f"{target} 予測平均",
            line=dict(color="red", width=2),
            customdata=uncertainty_hover,
            hovertemplate=(
                f"{feature}: %{{x}}<br>"
                f"{target} 予測平均: %{{y}}<br>"
                "モデル不確実性 σ: %{customdata[0]}<br>"
                "観測ノイズ σ: %{customdata[1]}<br>"
                "総予測誤差 σ: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    cyc = cycle_series(cycle, X=X_df, y=y_df, length=len(X_df)) if cycle is not None else None
    color_map = cycle_color_map(cyc)
    if cyc is None:
        figure.add_trace(
            go.Scatter(
                x=X_df[feature],
                y=y_df[target],
                mode="markers",
                name="入力データ",
                marker=dict(color="blue", size=9),
            )
        )
    else:
        for cycle_value, color in color_map.items():
            mask = cyc == cycle_value
            figure.add_trace(
                go.Scatter(
                    x=X_df.loc[mask, feature],
                    y=y_df.loc[mask, target],
                    mode="markers",
                    name=f"cycle {cycle_value}",
                    marker=dict(
                        color=color,
                        size=9,
                        line=dict(width=0.5, color="black"),
                    ),
                )
            )

    candidate_df = candidates_dataframe(
        obj,
        candidate_result=candidate_result,
        feature_cols=resolved_feature_cols,
        target_cols=targets,
        include_prediction=False,
    )
    if candidate_df is not None and feature in candidate_df:
        result = candidate_result
        if result is None:
            history = getattr(obj, "history", None)
            result = history[-1] if history else None
        candidate_X = getattr(result, "candidates", None) if result is not None else None
        if candidate_X is not None:
            candidate_mean, candidate_epistemic = _target_prediction_moments(
                obj,
                candidate_X,
                target=target,
                target_index=target_index,
                observation_noise=False,
            )
            _, candidate_total = _target_prediction_moments(
                obj,
                candidate_X,
                target=target,
                target_index=target_index,
                observation_noise=True,
            )
            candidate_aleatoric = np.sqrt(
                np.clip(
                    np.square(candidate_total) - np.square(candidate_epistemic),
                    0.0,
                    None,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=candidate_df[feature],
                    y=candidate_mean,
                    mode="markers",
                    name="候補点",
                    marker=dict(color="green", size=10),
                    error_y=dict(type="data", array=candidate_total, visible=True),
                    customdata=np.column_stack(
                        [candidate_epistemic, candidate_aleatoric, candidate_total]
                    ),
                    hovertemplate=(
                        f"{feature}: %{{x}}<br>"
                        f"{target} 予測平均: %{{y}}<br>"
                        "モデル不確実性 σ: %{customdata[0]}<br>"
                        "観測ノイズ σ: %{customdata[1]}<br>"
                        "総予測誤差 σ: %{customdata[2]}<extra>候補点</extra>"
                    ),
                )
            )

    figure.update_layout(
        height=600,
        width=800,
        xaxis_title=feature,
        yaxis_title=target,
        legend_title_text="系列 / 誤差成分",
        font_size=16,
    )
    return apply_categorical_xaxis_labels(figure, obj, feature)


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
    **kwargs: Any,
) -> Any:
    """Show all uncertainty components for heteroscedastic regression models."""

    targets = infer_target_cols(
        obj,
        target_cols,
        ensure_2d(getattr(obj, "train_Y", np.empty((0, 1)))).shape[1],
    )
    target_index = targets.index(target) if target in targets else 0
    target_model = _target_model(get_model(obj), target, target_index)
    if not (
        _is_heteroscedastic_model(target_model)
        and _target_is_regression(obj, target, target_index)
    ):
        return _default_show_1dplot(
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

    return _heteroscedastic_figure(
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


__all__ = ["show_1dplot_from_optimizer"]
