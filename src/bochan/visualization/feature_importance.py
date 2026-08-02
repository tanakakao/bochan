"""DataFrame and Plotly views for core feature-importance results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bochan.inspection.result_types import (
    CrossValidatedFeatureImportanceResult,
    FeatureImportanceResult,
)


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for feature-importance tables.") from exc
    return pd


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for feature-importance figures.") from exc
    return go


def _select_output(result: Any, output_name: str | None) -> tuple[str, Any]:
    outputs = result.outputs
    if output_name is None:
        if len(outputs) != 1:
            raise ValueError("output_name is required for a multi-output feature-importance result.")
        output_name = next(iter(outputs))
    if output_name not in outputs:
        raise ValueError(f"Unknown output_name {output_name!r}; available outputs: {list(outputs)!r}.")
    return output_name, outputs[output_name]


def _select_method(output: Any, kind: str, method: str, class_label: Any) -> Any:
    if kind == "predictive":
        methods = output.predictive_methods
    elif kind == "noise":
        methods = output.noise_methods
    elif kind == "classwise":
        classwise = output.classwise_methods or {}
        if class_label not in classwise and str(class_label) in classwise:
            class_label = str(class_label)
        if class_label not in classwise:
            raise ValueError(f"Unknown class_label {class_label!r}; available labels: {list(classwise)!r}.")
        methods = classwise[class_label]
    else:
        raise ValueError("importance_kind must be predictive, noise, or classwise.")
    if method not in methods:
        raise ValueError(f"Method {method!r} is unavailable for {kind} importance.")
    return methods[method]


def _finish_frame(frame: Any, *, normalized: bool, sort: bool, rank_by: str, top_k: int | None, include_negative: bool):
    if rank_by not in {"value", "absolute"}:
        raise ValueError("rank_by must be 'value' or 'absolute'.")
    value_column = "normalized_mean" if normalized else "mean"
    frame["display_value"] = frame[value_column]
    if not include_negative:
        frame = frame[frame["display_value"] >= 0]
    if sort:
        key = frame["display_value"].abs() if rank_by == "absolute" else frame["display_value"]
        frame = frame.assign(_sort_key=key).sort_values("_sort_key", ascending=False).drop(columns="_sort_key")
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive when provided.")
        frame = frame.head(top_k)
    return frame.reset_index(drop=True)


def feature_importance_dataframe(
    result: FeatureImportanceResult,
    *,
    output_name: str | None = None,
    method: str = "permutation",
    importance_kind: str = "predictive",
    class_label: Any | None = None,
    normalized: bool = False,
    sort: bool = True,
    rank_by: str = "value",
    top_k: int | None = None,
    include_negative: bool = True,
):
    """Convert one output and importance method to a long DataFrame.

    Args:
        result: Core feature-importance result.
        output_name: Output to select; optional only for a single output.
        method: Predictive method name.
        importance_kind: ``predictive``, ``noise``, or ``classwise``.
        class_label: Class label for classwise importance.
        normalized: Select normalized values for display and filtering.
        sort: Whether to sort rows by their display value.
        rank_by: Sort by signed ``value`` or ``absolute`` magnitude.
        top_k: Optional number of displayed rows.
        include_negative: Whether negative display values are retained.

    Returns:
        A pandas DataFrame retaining raw statistics and provenance.
    """
    pd = _pandas()
    output_name, output = _select_output(result, output_name)
    selected = _select_method(output, importance_kind, method, class_label)
    rows = []
    for entry in selected.entries.values():
        summary = entry.importance
        rows.append(
            {
                "output_name": output_name,
                "task_type": output.task_type,
                "importance_kind": importance_kind,
                "method": method,
                "class_label": class_label,
                "feature": entry.name,
                "rank": summary.rank,
                "indices": list(entry.indices),
                "feature_type": entry.feature_type,
                "role": entry.role,
                "mean": summary.mean,
                "std": summary.std,
                "minimum": summary.minimum,
                "maximum": summary.maximum,
                "median": summary.median,
                "normalized_mean": summary.normalized_mean,
                "normalized_std": entry.metadata.get("normalized_std"),
                "baseline_metric": entry.baseline_metric,
                "metric_name": entry.metric_name,
                "metric_direction": entry.metric_direction,
                "n_repeats": result.n_repeats,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The selected feature-importance result contains no entries.")
    if normalized and frame["normalized_mean"].isna().all():
        raise ValueError("Normalized importance is unavailable for the selected result.")
    return _finish_frame(
        frame, normalized=normalized, sort=sort, rank_by=rank_by, top_k=top_k, include_negative=include_negative
    )


def cross_validated_feature_importance_dataframe(
    result: CrossValidatedFeatureImportanceResult,
    *,
    output_name: str | None = None,
    method: str = "permutation",
    importance_kind: str = "predictive",
    class_label: Any | None = None,
    normalized: bool = False,
    sort: bool = True,
    rank_by: str = "value",
    top_k: int | None = None,
    include_negative: bool = True,
):
    """Convert cross-validated importance to a long DataFrame.

    Args:
        result: Aggregated validation-fold result.
        output_name: Output to select.
        method: Predictive method name.
        importance_kind: ``predictive`` or ``noise``.
        class_label: Reserved for API symmetry; CV classwise data is unsupported.
        normalized: Select normalized fold means when supplied by the core.
        sort: Whether to sort rows.
        rank_by: Signed-value or absolute-value sorting.
        top_k: Optional displayed row limit.
        include_negative: Whether negative values are retained.

    Returns:
        A DataFrame separating repeat and between-fold variability.
    """
    del class_label
    pd = _pandas()
    output_name, output = _select_output(result, output_name)
    if importance_kind not in {"predictive", "noise"}:
        raise ValueError("Cross-validated importance_kind must be predictive or noise.")
    methods = output.predictive_methods if importance_kind == "predictive" else output.noise_methods
    if method not in methods:
        raise ValueError(f"Method {method!r} is unavailable for {importance_kind} importance.")
    selected = methods[method]
    rows = []
    for name, summary in selected.entries.items():
        repeat = summary.within_fold_repeat_std
        rows.append(
            {
                "output_name": output_name,
                "task_type": output.task_type,
                "importance_kind": importance_kind,
                "method": method,
                "feature": name,
                "rank": summary.mean_rank,
                "indices": [],
                "feature_type": None,
                "role": None,
                "mean": summary.mean,
                "fold_mean": summary.mean,
                "std": (sum(repeat) / len(repeat)) if repeat else None,
                "repeat_std": repeat,
                "between_fold_std": summary.std,
                "minimum": summary.minimum,
                "maximum": summary.maximum,
                "median": summary.median,
                "minimum_fold": summary.minimum,
                "maximum_fold": summary.maximum,
                "median_fold": summary.median,
                "normalized_mean": None,
                "mean_rank": summary.mean_rank,
                "rank_std": summary.rank_std,
                "valid_fold_count": summary.valid_fold_count,
                "baseline_metric": None,
                "metric_name": None,
                "metric_direction": None,
                "n_repeats": None,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The selected cross-validated result contains no entries.")
    if normalized:
        raise ValueError("Normalized cross-validated importance is unavailable.")
    return _finish_frame(
        frame, normalized=normalized, sort=sort, rank_by=rank_by, top_k=top_k, include_negative=include_negative
    )


def _bar_figure(frame: Any, *, cv: bool, normalized: bool, show_error_bars: bool, orientation: str, title: str):
    go = _plotly()
    values = frame["display_value"].tolist()
    labels = frame["feature"].astype(str).tolist()
    errors = None
    if show_error_bars and not normalized:
        errors = frame["between_fold_std" if cv else "std"].tolist()
    custom_cols = ["rank", "metric_name", "baseline_metric", "feature_type", "role", "indices"]
    customdata = (
        frame.reindex(columns=custom_cols).astype(object).where(frame.reindex(columns=custom_cols).notna(), None).values
    )
    horizontal = orientation == "horizontal"
    trace = go.Bar(
        x=values if horizontal else labels,
        y=labels if horizontal else values,
        orientation="h" if horizontal else "v",
        error_x=dict(type="data", array=errors, visible=True) if horizontal and errors is not None else None,
        error_y=dict(type="data", array=errors, visible=True) if not horizontal and errors is not None else None,
        customdata=customdata,
        hovertemplate="feature=%{y}<br>importance=%{x}<br>rank=%{customdata[0]}<br>metric=%{customdata[1]}<br>baseline=%{customdata[2]}<br>type=%{customdata[3]}<br>role=%{customdata[4]}<br>indices=%{customdata[5]}<extra></extra>"
        if horizontal
        else None,
    )
    fig = go.Figure(trace)
    fig.add_vline(x=0, line_width=1, line_color="gray") if horizontal else fig.add_hline(
        y=0, line_width=1, line_color="gray"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Normalized importance" if normalized else "Metric degradation",
        yaxis_title="Feature" if horizontal else "Metric degradation",
    )
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    return fig


def show_feature_importance(
    result: FeatureImportanceResult,
    *,
    output_name: str | None = None,
    method: str = "permutation",
    importance_kind: str = "predictive",
    class_label: Any | None = None,
    normalized: bool = False,
    top_k: int | None = 15,
    rank_by: str = "value",
    include_negative: bool = True,
    show_error_bars: bool = True,
    orientation: str = "horizontal",
    title: str | None = None,
):
    """Build a permutation-importance bar figure from an existing result.

    Args:
        result: Core result; no importance is recomputed.
        output_name: Output to display.
        method: Importance method.
        importance_kind: Predictive, noise, or classwise importance.
        class_label: Class label for classwise importance.
        normalized: Display normalized importance.
        top_k: Maximum bars.
        rank_by: Signed or absolute sorting.
        include_negative: Retain negative bars.
        show_error_bars: Show repeat standard deviation for raw importance.
        orientation: ``horizontal`` or ``vertical``.
        title: Optional figure title.

    Returns:
        A Plotly Figure.
    """
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be horizontal or vertical.")
    frame = feature_importance_dataframe(
        result,
        output_name=output_name,
        method=method,
        importance_kind=importance_kind,
        class_label=class_label,
        normalized=normalized,
        rank_by=rank_by,
        top_k=top_k,
        include_negative=include_negative,
    )
    name = str(frame.iloc[0]["output_name"])
    prefix = (
        "Noise " if importance_kind == "noise" else (f"Class {class_label} " if importance_kind == "classwise" else "")
    )
    return _bar_figure(
        frame,
        cv=False,
        normalized=normalized,
        show_error_bars=show_error_bars,
        orientation=orientation,
        title=title or f"{name}: {prefix}{method.title()} importance",
    )


def show_cross_validated_feature_importance(
    result: CrossValidatedFeatureImportanceResult,
    *,
    output_name: str | None = None,
    method: str = "permutation",
    importance_kind: str = "predictive",
    class_label: Any | None = None,
    normalized: bool = False,
    top_k: int | None = 15,
    rank_by: str = "value",
    include_negative: bool = True,
    show_error_bars: bool = True,
    title: str | None = None,
):
    """Build a CV bar figure whose error bars are between-fold standard deviations.

    Args:
        result: Cross-validated result.
        output_name: Output to display.
        method: Importance method.
        importance_kind: Predictive or noise importance.
        class_label: Reserved for API symmetry.
        normalized: Display normalized values if available.
        top_k: Maximum bars.
        rank_by: Signed or absolute sorting.
        include_negative: Retain negative bars.
        show_error_bars: Show between-fold standard deviations.
        title: Optional title.

    Returns:
        A Plotly Figure.
    """
    frame = cross_validated_feature_importance_dataframe(
        result,
        output_name=output_name,
        method=method,
        importance_kind=importance_kind,
        class_label=class_label,
        normalized=normalized,
        rank_by=rank_by,
        top_k=top_k,
        include_negative=include_negative,
    )
    name = str(frame.iloc[0]["output_name"])
    return _bar_figure(
        frame,
        cv=True,
        normalized=normalized,
        show_error_bars=show_error_bars,
        orientation="horizontal",
        title=title or f"{name}: Cross-validated {method.title()} importance",
    )


def build_feature_importance_figures(
    result: FeatureImportanceResult | CrossValidatedFeatureImportanceResult,
    *,
    method: str = "permutation",
    output_names: Sequence[str] | None = None,
    include_predictive: bool = True,
    include_noise: bool = True,
    include_classwise: bool = False,
    normalized: bool = False,
    top_k: int | None = 15,
    rank_by: str = "value",
) -> dict[str, Any]:
    """Build all available output/kind figures without recomputing importance.

    Args:
        result: Core or cross-validated importance result.
        method: Method to visualize.
        output_names: Optional output subset.
        include_predictive: Include predictive figures.
        include_noise: Include noise figures.
        include_classwise: Include classwise figures.
        normalized: Display normalized means.
        top_k: Maximum bars per figure.
        rank_by: Signed or absolute sorting.

    Returns:
        Mapping of stable figure keys to Plotly Figures.
    """
    is_cv = isinstance(result, CrossValidatedFeatureImportanceResult)
    figures = {}
    for name in output_names or result.outputs.keys():
        output = result.outputs[name]
        for kind, enabled, methods in (
            ("predictive", include_predictive, output.predictive_methods),
            ("noise", include_noise, output.noise_methods),
        ):
            if enabled and method in methods:
                show = show_cross_validated_feature_importance if is_cv else show_feature_importance
                figures[f"{name}-{kind}-{method}"] = show(
                    result,
                    output_name=name,
                    method=method,
                    importance_kind=kind,
                    normalized=normalized,
                    top_k=top_k,
                    rank_by=rank_by,
                )
        if include_classwise and not is_cv:
            for label, methods in (output.classwise_methods or {}).items():
                if method in methods:
                    figures[f"{name}-class-{label}-{method}"] = show_feature_importance(
                        result,
                        output_name=name,
                        method=method,
                        importance_kind="classwise",
                        class_label=label,
                        normalized=normalized,
                        top_k=top_k,
                        rank_by=rank_by,
                    )
    return figures


def ard_diagnostics_dataframe(result: FeatureImportanceResult, *, output_name: str | None = None):
    """Return flattened ARD kernel diagnostics.

    Args:
        result: Core feature-importance result.
        output_name: Output to select.

    Returns:
        A DataFrame of kernel sensitivity values.
    """
    pd = _pandas()
    _, output = _select_output(result, output_name)
    ard = output.model_diagnostics.get("ard", {})
    components = ard.get("components", ard if isinstance(ard, list) else [])
    rows = []
    for component in components:
        lengths = component.get("lengthscale", [])
        inverse = component.get("inverse_lengthscale", [])
        active = component.get("active_dims") or list(range(len(lengths)))
        for i, value in enumerate(lengths):
            index = active[i] if i < len(active) else i
            rows.append(
                {
                    "kernel_component": component.get("name", component.get("type")),
                    "feature": result.feature_names[index]
                    if isinstance(index, int) and index < len(result.feature_names)
                    else str(index),
                    "lengthscale": value,
                    "inverse_lengthscale": inverse[i] if i < len(inverse) else None,
                    "source_space": component.get("source_space", "model"),
                    "active_dims": active,
                }
            )
    return pd.DataFrame(rows)


def show_ard_diagnostics(
    result: FeatureImportanceResult, *, output_name: str | None = None, component: str | None = None
):
    """Plot inverse lengthscales separately from predictive importance.

    Args:
        result: Core result.
        output_name: Output to select.
        component: Optional kernel component filter.

    Returns:
        A Plotly Figure.
    """
    frame = ard_diagnostics_dataframe(result, output_name=output_name)
    if component is not None:
        frame = frame[frame["kernel_component"] == component]
    if frame.empty:
        raise ValueError("ARD diagnostics are unavailable.")
    go = _plotly()
    fig = go.Figure(go.Bar(x=frame["inverse_lengthscale"], y=frame["feature"], orientation="h"))
    fig.update_layout(
        title="ARD sensitivity diagnostic",
        xaxis_title="Inverse lengthscale",
        meta={"description": "ARD is a kernel sensitivity diagnostic, not permutation importance."},
    )
    return fig


def show_pca_explained_variance(result: FeatureImportanceResult, *, output_name: str | None = None):
    """Plot PCA explained variance as a representation diagnostic.

    Args:
        result: Core result.
        output_name: Output to select.

    Returns:
        A Plotly Figure.
    """
    _, output = _select_output(result, output_name)
    pca = output.model_diagnostics.get("pca", {})
    ratios = pca.get("explained_variance_ratio", pca.get("explained_variance_ratio_", []))
    if not ratios:
        raise ValueError("PCA explained variance diagnostics are unavailable.")
    go = _plotly()
    cumulative, total = [], 0.0
    for value in ratios:
        total += float(value)
        cumulative.append(total)
    x = list(range(1, len(ratios) + 1))
    fig = go.Figure(
        [go.Bar(x=x, y=ratios, name="Explained variance ratio"), go.Scatter(x=x, y=cumulative, name="Cumulative")]
    )
    fig.update_layout(title="PCA explained variance diagnostic", xaxis_title="Component")
    return fig


def show_task_correlation_diagnostics(result: FeatureImportanceResult, *, output_name: str | None = None):
    """Plot learned task-kernel correlation diagnostics.

    Args:
        result: Core result.
        output_name: Output to select.

    Returns:
        A Plotly heatmap.
    """
    _, output = _select_output(result, output_name)
    info = output.model_diagnostics.get("multitask", {})
    matrix = info.get("task_correlation", info.get("correlation"))
    if matrix is None:
        raise ValueError("Learned task correlation diagnostics are unavailable.")
    labels = info.get("task_names")
    go = _plotly()
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels))
    fig.update_layout(
        title="Learned task-kernel correlation diagnostic",
        meta={"description": "This is learned task-kernel correlation, not raw target correlation."},
    )
    return fig
