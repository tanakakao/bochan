"""Web response and Plotly presentation for composition feature importance."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


def _setting(value: Any, name: str, default: Any) -> Any:
    """Read one presentation setting from a Pydantic object or mapping."""

    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_float(value: Any) -> float | None:
    """Return a finite float or ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return number if math.isfinite(number) else None


def _replace_predictive_summary(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace transformed composition-coordinate PI by one joint group row."""

    coordinate_features = {
        str(value) for value in payload.get("coordinate_features") or ()
    }
    summary = [
        dict(row) for row in list(result.get("feature_importance_summary") or ())
    ]
    filtered = [
        row
        for row in summary
        if not (
            str(row.get("importance_kind")) == "predictive"
            and str(row.get("feature")) in coordinate_features
        )
    ]
    filtered.extend(dict(row) for row in payload.get("overall") or ())

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        key = (
            str(row.get("output_name") or ""),
            str(row.get("importance_kind") or ""),
            str(row.get("method") or ""),
        )
        groups[key].append(row)

    for (_output, kind, _method), rows in groups.items():
        if kind != "predictive":
            continue
        ranked = sorted(
            rows,
            key=lambda row: _safe_float(row.get("mean"))
            if _safe_float(row.get("mean")) is not None
            else float("-inf"),
            reverse=True,
        )
        positive_total = sum(
            max(_safe_float(row.get("mean")) or 0.0, 0.0) for row in ranked
        )
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
            row["normalized_mean"] = (
                max(_safe_float(row.get("mean")) or 0.0, 0.0) / positive_total
                if positive_total > 0.0
                else 0.0
            )

    result["feature_importance_summary"] = filtered
    return filtered


def _matches_target(figure: dict[str, Any], target: str) -> bool:
    """Return whether a feature-importance figure belongs to one target."""

    from .target_results import _safe_figure_id

    identifier = str(figure.get("id") or "").lower()
    title = str(figure.get("title") or "")
    safe_target = _safe_figure_id(target).lower()
    return (
        safe_target in identifier
        or target.lower() in identifier
        or target in title
    )


def _composition_predictive_figures(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    visualization_settings: Any,
) -> list[dict[str, Any]]:
    """Build normal-variable plus composition-total PI figures."""

    if not bool(_setting(visualization_settings, "include_predictive", True)):
        return []

    import plotly.graph_objects as go

    from .target_results import _figure_payload, _safe_figure_id

    normalized = bool(_setting(visualization_settings, "normalized", False))
    top_k = _setting(visualization_settings, "top_k", 15)
    rank_by = str(_setting(visualization_settings, "rank_by", "value"))
    include_negative = bool(
        _setting(visualization_settings, "include_negative", True)
    )
    show_error_bars = bool(
        _setting(visualization_settings, "show_error_bars", True)
    )
    targets = [
        str(row.get("output_name")) for row in payload.get("overall") or ()
    ]
    targets = list(dict.fromkeys(targets))
    figures: list[dict[str, Any]] = []

    for target in targets:
        selected = [
            dict(row)
            for row in rows
            if str(row.get("output_name")) == target
            and str(row.get("importance_kind")) == "predictive"
            and str(row.get("method")) == "permutation"
        ]
        if not include_negative:
            selected = [
                row
                for row in selected
                if (_safe_float(row.get("normalized_mean" if normalized else "mean")) or 0.0)
                >= 0.0
            ]
        value_key = "normalized_mean" if normalized else "mean"
        selected.sort(
            key=lambda row: abs(_safe_float(row.get(value_key)) or 0.0)
            if rank_by == "absolute"
            else (_safe_float(row.get(value_key)) or float("-inf")),
            reverse=True,
        )
        if top_k is not None:
            selected = selected[: max(int(top_k), 1)]
        if not selected:
            continue

        labels = [str(row.get("feature")) for row in selected]
        values = [_safe_float(row.get(value_key)) or 0.0 for row in selected]
        errors = []
        for row in selected:
            between_fold = _safe_float(row.get("between_fold_std"))
            repeat_std = _safe_float(row.get("std"))
            errors.append(between_fold if between_fold is not None else (repeat_std or 0.0))

        figure = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                error_x=(
                    {"type": "data", "array": errors, "visible": True}
                    if show_error_bars and not normalized
                    else None
                ),
                customdata=[
                    [
                        row.get("rank"),
                        row.get("metric_name"),
                        row.get("baseline_metric"),
                        row.get("feature_type"),
                        row.get("role"),
                    ]
                    for row in selected
                ],
                hovertemplate=(
                    "feature=%{y}<br>importance=%{x:.6g}<br>rank=%{customdata[0]}"
                    "<br>metric=%{customdata[1]}<br>baseline=%{customdata[2]}"
                    "<br>type=%{customdata[3]}<br>role=%{customdata[4]}<extra></extra>"
                ),
            )
        )
        figure.add_vline(x=0, line_width=1, line_color="gray")
        figure.update_layout(
            xaxis_title=(
                "正規化重要度" if normalized else "評価指標の悪化量"
            ),
            yaxis_title="説明変数",
        )
        figure.update_yaxes(autorange="reversed")
        figures.append(
            _figure_payload(
                figure,
                figure_id=(
                    f"feature-importance-{_safe_figure_id(target)}-"
                    "predictive-permutation"
                ),
                title=f"{target}: Permutation Importance",
                description=(
                    "組成の変換座標は個別表示せず、全座標を同時に入れ替えた"
                    "「組成全体」として通常変数と比較します。"
                ),
            )
        )
    return figures


def postprocess_composition_feature_importance(
    result: dict[str, Any],
    request: Any,
) -> None:
    """Replace coordinate-level PI and rebuild affected Web figures."""

    payload = result.get("composition_feature_importance")
    if not isinstance(payload, dict):
        return
    summary = _replace_predictive_summary(result, payload)

    settings = getattr(request, "feature_importance", None)
    visualization_settings = getattr(settings, "visualization", None)
    targets = {
        str(row.get("output_name")) for row in payload.get("overall") or ()
    }
    existing = [
        dict(figure)
        for figure in list(result.get("feature_importance_visualizations") or ())
        if not (
            "predictive-permutation" in str(figure.get("id") or "").lower()
            and any(_matches_target(figure, target) for target in targets)
        )
    ]
    existing.extend(
        _composition_predictive_figures(
            summary,
            payload,
            visualization_settings,
        )
    )
    result["feature_importance_visualizations"] = existing


def install_composition_feature_importance_views() -> None:
    """Install response postprocessing before app.py binds the workflow."""

    from . import workflows

    if getattr(
        workflows,
        "_composition_feature_importance_views_installed",
        False,
    ):
        return
    original = workflows.run_regression_web_workflow

    def workflow_adapter(request: Any, store: Any) -> dict[str, Any]:
        result = original(request, store)
        postprocess_composition_feature_importance(result, request)
        from .logging import current_request_id
        from .visualization_sessions import get_visualization_session

        run_id = current_request_id()
        if run_id:
            try:
                get_visualization_session(run_id).result = copy.deepcopy(result)
            except KeyError:
                pass
        return result

    workflows.run_regression_web_workflow = workflow_adapter
    workflows._composition_feature_importance_views_installed = True


__all__ = [
    "install_composition_feature_importance_views",
    "postprocess_composition_feature_importance",
]
