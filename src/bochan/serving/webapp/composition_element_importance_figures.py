"""Plotly cards for element-wise composition perturbation importance."""

from __future__ import annotations

import copy
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    """Return one finite numeric value for Plotly serialization."""

    import math

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _element_figures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one element-wise PI card for each regression output."""

    import plotly.graph_objects as go

    from .target_results import _figure_payload, _safe_figure_id

    records = [dict(row) for row in payload.get("elements") or ()]
    outputs = list(
        dict.fromkeys(str(row.get("output_name") or "") for row in records)
    )
    figures: list[dict[str, Any]] = []
    for output in outputs:
        rows = [
            row for row in records if str(row.get("output_name") or "") == output
        ]
        rows.sort(key=lambda row: _finite(row.get("mean")), reverse=True)
        if not rows:
            continue
        labels = [
            str(row.get("label") or f"{row.get('feature')} 比率") for row in rows
        ]
        figure = go.Figure(
            go.Bar(
                x=[_finite(row.get("mean")) for row in rows],
                y=labels,
                orientation="h",
                error_x={
                    "type": "data",
                    "array": [_finite(row.get("std")) for row in rows],
                    "visible": True,
                },
                customdata=[
                    [
                        _finite(row.get("normalized_mean")),
                        row.get("metric_name"),
                        row.get("baseline_metric"),
                        row.get("n_repeats"),
                    ]
                    for row in rows
                ],
                hovertemplate=(
                    "element=%{y}<br>importance=%{x:.6g}"
                    "<br>composition-normalized=%{customdata[0]:.6g}"
                    "<br>metric=%{customdata[1]}"
                    "<br>baseline=%{customdata[2]}"
                    "<br>repeats=%{customdata[3]}<extra></extra>"
                ),
            )
        )
        figure.add_vline(x=0, line_width=1, line_color="gray")
        figure.update_layout(
            xaxis_title="評価指標の悪化量",
            yaxis_title="元素比率",
        )
        figure.update_yaxes(autorange="reversed")
        figures.append(
            _figure_payload(
                figure,
                figure_id=(
                    f"feature-importance-{_safe_figure_id(output)}-"
                    "predictive-composition-elements"
                ),
                title=f"{output}: 組成内の元素別影響度",
                description=(
                    f"各元素比率を入れ替え、{payload.get('mode_label', '残りの元素比を維持')}"
                    "ながら合計1と組成制約を維持したPermutation Importanceです。"
                    "組成内正規化値はhoverに表示します。元素単独の因果効果ではありません。"
                ),
            )
        )
    return figures


def append_element_importance_figures(result: dict[str, Any]) -> None:
    """Append element-wise cards without duplicating restored result figures."""

    payload = result.get("composition_feature_importance")
    if not isinstance(payload, dict):
        return
    try:
        generated = _element_figures(payload)
    except ImportError as exc:
        warning = (
            "Composition element-importance visualization requires the Web or "
            f"visualization extra: {exc}"
        )
        warnings = list(result.get("feature_importance_warnings") or ())
        if warning not in warnings:
            warnings.append(warning)
        result["feature_importance_warnings"] = warnings
        return
    if not generated:
        return
    generated_ids = {str(figure.get("id")) for figure in generated}
    existing = [
        dict(figure)
        for figure in list(result.get("feature_importance_visualizations") or ())
        if str(figure.get("id")) not in generated_ids
    ]
    existing.extend(generated)
    result["feature_importance_visualizations"] = existing


def install_composition_element_importance_figures() -> None:
    """Install element-card generation before app.py binds the workflow."""

    from . import workflows

    if getattr(
        workflows,
        "_composition_element_importance_figures_installed",
        False,
    ):
        return
    original = workflows.run_regression_web_workflow

    def workflow_adapter(request: Any, store: Any) -> dict[str, Any]:
        result = original(request, store)
        append_element_importance_figures(result)
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
    workflows._composition_element_importance_figures_installed = True


__all__ = [
    "append_element_importance_figures",
    "install_composition_element_importance_figures",
]
