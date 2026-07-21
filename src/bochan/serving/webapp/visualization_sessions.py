"""Bounded in-memory sessions for interactive Web result visualizations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .target_results import _display_predictions, _figure_payload


@dataclass
class VisualizationSession:
    """Objects required to render plots after a one-shot Web optimization run."""

    optimizer: Any
    tabular_optimizer: Any
    candidate_result: Any
    data: Any
    encoded_targets: Any
    rows: list[dict[str, Any]]
    feature_columns: list[str]
    target_columns: list[str]
    target_metadata: dict[str, dict[str, Any]]
    hybrid_model: bool
    feature_constraints: list[Any]


_LOCK = RLock()
_SESSIONS: OrderedDict[str, VisualizationSession] = OrderedDict()
_MAX_SESSIONS = 12


def register_visualization_session(run_id: str, session: VisualizationSession) -> None:
    """Store a fitted run while keeping process memory bounded."""

    with _LOCK:
        _SESSIONS.pop(run_id, None)
        _SESSIONS[run_id] = session
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)


def get_visualization_session(run_id: str) -> VisualizationSession:
    """Return one fitted session and refresh its LRU position."""

    with _LOCK:
        try:
            session = _SESSIONS.pop(run_id)
        except KeyError as exc:
            raise KeyError(
                "Visualization session was not found. Run optimization again; "
                "sessions are stored only in the current FastAPI process."
            ) from exc
        _SESSIONS[run_id] = session
        return session


def _numeric_features(session: VisualizationSession) -> list[str]:
    cat_dims = set(int(value) for value in (session.tabular_optimizer.dataset.cat_dims or []))
    return [
        name
        for index, name in enumerate(session.feature_columns)
        if index not in cat_dims
    ]


def _ternary_groups(session: VisualizationSession) -> list[dict[str, Any]]:
    """Find three-term unit-coefficient equality constraints usable as ternary sums."""

    groups: list[dict[str, Any]] = []
    numeric = set(_numeric_features(session))
    for constraint in session.feature_constraints:
        if str(getattr(constraint, "sense", "")) != "eq":
            continue
        terms = list(getattr(constraint, "terms", []) or [])
        names = [str(term.column) for term in terms]
        coefficients = [float(term.coefficient) for term in terms]
        if len(names) != 3 or not set(names).issubset(numeric):
            continue
        if not all(abs(coefficient - 1.0) <= 1e-12 for coefficient in coefficients):
            continue
        groups.append(
            {
                "features": names,
                "sum_value": float(constraint.rhs),
            }
        )
    return groups


def visualization_options(session: VisualizationSession) -> dict[str, Any]:
    """Return valid selectors for the interactive result UI."""

    regression_targets = [
        target
        for target in session.target_columns
        if session.target_metadata[target].get("internal_task") == "regression"
    ]
    return {
        "feature_columns": list(session.feature_columns),
        "numeric_features": _numeric_features(session),
        "target_columns": list(session.target_columns),
        "regression_targets": regression_targets,
        "ternary_groups": _ternary_groups(session),
    }


def _candidate_dataframe(session: VisualizationSession):
    import pandas as pd

    records: list[dict[str, Any]] = []
    for row in session.rows:
        record = dict(row["values"])
        for target, prediction in row["predictions"].items():
            record[f"{target}_mean"] = prediction["mean"]
            record[f"{target}_std"] = prediction["std"]
        record["acq_value"] = row.get("acq_value")
        records.append(record)
    return pd.DataFrame(records)


def _yyplot(session: VisualizationSession, target: str):
    if target not in session.target_columns:
        raise ValueError(f"target must be one of {session.target_columns!r}.")

    import pandas as pd

    from bochan.visualization import show_multiclass_yyplot, show_yyplot

    display, class_probabilities = _display_predictions(
        session.optimizer,
        session.tabular_optimizer.dataset.X,
        target_columns=session.target_columns,
        target_metadata=session.target_metadata,
        hybrid_model=session.hybrid_model,
    )
    metadata = session.target_metadata[target]
    task = str(metadata["internal_task"])
    if task in {"binary", "multiclass"}:
        probabilities = class_probabilities.get(target)
        if probabilities is None:
            raise RuntimeError(
                f"{target}: the fitted model did not expose class probabilities for YY plotting."
            )
        return show_multiclass_yyplot(
            session.data[target],
            probabilities.detach().cpu().numpy(),
            target=target,
            class_labels=list(metadata.get("classes") or []),
        )

    mean = display[target]["mean"].detach().cpu().numpy()
    std = display[target]["std"].detach().cpu().numpy()
    if task == "ordinal":
        observed = session.encoded_targets[target]
    else:
        observed = session.data[target]
    y = pd.DataFrame({target: observed})
    preds = (
        pd.DataFrame({target: mean}),
        pd.DataFrame({target: std}),
    )
    return show_yyplot(
        y,
        target,
        preds=preds,
        df_cand=_candidate_dataframe(session),
    )


def _pareto(session: VisualizationSession, target_x: str, target_y: str):
    from bochan.visualization import show_pareto_plot

    regression = set(visualization_options(session)["regression_targets"])
    if target_x == target_y:
        raise ValueError("Pareto plot requires two different target variables.")
    if target_x not in regression or target_y not in regression:
        raise ValueError(
            "The existing Pareto Plotly implementation currently requires two "
            "regression targets."
        )
    return show_pareto_plot(
        session.data[session.target_columns],
        target_x,
        target_y,
        df_cand=_candidate_dataframe(session),
    )


def _require_features(
    session: VisualizationSession,
    values: list[str],
    *,
    count: int,
    numeric_only: bool,
) -> list[str]:
    available = (
        visualization_options(session)["numeric_features"]
        if numeric_only
        else session.feature_columns
    )
    if len(values) != count:
        raise ValueError(f"Exactly {count} feature variables are required.")
    if len(set(values)) != count:
        raise ValueError("Selected feature variables must be different.")
    missing = [value for value in values if value not in available]
    if missing:
        raise ValueError(f"Unsupported feature variables: {missing!r}; available={available!r}.")
    return values


def build_visualization(run_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Build one Plotly payload with the repository's existing plot functions."""

    session = get_visualization_session(run_id)
    kind = str(request.get("kind", "yyplot")).lower()
    target = str(request.get("target") or session.target_columns[0])
    features = [str(value) for value in list(request.get("features") or [])]
    fixed_values = dict(request.get("fixed_values") or {})
    n = max(10, min(int(request.get("n") or 50), 150))
    show_type = str(request.get("show_type") or "pred")
    if show_type not in {"pred", "acqf"}:
        raise ValueError("show_type must be pred or acqf.")

    if kind == "yyplot":
        figure = _yyplot(session, target)
        return _figure_payload(
            figure,
            figure_id=f"yyplot-{target}",
            title=f"{target}: YY plot",
            description="既存のPlotly YY plotで実測値と予測値を比較します。",
        )

    if kind == "pareto":
        target_x = str(request.get("target_x") or "")
        target_y = str(request.get("target_y") or "")
        figure = _pareto(session, target_x, target_y)
        return _figure_payload(
            figure,
            figure_id=f"pareto-{target_x}-{target_y}",
            title=f"{target_x} × {target_y}",
            description="既存のPlotly Pareto散布図で入力データと候補を比較します。",
        )

    if target not in session.target_columns:
        raise ValueError(f"target must be one of {session.target_columns!r}.")

    if kind == "1d":
        from bochan.visualization import show_1dplot_from_optimizer

        feature = _require_features(session, features, count=1, numeric_only=False)[0]
        figure = show_1dplot_from_optimizer(
            session.optimizer,
            feature,
            target,
            feature_cols=session.feature_columns,
            target_cols=session.target_columns,
            value_dict=fixed_values,
            candidate_result=session.candidate_result,
            n=n,
        )
        return _figure_payload(
            figure,
            figure_id=f"prediction-1d-{feature}-{target}",
            title=f"{feature} → {target}",
            description="既存のPlotly 1次元予測プロットです。",
        )

    if kind == "2d":
        from bochan.visualization import show_scatter_with_acqf_from_optimizer

        feature_x, feature_y = _require_features(
            session,
            features,
            count=2,
            numeric_only=True,
        )
        figure = show_scatter_with_acqf_from_optimizer(
            session.optimizer,
            feature_x,
            feature_y,
            target,
            feature_cols=session.feature_columns,
            target_cols=session.target_columns,
            value_dict=fixed_values,
            candidate_result=session.candidate_result,
            n=min(n, 50),
            show_type=show_type,
        )
        return _figure_payload(
            figure,
            figure_id=f"prediction-2d-{feature_x}-{feature_y}-{target}",
            title=f"{feature_x} × {feature_y} → {target}",
            description=f"既存のPlotly 2次元{('予測' if show_type == 'pred' else '獲得関数')}プロットです。",
        )

    if kind == "ternary":
        from bochan.visualization import show_triscatter_with_acqf_from_optimizer

        feature_a, feature_b, feature_c = _require_features(
            session,
            features,
            count=3,
            numeric_only=True,
        )
        sum_value = request.get("sum_value")
        if sum_value is None:
            matching = [
                group
                for group in _ternary_groups(session)
                if set(group["features"]) == {feature_a, feature_b, feature_c}
            ]
            if not matching:
                raise ValueError(
                    "The existing ternary Plotly implementation requires a sum value. "
                    "Add a three-variable equality constraint with coefficient 1, or "
                    "provide sum_value."
                )
            sum_value = matching[0]["sum_value"]
        figure = show_triscatter_with_acqf_from_optimizer(
            session.optimizer,
            feature_a,
            feature_b,
            feature_c,
            target,
            feature_cols=session.feature_columns,
            target_cols=session.target_columns,
            value_dict=fixed_values,
            candidate_result=session.candidate_result,
            sum_value=float(sum_value),
            n=min(n, 60),
            show_type=show_type,
        )
        return _figure_payload(
            figure,
            figure_id=f"ternary-{feature_a}-{feature_b}-{feature_c}-{target}",
            title=f"{feature_a} / {feature_b} / {feature_c} → {target}",
            description="既存のPlotly三角図です。",
        )

    raise ValueError("kind must be yyplot, pareto, 1d, 2d, or ternary.")


__all__ = [
    "VisualizationSession",
    "build_visualization",
    "get_visualization_session",
    "register_visualization_session",
    "visualization_options",
]
