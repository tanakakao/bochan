"""Bounded in-memory sessions for interactive Web result visualizations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any

from .target_results import _display_predictions, _figure_payload


@dataclass
class VisualizationSession:
    """Objects required to render plots after a one-shot Web optimization run."""

    optimizer: Any
    tabular_optimizer: Any
    data: Any
    encoded_targets: Any
    feature_columns: list[str]
    target_columns: list[str]
    target_metadata: dict[str, dict[str, Any]]
    hybrid_model: bool
    feature_constraints: list[Any] = field(default_factory=list)
    candidate_result: Any | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    request_details: dict[str, Any] = field(default_factory=dict)


_LOCK = RLock()
_SESSIONS: OrderedDict[str, VisualizationSession] = OrderedDict()
_PENDING: dict[str, dict[str, Any]] = {}
_MAX_SESSIONS = 12


def begin_visualization_run(run_id: str, request: Any) -> None:
    """Capture request settings before the existing Web workflow begins."""

    from .search_settings import normalize_feature_constraints

    feature_columns = list(request.feature_columns)
    constraints = normalize_feature_constraints(
        list(request.constraints or []),
        feature_columns=feature_columns,
    )
    _PENDING[run_id] = {
        "feature_constraints": constraints,
        "request_details": {
            "requested_model_type": str(request.model_type),
            "requested_acquisition": str(request.acquisition.name),
            "requested_optimizer": str(request.optimizer.name),
            "normalize": bool(request.normalize),
            "input_perturbation": bool(request.input_perturbation),
            "n_w": int(request.n_w),
            "perturbation_std": float(request.perturbation_std),
            "fit_maxiter": int(request.fit_maxiter),
            "model_kwargs": dict(request.model_kwargs or {}),
        },
    }


def attach_fitted_tabular_optimizer(
    run_id: str,
    *,
    tabular_optimizer: Any,
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    target_metadata: dict[str, dict[str, Any]],
    hybrid_model: bool,
) -> None:
    """Attach a fitted tabular optimizer and wrap candidate generation for capture."""

    import pandas as pd

    dataset = tabular_optimizer.dataset
    if dataset is None or dataset.Y is None:
        raise RuntimeError("Cannot register visualization session without fitted X/Y data.")
    pending = _PENDING.get(run_id, {})
    encoded_targets = pd.DataFrame(
        dataset.Y.detach().cpu().numpy(),
        columns=list(target_columns),
        index=data.index,
    )
    session = VisualizationSession(
        optimizer=tabular_optimizer.bo,
        tabular_optimizer=tabular_optimizer,
        data=data.copy(),
        encoded_targets=encoded_targets,
        feature_columns=list(feature_columns),
        target_columns=list(target_columns),
        target_metadata=target_metadata,
        hybrid_model=bool(hybrid_model),
        feature_constraints=list(pending.get("feature_constraints") or []),
        request_details=dict(pending.get("request_details") or {}),
    )
    register_visualization_session(run_id, session)

    original_candidate = tabular_optimizer.candidate

    @wraps(original_candidate)
    def captured_candidate(*args: Any, **kwargs: Any) -> Any:
        result = original_candidate(*args, **kwargs)
        if bool(kwargs.get("return_result", False)):
            with _LOCK:
                stored = _SESSIONS.get(run_id)
                if stored is not None:
                    stored.candidate_result = result
        return result

    tabular_optimizer.candidate = captured_candidate


def finalize_visualization_run(run_id: str, result: dict[str, Any]) -> VisualizationSession:
    """Attach displayed candidate rows and replace raw candidates by repaired values."""

    import torch

    session = get_visualization_session(run_id)
    session.rows = list(result.get("candidates") or [])
    if session.candidate_result is not None and session.rows:
        raw_values = [row.get("raw", {}).get("candidate") for row in session.rows]
        if all(value is not None for value in raw_values):
            reference = session.candidate_result.candidates
            session.candidate_result.candidates = torch.as_tensor(
                raw_values,
                dtype=reference.dtype,
                device=reference.device,
            )
    _PENDING.pop(run_id, None)
    return session


def discard_visualization_run(run_id: str) -> None:
    """Discard incomplete session state after a failed workflow."""

    with _LOCK:
        _PENDING.pop(run_id, None)
        _SESSIONS.pop(run_id, None)


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
    return [name for index, name in enumerate(session.feature_columns) if index not in cat_dims]


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
        groups.append({"features": names, "sum_value": float(constraint.rhs)})
    return groups


def visualization_options(session: VisualizationSession) -> dict[str, Any]:
    """Return valid selectors and slice controls for the interactive result UI."""

    regression_targets = [
        target
        for target in session.target_columns
        if session.target_metadata[target].get("internal_task") == "regression"
    ]
    numeric = set(_numeric_features(session))
    feature_controls: dict[str, dict[str, Any]] = {}
    for feature in session.feature_columns:
        series = session.data[feature].dropna()
        if feature in numeric:
            feature_controls[feature] = {
                "kind": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "default": float(series.median()),
            }
        else:
            values = list(dict.fromkeys(series.tolist()))
            feature_controls[feature] = {
                "kind": "categorical",
                "values": values,
                "default": values[0] if values else "",
            }
    return {
        "feature_columns": list(session.feature_columns),
        "numeric_features": _numeric_features(session),
        "target_columns": list(session.target_columns),
        "regression_targets": regression_targets,
        "ternary_groups": _ternary_groups(session),
        "feature_controls": feature_controls,
    }


def model_details(session: VisualizationSession, result: dict[str, Any]) -> dict[str, Any]:
    """Return compact, JSON-safe details about the actual fitted execution graph."""

    model = session.optimizer.model
    submodels = list(getattr(model, "models", []) or [])
    specs = list(getattr(model, "specs", []) or [])
    candidate_result = session.candidate_result
    acqf = getattr(candidate_result, "acqf", None)
    metadata = dict(result.get("metadata") or {})
    return {
        "optimizer_backend": "TabularBayesianOptimizer",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "hybrid_model": bool(session.hybrid_model),
        "submodel_classes": [f"{type(value).__module__}.{type(value).__name__}" for value in submodels],
        "output_specs": [
            {
                "name": getattr(spec, "name", None),
                "task_type": getattr(spec, "task_type", None),
                "model_class": (
                    f"{type(spec.model).__module__}.{type(spec.model).__name__}"
                    if getattr(spec, "model", None) is not None
                    else None
                ),
            }
            for spec in specs
        ],
        "requested_model_type": session.request_details.get("requested_model_type"),
        "internal_model_type": metadata.get("internal_model_type"),
        "requested_acquisition": metadata.get("requested_acquisition"),
        "effective_acquisition": metadata.get("acquisition"),
        "acquisition_class": (
            f"{type(acqf).__module__}.{type(acqf).__name__}" if acqf is not None else None
        ),
        "acquisition_family": metadata.get("acquisition_family"),
        "requested_search_method": session.request_details.get("requested_optimizer"),
        "effective_optimizer": metadata.get("optimizer"),
        "normalize": session.request_details.get("normalize"),
        "input_perturbation": session.request_details.get("input_perturbation"),
        "n_w": session.request_details.get("n_w"),
        "perturbation_std": session.request_details.get("perturbation_std"),
        "model_kwargs": session.request_details.get("model_kwargs", {}),
        "feature_names": list(session.feature_columns),
        "target_names": list(session.target_columns),
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
            raise RuntimeError(f"{target}: the fitted model did not expose class probabilities for YY plotting.")
        return show_multiclass_yyplot(
            session.data[target],
            probabilities.detach().cpu().numpy(),
            target=target,
            class_labels=list(metadata.get("classes") or []),
        )

    mean = display[target]["mean"].detach().cpu().numpy()
    std = display[target]["std"].detach().cpu().numpy()
    observed = session.encoded_targets[target] if task == "ordinal" else session.data[target]
    y = pd.DataFrame({target: observed})
    preds = (pd.DataFrame({target: mean}), pd.DataFrame({target: std}))
    return show_yyplot(y, target, preds=preds, df_cand=_candidate_dataframe(session))


def _pareto(session: VisualizationSession, target_x: str, target_y: str):
    from bochan.visualization import show_pareto_plot

    regression = set(visualization_options(session)["regression_targets"])
    if target_x == target_y:
        raise ValueError("Pareto plot requires two different target variables.")
    if target_x not in regression or target_y not in regression:
        raise ValueError("The existing Pareto Plotly implementation currently requires two regression targets.")
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
    available = visualization_options(session)["numeric_features"] if numeric_only else session.feature_columns
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
        return _figure_payload(
            _yyplot(session, target),
            figure_id=f"yyplot-{target}",
            title=f"{target}: YY plot",
            description="既存のPlotly YY plotで実測値と予測値を比較します。",
        )

    if kind == "pareto":
        target_x = str(request.get("target_x") or "")
        target_y = str(request.get("target_y") or "")
        return _figure_payload(
            _pareto(session, target_x, target_y),
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

        feature_x, feature_y = _require_features(session, features, count=2, numeric_only=True)
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
        label = "予測" if show_type == "pred" else "獲得関数"
        return _figure_payload(
            figure,
            figure_id=f"prediction-2d-{feature_x}-{feature_y}-{target}",
            title=f"{feature_x} × {feature_y} → {target}",
            description=f"既存のPlotly 2次元{label}プロットです。",
        )

    if kind == "ternary":
        from bochan.visualization import show_triscatter_with_acqf_from_optimizer

        feature_a, feature_b, feature_c = _require_features(session, features, count=3, numeric_only=True)
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
                    "Add a three-variable equality constraint with coefficient 1, or provide sum_value."
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
    "attach_fitted_tabular_optimizer",
    "begin_visualization_run",
    "build_visualization",
    "discard_visualization_run",
    "finalize_visualization_run",
    "get_visualization_session",
    "model_details",
    "register_visualization_session",
    "visualization_options",
]
