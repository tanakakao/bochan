"""Compatibility fixes for composition-aware Web result visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np


def _object_backed_string_columns(frame: Any) -> Any:
    """Return a copy whose extension string/category columns accept integer encoding.

    Pandas 3 infers ``StringDtype`` for newly generated string columns.  The
    tabular converter subsequently replaces categorical labels by integer codes,
    which cannot be assigned back into a StringArray.  Object-backed columns keep
    the established converter contract without changing user-visible values.
    """

    import pandas as pd

    result = frame.copy()
    for column in result.columns:
        dtype = result[column].dtype
        string_extension = (
            pd.api.types.is_string_dtype(dtype)
            and not pd.api.types.is_object_dtype(dtype)
        )
        if string_extension or isinstance(dtype, pd.CategoricalDtype):
            result[column] = result[column].astype(object)
    return result


def _constant_composition_grid(
    context: Any,
    baseline: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the baseline composition through the existing closure routine."""

    from . import composition_visualization as composition

    return composition._resolve_fraction_matrix(
        context,
        baseline=baseline,
        axis_values={
            context.fraction_features[0]: np.full(n_rows, baseline[0], dtype=float)
        },
        mode="proportional",
        balance_element=None,
    )


def _build_ordinary_axis_composition_visualization(
    session: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Plot ordinary source variables while holding the composition at baseline."""

    import pandas as pd

    from bochan.visualization import show_1dplot_with_pred, show_scatter_with_acqf

    from . import composition_visualization as composition
    from .target_results import _figure_payload

    context = composition._composition_context(session)
    if context is None:
        raise ValueError("A fitted single-composition model is required.")

    kind = str(request.get("kind", "1d")).lower()
    if kind not in {"1d", "2d"}:
        raise ValueError("Ordinary-axis composition plotting supports 1d and 2d only.")
    target = str(request.get("target") or session.target_columns[0])
    if target not in session.target_columns:
        raise ValueError(f"target must be one of {session.target_columns!r}.")
    features = [str(value) for value in list(request.get("features") or [])]
    expected = 1 if kind == "1d" else 2
    if len(features) != expected or len(set(features)) != expected:
        raise ValueError(f"{kind} requires {expected} different feature variables.")
    if context.column in features:
        return _unavailable_payload(
            kind,
            "組成式列そのものは数値軸にできません。元素比率または通常の数値変数を選択してください。",
        )

    fixed_values = dict(request.get("fixed_values") or {})
    fixed_values.pop(composition._MODE_KEY, None)
    fixed_values.pop(composition._BALANCE_KEY, None)
    baseline = composition._baseline_fractions(session, context, fixed_values)
    show_type = str(request.get("show_type") or "pred")
    n = max(10, min(int(request.get("n") or 50), 150))
    observed = composition._display_training_frame(session, context)
    observed_targets = session.data.loc[:, session.target_columns]
    candidates = composition._candidate_dataframe(session)

    if kind == "1d":
        feature = features[0]
        lower, upper = composition._axis_bounds(session, context, feature)
        grid = np.linspace(lower, upper, n)
        fractions, base_valid = _constant_composition_grid(context, baseline, n)
        values, std = composition._evaluate_grid(
            session,
            context,
            fractions=fractions,
            base_valid=base_valid,
            ordinary_axis_values={feature: grid},
            fixed_values=fixed_values,
            target=target,
            show_type="pred",
        )
        figure = show_1dplot_with_pred(
            feature,
            target,
            (pd.DataFrame({target: values}), pd.DataFrame({target: std}), grid),
            observed,
            observed_targets,
            candidates,
        )
        return _figure_payload(
            figure,
            figure_id=f"composition-fixed-1d-{feature}-{target}",
            title=f"{feature} → {target}",
            description="平均組成を固定し、通常変数を変化させた1次元予測です。",
        )

    feature_x, feature_y = features
    x_lower, x_upper = composition._axis_bounds(session, context, feature_x)
    y_lower, y_upper = composition._axis_bounds(session, context, feature_y)
    x_grid = np.linspace(x_lower, x_upper, min(n, 50))
    y_grid = np.linspace(y_lower, y_upper, min(n, 50))
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    flat_size = mesh_x.size
    fractions, base_valid = _constant_composition_grid(
        context,
        baseline,
        flat_size,
    )
    values, _std = composition._evaluate_grid(
        session,
        context,
        fractions=fractions,
        base_valid=base_valid,
        ordinary_axis_values={
            feature_x: mesh_x.reshape(-1),
            feature_y: mesh_y.reshape(-1),
        },
        fixed_values=fixed_values,
        target=target,
        show_type=show_type,
    )
    surface = values.reshape(len(y_grid), len(x_grid))
    figure = show_scatter_with_acqf(
        feature_x,
        feature_y,
        target,
        (surface, x_grid, y_grid),
        observed,
        observed_targets,
        candidates,
        show_type=show_type,
    )
    return _figure_payload(
        figure,
        figure_id=f"composition-fixed-2d-{feature_x}-{feature_y}-{target}",
        title=f"{feature_x} × {feature_y} → {target}",
        description="平均組成を固定し、通常変数を変化させた2次元予測です。",
    )


def _unavailable_payload(kind: str, message: str) -> dict[str, Any]:
    """Return an explanatory Plotly payload instead of an HTTP 400 error."""

    import plotly.graph_objects as go

    from .target_results import _figure_payload

    figure = go.Figure()
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        align="center",
        font={"size": 15},
    )
    figure.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 24, "r": 24, "t": 36, "b": 24},
    )
    return _figure_payload(
        figure,
        figure_id=f"composition-{kind}-unavailable",
        title="この可視化は現在の組成設定では利用できません",
        description=message,
    )


def install_composition_visualization_compat() -> None:
    """Install routing, Pandas 3, and graceful ternary compatibility fixes."""

    from . import composition_visualization as composition
    from . import visualization_sessions

    if getattr(visualization_sessions, "_composition_visualization_compat", False):
        return

    original_predict_values = composition._predict_values
    original_acquisition_values = composition._acquisition_values
    current_build = visualization_sessions.build_visualization
    current_options = visualization_sessions.visualization_options

    def predict_values_adapter(
        session: Any,
        source: Any,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        return original_predict_values(
            session,
            _object_backed_string_columns(source),
            target,
        )

    def acquisition_values_adapter(session: Any, source: Any) -> np.ndarray:
        return original_acquisition_values(
            session,
            _object_backed_string_columns(source),
        )

    def options_adapter(session: Any) -> dict[str, Any]:
        options = current_options(session)
        context = composition._composition_context(session)
        if context is None:
            return options
        result = dict(options)
        result["feature_columns"] = [
            feature
            for feature in list(result.get("feature_columns") or ())
            if feature != context.column
        ]
        return result

    def build_adapter(run_id: str, request: dict[str, Any]) -> dict[str, Any]:
        session = visualization_sessions.get_visualization_session(run_id)
        context = composition._composition_context(session)
        if context is None:
            return current_build(run_id, request)

        kind = str(request.get("kind", "1d")).lower()
        features = [str(value) for value in list(request.get("features") or [])]
        composition_axes = composition._composition_axes(context, features)

        if kind in {"1d", "2d"} and not composition_axes:
            return _build_ordinary_axis_composition_visualization(session, request)

        if kind == "ternary" and (
            len(context.elements) != 3
            or set(features) != set(context.fraction_features)
        ):
            return _unavailable_payload(
                "ternary",
                "組成の三角図は、候補元素がちょうど3種類で、3軸すべてにその元素比率を選択した場合に利用できます。",
            )
        return current_build(run_id, request)

    composition._predict_values = predict_values_adapter
    composition._acquisition_values = acquisition_values_adapter
    visualization_sessions.visualization_options = options_adapter
    visualization_sessions.build_visualization = build_adapter
    visualization_sessions._composition_visualization_compat = True


__all__ = ["install_composition_visualization_compat"]
