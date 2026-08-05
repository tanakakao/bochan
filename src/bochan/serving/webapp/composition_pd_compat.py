"""Partial-dependence compatibility for composition-enabled Web plots."""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_PD_ROWS = 256


def _sample_source_rows(session: Any) -> Any:
    """Return bounded, complete source rows for responsive Web PD evaluation."""

    frame = session.data.loc[:, session.feature_columns].dropna().copy()
    if frame.empty:
        raise ValueError("Partial dependence requires at least one complete training row.")
    if len(frame) <= _MAX_PD_ROWS:
        return frame
    indices = np.linspace(0, len(frame) - 1, _MAX_PD_ROWS, dtype=int)
    return frame.iloc[np.unique(indices)].copy()


def _axis_values(session: Any, context: Any, feature: str, n: int) -> np.ndarray:
    """Build numeric or categorical values for one PD axis."""

    from . import composition_visualization as composition

    if composition._element_for_feature(context, feature) is not None:
        lower, upper = composition._axis_bounds(session, context, feature)
        return np.linspace(lower, upper, n)
    if feature not in session.data.columns:
        raise ValueError(f"Unknown visualization feature {feature!r}.")

    series = session.data[feature].dropna()
    if series.empty:
        raise ValueError(f"No observed value is available for {feature!r}.")
    try:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series.dtype):
            lower = float(series.min())
            upper = float(series.max())
            if lower == upper:
                padding = 0.5 if lower == 0.0 else max(abs(lower) * 0.05, 1e-9)
                lower, upper = lower - padding, upper + padding
            return np.linspace(lower, upper, n)
    except (TypeError, ValueError):
        pass
    return np.asarray(list(dict.fromkeys(series.tolist()))[:n], dtype=object)


def _vary_fraction_rows(
    context: Any,
    baselines: np.ndarray,
    feature: str,
    value: float,
    *,
    mode: str,
    balance_element: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vary one element ratio independently for every observed composition row."""

    from . import composition_visualization as composition

    element = composition._element_for_feature(context, feature)
    if element is None:
        raise ValueError(f"{feature!r} is not an element-fraction feature.")
    selected_index = context.elements.index(element)
    fractions = np.asarray(baselines, dtype=float).copy()
    fractions[:, selected_index] = float(value)
    remaining = [index for index in range(len(context.elements)) if index != selected_index]
    residual = 1.0 - float(value)
    valid = np.isfinite(fractions).all(axis=1) & (residual >= -1e-10)

    balance_index = (
        context.elements.index(balance_element)
        if mode == "balance" and balance_element in context.elements
        else None
    )
    if balance_index == selected_index:
        balance_index = None

    if balance_index is not None:
        fixed_indices = [index for index in remaining if index != balance_index]
        fixed_total = fractions[:, fixed_indices].sum(axis=1) if fixed_indices else 0.0
        fractions[:, balance_index] = residual - fixed_total
    else:
        weights = np.clip(fractions[:, remaining], 0.0, None)
        totals = weights.sum(axis=1, keepdims=True)
        zero_rows = totals[:, 0] <= 0.0
        if zero_rows.any():
            weights[zero_rows] = 1.0
            totals = weights.sum(axis=1, keepdims=True)
        fractions[:, remaining] = residual * weights / totals

    valid &= np.isfinite(fractions).all(axis=1)
    valid &= (fractions >= -1e-9).all(axis=1)
    fractions = np.where(np.abs(fractions) < 1e-12, 0.0, fractions)
    row_sums = fractions.sum(axis=1)
    valid &= row_sums > 0.0
    valid &= np.abs(row_sums - 1.0) <= 1e-7
    fractions[valid] = fractions[valid] / row_sums[valid, None]
    return fractions, valid


def _aggregate_prediction(mean: np.ndarray, std: np.ndarray) -> tuple[float, float]:
    """Aggregate row-wise predictive distributions using total variance."""

    finite = np.isfinite(mean) & np.isfinite(std)
    if not finite.any():
        return float("nan"), float("nan")
    resolved_mean = np.asarray(mean[finite], dtype=float)
    resolved_std = np.asarray(std[finite], dtype=float)
    average = float(np.mean(resolved_mean))
    second_moment = float(np.mean(resolved_std**2 + resolved_mean**2))
    variance = max(second_moment - average**2, 0.0)
    return average, float(np.sqrt(variance))


def _build_partial_dependence_1d(
    session: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Average predictions while retaining each observed composition and context."""

    import pandas as pd

    from bochan.visualization import show_1dplot_with_pred

    from . import composition_visualization as composition
    from .target_results import _figure_payload

    context = composition._composition_context(session)
    if context is None:
        raise ValueError("A fitted single-composition model is required.")
    target = str(request.get("target") or session.target_columns[0])
    if target not in session.target_columns:
        raise ValueError(f"target must be one of {session.target_columns!r}.")
    features = [str(value) for value in list(request.get("features") or [])]
    if len(features) != 1:
        raise ValueError("1d partial dependence requires one feature variable.")
    feature = features[0]
    if feature == context.column:
        raise ValueError("組成式列そのものではなく、元素比率または通常変数を選択してください。")

    n = max(10, min(int(request.get("n") or 50), 150))
    grid = _axis_values(session, context, feature, n)
    source_rows = _sample_source_rows(session)
    composition_feature = composition._element_for_feature(context, feature) is not None
    mode, balance_element = composition._request_mode(request)

    baselines: np.ndarray | None = None
    if composition_feature:
        observed = composition._observed_composition_frame(session).loc[source_rows.index]
        baselines = observed.loc[:, list(context.fraction_features)].to_numpy(dtype=float)

    blocks: list[Any] = []
    block_sizes: list[int] = []
    for value in grid:
        source = source_rows.copy()
        if composition_feature:
            assert baselines is not None
            fractions, base_valid = _vary_fraction_rows(
                context,
                baselines,
                feature,
                float(value),
                mode=mode,
                balance_element=balance_element,
            )
            constrained = base_valid & composition._composition_validity(
                session,
                context,
                fractions,
            )
            valid = constrained if constrained.any() else base_valid
            source = source.loc[valid].copy()
            fractions = fractions[valid]
            source[context.column] = composition._formula_strings(context, fractions)
        else:
            source[feature] = value
        if source.empty:
            block_sizes.append(0)
            continue
        blocks.append(source)
        block_sizes.append(len(source))

    if not blocks:
        raise ValueError("No valid rows are available for partial-dependence prediction.")
    stacked = pd.concat(blocks, ignore_index=True)
    predicted_mean, predicted_std = composition._predict_values(session, stacked, target)

    pd_mean: list[float] = []
    pd_std: list[float] = []
    offset = 0
    for size in block_sizes:
        if size <= 0:
            pd_mean.append(float("nan"))
            pd_std.append(float("nan"))
            continue
        next_offset = offset + size
        mean_value, std_value = _aggregate_prediction(
            predicted_mean[offset:next_offset],
            predicted_std[offset:next_offset],
        )
        pd_mean.append(mean_value)
        pd_std.append(std_value)
        offset = next_offset

    display_training = composition._display_training_frame(session, context)
    observed_targets = session.data.loc[:, session.target_columns]
    candidates = composition._candidate_dataframe(session)
    figure = show_1dplot_with_pred(
        feature,
        target,
        (
            pd.DataFrame({target: pd_mean}),
            pd.DataFrame({target: pd_std}),
            grid,
        ),
        display_training,
        observed_targets,
        candidates,
    )
    figure.update_xaxes(title_text=composition._feature_label(context, feature))
    return _figure_payload(
        figure,
        figure_id=f"composition-pd-1d-{feature}-{target}",
        title=f"{composition._feature_label(context, feature)} → {target}: PD",
        description=(
            "各学習行の組成と他の説明変数を保持したまま選択変数だけを置換し、"
            "予測分布を平均したPartial Dependenceです。"
        ),
    )


def install_composition_pd_compat() -> None:
    """Use finite row-averaged PD for composition-enabled one-dimensional plots."""

    from . import composition_visualization as composition
    from . import composition_visualization_compat as compat
    from . import visualization_sessions

    if getattr(visualization_sessions, "_composition_pd_compat_installed", False):
        return

    original_composition_builder = composition._build_composition_visualization
    original_ordinary_builder = compat._build_ordinary_axis_composition_visualization

    def composition_builder(session: Any, request: dict[str, Any]) -> dict[str, Any]:
        target = str(request.get("target") or session.target_columns[0])
        task = str(
            session.target_metadata.get(target, {}).get("internal_task") or "regression"
        )
        if (
            str(request.get("kind", "1d")).lower() == "1d"
            and str(request.get("show_type") or "pred") == "pred"
            and task == "regression"
        ):
            return _build_partial_dependence_1d(session, request)
        return original_composition_builder(session, request)

    def ordinary_builder(session: Any, request: dict[str, Any]) -> dict[str, Any]:
        target = str(request.get("target") or session.target_columns[0])
        task = str(
            session.target_metadata.get(target, {}).get("internal_task") or "regression"
        )
        if (
            str(request.get("kind", "1d")).lower() == "1d"
            and str(request.get("show_type") or "pred") == "pred"
            and task == "regression"
        ):
            return _build_partial_dependence_1d(session, request)
        return original_ordinary_builder(session, request)

    composition._build_composition_visualization = composition_builder
    compat._build_ordinary_axis_composition_visualization = ordinary_builder
    visualization_sessions._composition_pd_compat_installed = True


__all__ = ["install_composition_pd_compat"]
