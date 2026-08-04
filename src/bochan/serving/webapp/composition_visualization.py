"""Composition-ratio adapters for the existing Web Plotly visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import numpy as np

_FRACTION_TOKEN = "__fraction__"
_MODE_KEY = "__composition_mode__"
_BALANCE_KEY = "__composition_balance_element__"


@dataclass(frozen=True)
class _CompositionContext:
    """Resolved single-site composition metadata for one fitted Web session."""

    site_name: str
    column: str
    prefix: str
    elements: tuple[str, ...]
    fraction_features: tuple[str, ...]
    config: dict[str, Any]


def _composition_context(session: Any) -> _CompositionContext | None:
    optimizer = session.tabular_optimizer
    sites = dict(getattr(optimizer, "composition_sites", None) or {})
    if len(sites) != 1:
        return None
    site_name, raw_config = next(iter(sites.items()))
    transformer = dict(
        getattr(optimizer, "composition_transformers_", None) or {}
    ).get(site_name)
    if transformer is None:
        return None
    elements = tuple(str(value) for value in transformer._require_fitted())
    prefix = str(transformer.prefix)
    return _CompositionContext(
        site_name=str(site_name),
        column=str(raw_config["column"]),
        prefix=prefix,
        elements=elements,
        fraction_features=tuple(
            f"{prefix}{_FRACTION_TOKEN}{element}" for element in elements
        ),
        config=dict(raw_config),
    )


def _element_for_feature(context: _CompositionContext, feature: str) -> str | None:
    prefix = f"{context.prefix}{_FRACTION_TOKEN}"
    if not str(feature).startswith(prefix):
        return None
    element = str(feature)[len(prefix) :]
    return element if element in context.elements else None


def _feature_label(context: _CompositionContext, feature: str) -> str:
    element = _element_for_feature(context, feature)
    return f"{element} 比率" if element is not None else str(feature)


def _observed_composition_frame(session: Any) -> Any:
    optimizer = session.tabular_optimizer
    transformed = optimizer.transform_compositions(session.data)
    return optimizer.inverse_compositions(
        transformed,
        repair=False,
        keep_coordinates=False,
    )


def _normalized_default_fractions(
    frame: Any,
    context: _CompositionContext,
) -> np.ndarray:
    values = frame.loc[:, list(context.fraction_features)].to_numpy(dtype=float)
    defaults = np.nanmean(values, axis=0)
    defaults = np.where(np.isfinite(defaults), defaults, 0.0)
    defaults = np.clip(defaults, 0.0, None)
    total = float(defaults.sum())
    if total <= 0.0:
        defaults = np.full(len(context.elements), 1.0 / len(context.elements))
    else:
        defaults = defaults / total
    return defaults


def _fraction_bounds(
    context: _CompositionContext,
    element: str,
) -> tuple[float, float]:
    total = float(context.config.get("total", 1.0))
    configured = dict(context.config.get("bounds") or {}).get(
        element,
        (0.0, total),
    )
    return float(configured[0]) / total, float(configured[1]) / total


def _extend_visualization_options(
    options: dict[str, Any],
    session: Any,
) -> dict[str, Any]:
    context = _composition_context(session)
    if context is None:
        return options

    observed = _observed_composition_frame(session)
    defaults = _normalized_default_fractions(observed, context)
    result = dict(options)
    feature_columns = list(result.get("feature_columns") or ())
    numeric_features = list(result.get("numeric_features") or ())
    feature_controls = dict(result.get("feature_controls") or {})
    feature_labels = dict(result.get("feature_labels") or {})
    composition_features: list[dict[str, Any]] = []

    for index, (element, feature) in enumerate(
        zip(context.elements, context.fraction_features, strict=True)
    ):
        lower, upper = _fraction_bounds(context, element)
        default = float(np.clip(defaults[index], lower, upper))
        if feature not in feature_columns:
            feature_columns.append(feature)
        if feature not in numeric_features:
            numeric_features.append(feature)
        feature_controls[feature] = {
            "kind": "numeric",
            "min": lower,
            "max": upper,
            "default": default,
        }
        label = _feature_label(context, feature)
        feature_labels[feature] = label
        composition_features.append(
            {
                "name": feature,
                "label": label,
                "element": element,
                "min": lower,
                "max": upper,
                "default": default,
            }
        )

    ternary_groups = list(result.get("ternary_groups") or ())
    if len(context.elements) == 3:
        group = {
            "features": list(context.fraction_features),
            "sum_value": 1.0,
        }
        if group not in ternary_groups:
            ternary_groups.append(group)

    result.update(
        {
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "feature_controls": feature_controls,
            "feature_labels": feature_labels,
            "ternary_groups": ternary_groups,
            "composition": {
                "column": context.column,
                "elements": list(context.elements),
                "fraction_features": list(context.fraction_features),
                "features": composition_features,
                "default_mode": "proportional",
                "modes": ["proportional", "balance"],
            },
        }
    )
    return result


def _baseline_fractions(
    session: Any,
    context: _CompositionContext,
    fixed_values: dict[str, Any],
) -> np.ndarray:
    observed = _observed_composition_frame(session)
    baseline = _normalized_default_fractions(observed, context)
    for index, feature in enumerate(context.fraction_features):
        if feature in fixed_values:
            baseline[index] = float(fixed_values[feature])
    baseline = np.clip(baseline, 0.0, None)
    total = float(baseline.sum())
    if total <= 0.0:
        return np.full(len(context.elements), 1.0 / len(context.elements))
    return baseline / total


def _resolve_fraction_matrix(
    context: _CompositionContext,
    *,
    baseline: np.ndarray,
    axis_values: dict[str, np.ndarray],
    mode: str,
    balance_element: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve selected element axes to closed compositions and validity flags."""

    if not axis_values:
        raise ValueError("At least one composition-ratio axis is required.")
    lengths = {np.asarray(value).reshape(-1).shape[0] for value in axis_values.values()}
    if len(lengths) != 1:
        raise ValueError("Composition axis arrays must have the same length.")
    n_rows = lengths.pop()
    fractions = np.tile(np.asarray(baseline, dtype=float), (n_rows, 1))
    selected_indices: list[int] = []
    for feature, values in axis_values.items():
        element = _element_for_feature(context, feature)
        if element is None:
            continue
        index = context.elements.index(element)
        fractions[:, index] = np.asarray(values, dtype=float).reshape(-1)
        selected_indices.append(index)

    selected = set(selected_indices)
    remaining = [index for index in range(len(context.elements)) if index not in selected]
    selected_sum = fractions[:, selected_indices].sum(axis=1)
    residual = 1.0 - selected_sum
    valid = np.isfinite(fractions).all(axis=1) & (residual >= -1e-10)

    if remaining:
        balance_index = (
            context.elements.index(balance_element)
            if mode == "balance" and balance_element in context.elements
            else None
        )
        if balance_index in selected:
            balance_index = None
        if balance_index is not None:
            fixed_indices = [index for index in remaining if index != balance_index]
            fixed_total = fractions[:, fixed_indices].sum(axis=1) if fixed_indices else 0.0
            fractions[:, balance_index] = residual - fixed_total
        else:
            weights = np.clip(np.asarray(baseline)[remaining], 0.0, None)
            if float(weights.sum()) <= 0.0:
                weights = np.ones(len(remaining), dtype=float)
            weights = weights / weights.sum()
            fractions[:, remaining] = residual[:, None] * weights[None, :]
    else:
        valid &= np.abs(residual) <= 1e-7

    valid &= np.isfinite(fractions).all(axis=1)
    valid &= (fractions >= -1e-9).all(axis=1)
    fractions = np.where(np.abs(fractions) < 1e-12, 0.0, fractions)
    row_sums = fractions.sum(axis=1)
    valid &= np.abs(row_sums - 1.0) <= 1e-7
    nonzero = row_sums > 0.0
    fractions[nonzero] = fractions[nonzero] / row_sums[nonzero, None]
    return fractions, valid


def _composition_validity(
    session: Any,
    context: _CompositionContext,
    fractions: np.ndarray,
) -> np.ndarray:
    optimizer = session.tabular_optimizer
    valid = np.isfinite(fractions).all(axis=1)
    valid &= np.abs(fractions.sum(axis=1) - 1.0) <= 1e-7
    active = fractions > 1e-10
    total = float(context.config.get("total", 1.0))

    for index, element in enumerate(context.elements):
        lower, upper = _fraction_bounds(context, element)
        valid &= fractions[:, index] >= lower - 1e-9
        valid &= fractions[:, index] <= upper + 1e-9

    min_components = int(context.config.get("min_components", 1))
    max_components = context.config.get("max_components")
    counts = active.sum(axis=1)
    valid &= counts >= min_components
    if max_components is not None:
        valid &= counts <= int(max_components)
    for element in tuple(context.config.get("required_components") or ()):
        if element in context.elements:
            valid &= active[:, context.elements.index(element)]

    constraints = list(
        getattr(optimizer, "composition_element_constraints", None) or ()
    )
    for constraint in constraints:
        lhs = np.zeros(fractions.shape[0], dtype=float)
        for term in constraint["terms"]:
            if str(term.get("site")) != context.site_name:
                continue
            element = str(term["element"])
            if element not in context.elements:
                continue
            index = context.elements.index(element)
            scale = optimizer._basis_scale(
                context.config,
                element,
                constraint["basis"],
            )
            lhs += (
                float(term["coefficient"])
                * float(scale)
                * fractions[:, index]
                * total
            )
        rhs = float(constraint["rhs"])
        operator = str(constraint["operator"])
        if operator == "=":
            valid &= np.abs(lhs - rhs) <= 1e-7
        elif operator == "<=":
            valid &= lhs <= rhs + 1e-7
        else:
            valid &= lhs >= rhs - 1e-7
    return valid


def _formula_strings(
    context: _CompositionContext,
    fractions: np.ndarray,
) -> list[str]:
    from bochan.tabular.composition import (
        ATOMIC_WEIGHTS,
        close_compositions,
        format_formula,
    )

    basis = np.asarray(fractions, dtype=float)
    normalization = str(context.config.get("normalization", "atomic_fraction")).lower()
    if normalization in {"weight_fraction", "weight", "mass_fraction"}:
        weights = np.asarray(
            [ATOMIC_WEIGHTS[element] for element in context.elements],
            dtype=float,
        )
        atomic = close_compositions(basis / weights)
    else:
        atomic = close_compositions(basis)
    precision = int(context.config.get("precision", 6))
    return [
        format_formula(
            dict(zip(context.elements, row, strict=True)),
            order=context.elements,
            precision=precision,
        )
        for row in atomic
    ]


def _source_default(session: Any, column: str, fixed_values: dict[str, Any]) -> Any:
    series = session.data[column].dropna()
    if series.empty:
        raise ValueError(f"No observed value is available for {column!r}.")
    if column in fixed_values:
        from .tabular_backend import _category_key_from_label

        return _category_key_from_label(series, fixed_values[column])
    try:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series.dtype):
            return float(series.median())
    except (TypeError, ValueError):
        pass
    return series.iloc[0]


def _source_frame(
    session: Any,
    context: _CompositionContext,
    *,
    fractions: np.ndarray,
    ordinary_axis_values: dict[str, np.ndarray],
    fixed_values: dict[str, Any],
) -> Any:
    import pandas as pd

    n_rows = fractions.shape[0]
    frame = pd.DataFrame(index=np.arange(n_rows))
    for feature in session.feature_columns:
        if feature == context.column:
            frame[feature] = _formula_strings(context, fractions)
        elif feature in ordinary_axis_values:
            frame[feature] = np.asarray(ordinary_axis_values[feature]).reshape(-1)
        else:
            frame[feature] = _source_default(session, feature, fixed_values)
    return frame


def _predict_values(
    session: Any,
    source: Any,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = session.tabular_optimizer.predict(source)
    mean_column = f"{target}_mean"
    variance_column = f"{target}_variance"
    if mean_column not in prediction or variance_column not in prediction:
        raise ValueError(
            f"The fitted model did not expose numeric mean/variance for {target!r}."
        )
    mean = prediction[mean_column].to_numpy(dtype=float)
    variance = prediction[variance_column].to_numpy(dtype=float)
    return mean, np.sqrt(np.clip(variance, 0.0, None))


def _acquisition_values(session: Any, source: Any) -> np.ndarray:
    result = session.candidate_result
    acqf = getattr(result, "acqf", None)
    if acqf is None:
        raise ValueError("The acquisition function is unavailable for this run.")

    import torch

    from bochan.tabular.converter import dataframe_to_tensors

    optimizer = session.tabular_optimizer
    transformed = optimizer.transform_compositions(source)
    config = replace(
        optimizer.data_config,
        input_cols=optimizer.dataset.feature_names,
        target_cols=None,
    )
    X = dataframe_to_tensors(transformed, config).X
    with torch.no_grad():
        values = acqf(X.unsqueeze(-2)).detach().reshape(-1)
    return values.cpu().numpy().astype(float)


def _evaluate_grid(
    session: Any,
    context: _CompositionContext,
    *,
    fractions: np.ndarray,
    base_valid: np.ndarray,
    ordinary_axis_values: dict[str, np.ndarray],
    fixed_values: dict[str, Any],
    target: str,
    show_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(base_valid, dtype=bool) & _composition_validity(
        session,
        context,
        fractions,
    )
    values = np.full(fractions.shape[0], np.nan, dtype=float)
    std = np.full(fractions.shape[0], np.nan, dtype=float)
    if not valid.any():
        return values, std
    valid_source = _source_frame(
        session,
        context,
        fractions=fractions[valid],
        ordinary_axis_values={
            name: np.asarray(data).reshape(-1)[valid]
            for name, data in ordinary_axis_values.items()
        },
        fixed_values=fixed_values,
    )
    if show_type == "acqf":
        values[valid] = _acquisition_values(session, valid_source)
    else:
        mean, predicted_std = _predict_values(session, valid_source, target)
        values[valid] = mean
        std[valid] = predicted_std
    return values, std


def _axis_bounds(
    session: Any,
    context: _CompositionContext,
    feature: str,
) -> tuple[float, float]:
    element = _element_for_feature(context, feature)
    if element is not None:
        return _fraction_bounds(context, element)
    if feature not in session.data:
        raise ValueError(f"Unknown visualization feature {feature!r}.")
    series = session.data[feature].dropna().astype(float)
    lower = float(series.min())
    upper = float(series.max())
    if lower == upper:
        padding = 0.5 if lower == 0.0 else max(abs(lower) * 0.05, 1e-9)
        return lower - padding, upper + padding
    return lower, upper


def _display_training_frame(session: Any, context: _CompositionContext) -> Any:
    return _observed_composition_frame(session)


def _candidate_dataframe(session: Any) -> Any:
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


def _request_mode(request: dict[str, Any]) -> tuple[str, str | None]:
    fixed_values = dict(request.get("fixed_values") or {})
    mode = str(fixed_values.pop(_MODE_KEY, "proportional"))
    if mode not in {"proportional", "balance"}:
        mode = "proportional"
    balance = fixed_values.pop(_BALANCE_KEY, None)
    return mode, str(balance) if balance not in {None, ""} else None


def _composition_axes(
    context: _CompositionContext,
    features: list[str],
) -> list[str]:
    return [
        feature
        for feature in features
        if _element_for_feature(context, feature) is not None
    ]


def _build_composition_visualization(
    session: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    from .target_results import _figure_payload

    context = _composition_context(session)
    if context is None:
        raise ValueError("A fitted single-composition model is required.")
    kind = str(request.get("kind", "1d")).lower()
    target = str(request.get("target") or session.target_columns[0])
    if target not in session.target_columns:
        raise ValueError(f"target must be one of {session.target_columns!r}.")
    features = [str(value) for value in list(request.get("features") or [])]
    expected = {"1d": 1, "2d": 2, "ternary": 3}.get(kind)
    if expected is None or len(features) != expected or len(set(features)) != expected:
        raise ValueError(f"{kind} requires {expected or 0} different feature variables.")
    composition_axes = _composition_axes(context, features)
    if not composition_axes:
        raise ValueError("Select at least one element-ratio axis for composition plotting.")

    fixed_values = dict(request.get("fixed_values") or {})
    mode, balance_element = _request_mode(request)
    fixed_values.pop(_MODE_KEY, None)
    fixed_values.pop(_BALANCE_KEY, None)
    baseline = _baseline_fractions(session, context, fixed_values)
    show_type = str(request.get("show_type") or "pred")
    n = max(10, min(int(request.get("n") or 50), 150))
    observed = _display_training_frame(session, context)
    observed_targets = session.data.loc[:, session.target_columns]
    candidates = _candidate_dataframe(session)

    if kind == "1d":
        from bochan.visualization import show_1dplot_with_pred

        feature = features[0]
        lower, upper = _axis_bounds(session, context, feature)
        grid = np.linspace(lower, upper, n)
        fraction_axes = (
            {feature: grid} if feature in composition_axes else {}
        )
        ordinary_axes = (
            {} if feature in composition_axes else {feature: grid}
        )
        fractions, base_valid = _resolve_fraction_matrix(
            context,
            baseline=baseline,
            axis_values=fraction_axes or {
                context.fraction_features[0]: np.full(n, baseline[0])
            },
            mode=mode,
            balance_element=balance_element,
        )
        values, std = _evaluate_grid(
            session,
            context,
            fractions=fractions,
            base_valid=base_valid,
            ordinary_axis_values=ordinary_axes,
            fixed_values=fixed_values,
            target=target,
            show_type="pred",
        )
        import pandas as pd

        figure = show_1dplot_with_pred(
            feature,
            target,
            (pd.DataFrame({target: values}), pd.DataFrame({target: std}), grid),
            observed,
            observed_targets,
            candidates,
        )
        figure.update_xaxes(title_text=_feature_label(context, feature))
        return _figure_payload(
            figure,
            figure_id=f"composition-1d-{feature}-{target}",
            title=f"{_feature_label(context, feature)} → {target}",
            description="組成比を内部表現へ変換し、既存の1次元Plotly関数で表示します。",
        )

    if kind == "2d":
        from bochan.visualization import show_scatter_with_acqf

        feature_x, feature_y = features
        x_lower, x_upper = _axis_bounds(session, context, feature_x)
        y_lower, y_upper = _axis_bounds(session, context, feature_y)
        x_grid = np.linspace(x_lower, x_upper, min(n, 50))
        y_grid = np.linspace(y_lower, y_upper, min(n, 50))
        mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
        axis_map = {
            feature_x: mesh_x.reshape(-1),
            feature_y: mesh_y.reshape(-1),
        }
        fraction_axes = {
            name: values for name, values in axis_map.items() if name in composition_axes
        }
        ordinary_axes = {
            name: values for name, values in axis_map.items() if name not in composition_axes
        }
        fractions, base_valid = _resolve_fraction_matrix(
            context,
            baseline=baseline,
            axis_values=fraction_axes,
            mode=mode,
            balance_element=balance_element,
        )
        values, _std = _evaluate_grid(
            session,
            context,
            fractions=fractions,
            base_valid=base_valid,
            ordinary_axis_values=ordinary_axes,
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
        figure.update_xaxes(title_text=_feature_label(context, feature_x))
        figure.update_yaxes(title_text=_feature_label(context, feature_y))
        return _figure_payload(
            figure,
            figure_id=f"composition-2d-{feature_x}-{feature_y}-{target}",
            title=(
                f"{_feature_label(context, feature_x)} × "
                f"{_feature_label(context, feature_y)} → {target}"
            ),
            description="組成比グリッドを内部表現へ変換し、既存の2次元Plotly関数で表示します。",
        )

    if len(context.elements) != 3 or set(features) != set(context.fraction_features):
        raise ValueError(
            "Composition ternary plotting currently requires exactly three candidate "
            "elements and all three element-ratio axes."
        )
    from bochan.visualization import show_triscatter_with_acqf

    divisions = max(9, min(n, 60)) - 1
    rows: list[tuple[float, float, float]] = []
    for first in range(divisions + 1):
        for second in range(divisions + 1 - first):
            a = first / divisions
            b = second / divisions
            rows.append((a, b, 1.0 - a - b))
    grid = np.asarray(rows, dtype=float)
    axis_map = {
        features[0]: grid[:, 0],
        features[1]: grid[:, 1],
        features[2]: grid[:, 2],
    }
    fractions, base_valid = _resolve_fraction_matrix(
        context,
        baseline=baseline,
        axis_values=axis_map,
        mode="proportional",
        balance_element=None,
    )
    values, _std = _evaluate_grid(
        session,
        context,
        fractions=fractions,
        base_valid=base_valid,
        ordinary_axis_values={},
        fixed_values=fixed_values,
        target=target,
        show_type=show_type,
    )
    finite = np.isfinite(values)
    if finite.sum() < 3:
        raise ValueError("Fewer than three valid ternary grid points satisfy the constraints.")
    import pandas as pd

    display_grid = pd.DataFrame(
        {
            features[0]: grid[finite, 0],
            features[1]: grid[finite, 1],
            features[2]: grid[finite, 2],
        }
    )
    figure = show_triscatter_with_acqf(
        features[0],
        features[1],
        features[2],
        target,
        (values[finite], display_grid),
        observed,
        observed_targets,
        candidates,
        show_type=show_type,
    )
    figure.update_layout(
        ternary={
            "aaxis": {"title": _feature_label(context, features[0])},
            "baxis": {"title": _feature_label(context, features[1])},
            "caxis": {"title": _feature_label(context, features[2])},
            "sum": 1,
        }
    )
    return _figure_payload(
        figure,
        figure_id=f"composition-ternary-{'-'.join(features)}-{target}",
        title=f"{' / '.join(_feature_label(context, value) for value in features)} → {target}",
        description="元素分率を三角座標として、既存の三角Plotly関数で表示します。",
    )


def install_composition_visualization() -> None:
    """Install composition-aware option and plotting wrappers before app import."""

    from . import visualization_sessions

    if getattr(visualization_sessions, "_composition_visualization_installed", False):
        return
    original_options = visualization_sessions.visualization_options
    original_build = visualization_sessions.build_visualization

    def options_adapter(session: Any) -> dict[str, Any]:
        return _extend_visualization_options(original_options(session), session)

    def build_adapter(run_id: str, request: dict[str, Any]) -> dict[str, Any]:
        session = visualization_sessions.get_visualization_session(run_id)
        context = _composition_context(session)
        features = [str(value) for value in list(request.get("features") or [])]
        if context is not None and any(
            _element_for_feature(context, feature) is not None for feature in features
        ):
            return _build_composition_visualization(session, request)
        return original_build(run_id, request)

    visualization_sessions.visualization_options = options_adapter
    visualization_sessions.build_visualization = build_adapter
    visualization_sessions._composition_visualization_installed = True


__all__ = ["install_composition_visualization"]
