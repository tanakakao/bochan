"""Ternary slices for Web composition models with three or more elements."""

from __future__ import annotations

from typing import Any

import numpy as np

_INSTALLED = False


def _ternary_slice_grid(sum_value: float, divisions: int) -> np.ndarray:
    """Return a barycentric grid whose selected fractions sum to ``sum_value``."""

    total = float(sum_value)
    if not np.isfinite(total) or total <= 0.0 or total > 1.0:
        raise ValueError("Ternary composition sum_value must be greater than 0 and at most 1.")
    count = max(2, int(divisions))
    rows: list[tuple[float, float, float]] = []
    for first in range(count + 1):
        for second in range(count + 1 - first):
            a = first / count
            b = second / count
            rows.append((a * total, b * total, (1.0 - a - b) * total))
    return np.asarray(rows, dtype=float)


def _ternary_sum_value(
    context: Any,
    request: dict[str, Any],
    baseline: np.ndarray,
    features: list[str],
) -> float:
    """Resolve the selected three-element total for one ternary cross-section."""

    from . import composition_visualization as composition

    if len(context.elements) == 3:
        return 1.0
    selected_indices = [
        context.elements.index(composition._element_for_feature(context, feature))
        for feature in features
    ]
    default = float(np.asarray(baseline, dtype=float)[selected_indices].sum())
    raw = request.get("sum_value")
    value = default if raw is None else float(raw)
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(
            "For four or more candidate elements, ternary sum_value must be "
            "greater than 0 and at most 1."
        )
    return value


def _extend_multielement_ternary_options(
    options: dict[str, Any],
    session: Any,
) -> dict[str, Any]:
    """Register an initial three-element ternary slice for larger compositions."""

    from . import composition_visualization as composition

    context = composition._composition_context(session)
    if context is None or len(context.elements) < 3:
        return options

    result = dict(options)
    groups = list(result.get("ternary_groups") or ())
    features = list(context.fraction_features[:3])
    if not any(list(group.get("features") or ()) == features for group in groups):
        composition_payload = dict(result.get("composition") or {})
        controls = {
            str(item.get("name")): float(item.get("default", 0.0))
            for item in list(composition_payload.get("features") or ())
        }
        sum_value = 1.0 if len(context.elements) == 3 else sum(
            controls.get(feature, 0.0) for feature in features
        )
        sum_value = float(np.clip(sum_value, 1e-6, 1.0))
        groups.insert(0, {"features": features, "sum_value": sum_value})
    result["ternary_groups"] = groups
    return result


def _slice_frame(
    frame: Any,
    features: list[str],
    sum_value: float,
    tolerance: float,
) -> Any:
    """Keep observed or candidate rows close to the displayed ternary section."""

    if frame is None or any(feature not in frame.columns for feature in features):
        return frame
    values = frame.loc[:, features].apply(lambda column: column.astype(float))
    totals = values.sum(axis=1).to_numpy(dtype=float)
    mask = np.isfinite(totals) & np.isclose(
        totals,
        float(sum_value),
        atol=float(tolerance),
        rtol=0.0,
    )
    return frame.loc[mask].copy()


def _build_multielement_ternary_visualization(
    session: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Build a ternary cross-section while closing all unplotted elements."""

    import pandas as pd

    from bochan.visualization import show_triscatter_with_acqf

    from . import composition_visualization as composition
    from .target_results import _figure_payload

    context = composition._composition_context(session)
    if context is None:
        raise ValueError("A fitted single-composition model is required.")
    target = str(request.get("target") or session.target_columns[0])
    if target not in session.target_columns:
        raise ValueError(f"target must be one of {session.target_columns!r}.")

    features = [str(value) for value in list(request.get("features") or ())]
    composition_axes = composition._composition_axes(context, features)
    if (
        len(features) != 3
        or len(set(features)) != 3
        or len(composition_axes) != 3
    ):
        raise ValueError(
            "Composition ternary plotting requires three different element-ratio axes."
        )

    fixed_values = dict(request.get("fixed_values") or {})
    mode, balance_element = composition._request_mode(request)
    fixed_values.pop(composition._MODE_KEY, None)
    fixed_values.pop(composition._BALANCE_KEY, None)
    baseline = composition._baseline_fractions(session, context, fixed_values)
    sum_value = _ternary_sum_value(context, request, baseline, features)
    show_type = str(request.get("show_type") or "pred")
    n = max(10, min(int(request.get("n") or 50), 150))
    divisions = max(9, min(n, 60)) - 1
    grid = _ternary_slice_grid(sum_value, divisions)
    axis_map = {
        features[0]: grid[:, 0],
        features[1]: grid[:, 1],
        features[2]: grid[:, 2],
    }
    fractions, base_valid = composition._resolve_fraction_matrix(
        context,
        baseline=baseline,
        axis_values=axis_map,
        mode=mode,
        balance_element=balance_element,
    )
    values, _std = composition._evaluate_grid(
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

    observed = composition._display_training_frame(session, context)
    observed_targets = session.data.loc[:, session.target_columns]
    candidates = composition._candidate_dataframe(session)
    tolerance = max(0.02, sum_value / max(divisions, 1))
    displayed_observed = _slice_frame(observed, features, sum_value, tolerance)
    displayed_targets = (
        observed_targets.loc[displayed_observed.index]
        if displayed_observed is not None
        else observed_targets
    )
    displayed_candidates = _slice_frame(candidates, features, sum_value, tolerance)
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
        displayed_observed,
        displayed_targets,
        displayed_candidates,
        show_type=show_type,
    )
    figure.update_layout(
        ternary={
            "aaxis": {"title": composition._feature_label(context, features[0])},
            "baxis": {"title": composition._feature_label(context, features[1])},
            "caxis": {"title": composition._feature_label(context, features[2])},
            "sum": sum_value,
        }
    )
    if len(context.elements) == 3:
        description = "3元素の組成比を三角座標として表示します。"
    else:
        description = (
            f"選択した3元素の合計比率を {sum_value:.6g} に固定した断面です。"
            "残りの元素は指定した組成変化ルールで合計1になるよう配分します。"
        )
    return _figure_payload(
        figure,
        figure_id=f"composition-ternary-{'-'.join(features)}-{target}",
        title=(
            f"{' / '.join(composition._feature_label(context, value) for value in features)}"
            f" → {target}"
        ),
        description=description,
    )


def install_composition_multielement_ternary() -> None:
    """Install outer Web option and plotting adapters for ternary slices."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import composition_visualization as composition
    from . import visualization_sessions
    from .composition_visualization_compat import _unavailable_payload

    current_options = visualization_sessions.visualization_options
    current_build = visualization_sessions.build_visualization

    def options_adapter(session: Any) -> dict[str, Any]:
        return _extend_multielement_ternary_options(current_options(session), session)

    def build_adapter(run_id: str, request: dict[str, Any]) -> dict[str, Any]:
        session = visualization_sessions.get_visualization_session(run_id)
        context = composition._composition_context(session)
        if context is None or str(request.get("kind", "1d")).lower() != "ternary":
            return current_build(run_id, request)

        features = [str(value) for value in list(request.get("features") or ())]
        composition_axes = composition._composition_axes(context, features)
        if not composition_axes:
            return current_build(run_id, request)
        if len(features) == 3 and len(composition_axes) == 3:
            return _build_multielement_ternary_visualization(session, request)
        return _unavailable_payload(
            "ternary",
            "組成の三角図では、異なる3つの元素比率を軸として選択してください。",
        )

    visualization_sessions.visualization_options = options_adapter
    visualization_sessions.build_visualization = build_adapter
    visualization_sessions._composition_multielement_ternary = True
    _INSTALLED = True


__all__ = [
    "_build_multielement_ternary_visualization",
    "_extend_multielement_ternary_options",
    "_ternary_slice_grid",
    "_ternary_sum_value",
    "install_composition_multielement_ternary",
]
