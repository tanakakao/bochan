"""Source-level Level-Set Estimation settings for the Web workbench."""

from __future__ import annotations

import math
from typing import Any

from .target_roles import level_set_thresholds

_WEB_LEVEL_SET_PARAMETER_KEY = "web_level_set_parameter"


def level_set_output_weights(
    *,
    target_columns: list[str],
    target_settings: list[dict[str, Any]],
    objective_targets: list[str],
) -> list[float]:
    """Return relative Web LSE weights aligned with all modeled outputs."""

    settings_by_target = {
        str(setting["target"]): setting for setting in target_settings
    }
    selected = set(objective_targets)
    weights: list[float] = []
    for target in target_columns:
        if target not in selected:
            weights.append(0.0)
            continue
        setting = settings_by_target.get(target)
        if setting is None:
            raise ValueError(f"Missing target setting for level-set target: {target}")
        try:
            weight = float(setting.get("level_set_weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{target}: level_set_weight must be a non-negative finite number."
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{target}: level_set_weight must be a non-negative finite number."
            )
        weights.append(weight)

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one optimized level-set target must have a positive weight."
        )
    return weights


def _configure_acquisition_parameter(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
) -> None:
    raw_value = acqf_kwargs.pop(_WEB_LEVEL_SET_PARAMETER_KEY, None)
    if raw_value is None:
        return
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Web LSE acquisition parameter must be numeric.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Web LSE acquisition parameter must be non-negative and finite.")

    if acq_key == "straddle":
        acqf_kwargs.setdefault("beta", value)
        return
    if acq_key == "boundaryvariance":
        if value <= 0.0:
            raise ValueError("Boundary Variance tau must be greater than zero.")
        acqf_kwargs.setdefault("tau", value)
        return
    if acq_key == "icu":
        # Zero is the Web sentinel for the class default: bandwidth = posterior std.
        if value > 0.0:
            acqf_kwargs.setdefault("bandwidth", value)
        return
    raise ValueError(f"Unsupported Web level-set acquisition: {acq_key!r}.")


def _risk_score_objective(
    *,
    multi_output: bool,
    n_w: int,
    risk_type: str,
    alpha: float,
) -> Any | None:
    normalized = str(risk_type).lower()
    if normalized in {"", "none"}:
        return None
    if normalized not in {"var", "cvar"}:
        raise ValueError("LSE input-perturbation risk_type must be none, var, or cvar.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("LSE input-perturbation risk alpha must be in (0, 1].")

    from bochan.acquisition.regression.levelset_estimation import (
        MultiOutputRegressionLevelSetScoreObjective,
        RegressionLevelSetScoreObjective,
    )

    objective_cls = (
        MultiOutputRegressionLevelSetScoreObjective
        if multi_output
        else RegressionLevelSetScoreObjective
    )
    return objective_cls(
        n_w=n_w,
        risk_type=normalized,
        alpha=float(alpha),
        maximize=True,
    )


def configure_level_set_acqf_kwargs(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
    train_x: Any,
    target_columns: list[str],
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
    objective_targets: list[str],
    input_perturbation: bool,
    n_w: int,
    risk_type: str = "none",
    risk_alpha: float = 0.2,
) -> dict[str, Any]:
    """Attach Web LSE thresholds, weights, duplicate references, and risk."""

    if acq_key not in {"straddle", "boundaryvariance", "icu"}:
        raise ValueError(f"Unsupported Web level-set acquisition: {acq_key!r}.")

    thresholds = level_set_thresholds(
        target_columns=target_columns,
        target_metadata=target_metadata,
        objective_targets=objective_targets,
    )
    acqf_kwargs.setdefault("thresholds", thresholds)
    acqf_kwargs.setdefault(
        "output_weights",
        level_set_output_weights(
            target_columns=target_columns,
            target_settings=target_settings,
            objective_targets=objective_targets,
        ),
    )
    acqf_kwargs.setdefault("output_reduction", "weighted_mean")
    acqf_kwargs.setdefault("X_observed", train_x)
    _configure_acquisition_parameter(acqf_kwargs, acq_key=acq_key)

    if input_perturbation:
        n_w = int(n_w)
        if n_w <= 0:
            raise ValueError("LSE InputPerturbation n_w must be positive.")
        acqf_kwargs.setdefault("n_w", n_w)
        objective = _risk_score_objective(
            multi_output=len(target_columns) > 1,
            n_w=n_w,
            risk_type=risk_type,
            alpha=risk_alpha,
        )
        if objective is not None:
            acqf_kwargs.setdefault("objective", objective)
    elif str(risk_type).lower() not in {"", "none"}:
        raise ValueError("LSE VaR/CVaR requires input_perturbation=true.")

    return acqf_kwargs


__all__ = [
    "configure_level_set_acqf_kwargs",
    "level_set_output_weights",
]
