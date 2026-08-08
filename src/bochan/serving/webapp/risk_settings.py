"""Request-local risk settings for the Web input-perturbation workflow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from .prediction_shapes import normalize_prediction_rows

_WEB_RISK_TYPE_KEY = "web_risk_type"
_WEB_RISK_ALPHA_KEY = "web_risk_alpha"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_input_perturbation_risk",
    default=None,
)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def resolve_web_risk_settings(request: Any) -> dict[str, Any]:
    """Normalize Web risk markers without modifying API or workflow functions."""

    acquisition = _mapping(getattr(request, "acquisition", None))
    kwargs = _mapping(acquisition.get("acqf_kwargs"))
    risk_type = str(kwargs.get(_WEB_RISK_TYPE_KEY, "none")).lower()
    if risk_type not in {"none", "var", "cvar"}:
        raise ValueError("Input perturbation risk_type must be none, var, or cvar.")

    try:
        alpha = float(kwargs.get(_WEB_RISK_ALPHA_KEY, 0.2))
    except (TypeError, ValueError) as exc:
        raise ValueError("Input perturbation risk alpha must be numeric.") from exc
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Input perturbation risk alpha must be in (0, 1].")

    input_perturbation = bool(getattr(request, "input_perturbation", False))
    family = str(kwargs.get("web_family", "bayesian_optimization")).lower()
    enabled = input_perturbation and risk_type in {"var", "cvar"}
    if not input_perturbation and risk_type != "none":
        raise ValueError("VaR/CVaR requires input_perturbation=true.")
    if enabled and family not in {"bayesian_optimization", "level_set_estimation"}:
        raise ValueError(
            "VaR/CVaR input perturbation risk is available for Bayesian optimization "
            "or level-set estimation in the Web workbench."
        )

    return {
        "input_perturbation": input_perturbation,
        "risk_type": risk_type if input_perturbation else "none",
        "risk_alpha": alpha,
        "risk_enabled": enabled,
        "acquisition_family": family,
    }


@contextmanager
def web_risk_run(request: Any) -> Iterator[dict[str, Any]]:
    """Activate one request's Web input-perturbation risk metadata."""

    state = resolve_web_risk_settings(request)
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_web_risk_report() -> dict[str, Any]:
    """Return the active request's normalized risk settings."""

    return dict(_STATE.get() or {})


def apply_web_risk_to_objective_config(
    objective_config: Any,
    report: dict[str, Any],
) -> Any:
    """Return a BO ObjectiveConfig carrying explicit Web VaR/CVaR settings."""

    if objective_config is None or not report.get("risk_enabled"):
        return objective_config
    return replace(
        objective_config,
        risk_type=str(report["risk_type"]),
        alpha=float(report["risk_alpha"]),
    )


def normalize_web_prediction_rows(
    value: Any,
    *,
    n_rows: int,
    report: dict[str, Any],
) -> Any:
    """Aggregate InputPerturbation-expanded baseline values explicitly."""

    risk_type = str(report.get("risk_type")) if report.get("risk_enabled") else None
    return normalize_prediction_rows(
        value,
        n_rows=n_rows,
        risk_type=risk_type,
        alpha=float(report.get("risk_alpha", 0.2)),
    )


def attach_web_risk_metadata(
    result: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Attach effective risk settings to a Web result payload."""

    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "input_perturbation_risk_type": report.get("risk_type", "none"),
            "input_perturbation_risk_alpha": float(report.get("risk_alpha", 0.2)),
            "input_perturbation_risk_enabled": bool(report.get("risk_enabled")),
        }
    )
    result["metadata"] = metadata
    return metadata


__all__ = [
    "apply_web_risk_to_objective_config",
    "attach_web_risk_metadata",
    "current_web_risk_report",
    "normalize_web_prediction_rows",
    "resolve_web_risk_settings",
    "web_risk_run",
]
