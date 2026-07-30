"""Request-local risk settings for the Web input-perturbation workflow."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Iterator

from .prediction_shapes import normalize_prediction_rows

_WEB_RISK_TYPE_KEY = "web_risk_type"
_WEB_RISK_ALPHA_KEY = "web_risk_alpha"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_input_perturbation_risk",
    default=None,
)
_ORIGINAL_ACQUISITION_FAMILY: Any | None = None
_ORIGINAL_OBJECTIVE_RESOLVER: Any | None = None
_INSTALLED = False


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


def _request_settings(request: Any) -> dict[str, Any]:
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
    if enabled and family != "bayesian_optimization":
        raise ValueError(
            "VaR/CVaR input perturbation risk is currently available only for "
            "Bayesian optimization in the Web workbench."
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
    """Activate one request's Web-only input-perturbation risk settings."""

    state = _request_settings(request)
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_web_risk_report() -> dict[str, Any]:
    """Return the active request's normalized risk settings."""

    return dict(_STATE.get() or {})


def _risk_aware_objective_rows(value: Any, *, n_rows: int) -> Any:
    """Aggregate objective-space baseline predictions with the active risk rule."""

    state = _STATE.get() or {}
    risk_type = state.get("risk_type") if state.get("risk_enabled") else None
    return normalize_prediction_rows(
        value,
        n_rows=n_rows,
        risk_type=risk_type,
        alpha=float(state.get("risk_alpha", 0.2)),
    )


def _install_acquisition_kwargs_adapter(workflows_module: Any) -> None:
    global _ORIGINAL_ACQUISITION_FAMILY

    if _ORIGINAL_ACQUISITION_FAMILY is not None:
        return
    _ORIGINAL_ACQUISITION_FAMILY = workflows_module._acquisition_family

    def acquisition_family(acqf_kwargs: dict[str, Any]) -> str:
        acqf_kwargs.pop(_WEB_RISK_TYPE_KEY, None)
        acqf_kwargs.pop(_WEB_RISK_ALPHA_KEY, None)
        return _ORIGINAL_ACQUISITION_FAMILY(acqf_kwargs)

    workflows_module._acquisition_family = acquisition_family


def _install_objective_adapter() -> None:
    global _ORIGINAL_OBJECTIVE_RESOLVER

    if _ORIGINAL_OBJECTIVE_RESOLVER is not None:
        return

    import bochan.api.engine as engine

    _ORIGINAL_OBJECTIVE_RESOLVER = (
        engine._resolve_objective_config_n_w_from_input_transform
    )

    def resolve_objective_config(*, acq_config: Any, bundle: Any) -> Any:
        resolved = _ORIGINAL_OBJECTIVE_RESOLVER(
            acq_config=acq_config,
            bundle=bundle,
        )
        state = _STATE.get() or {}
        if not state.get("risk_enabled"):
            return resolved
        objective_config = resolved.objective_config
        if objective_config is None:
            raise RuntimeError(
                "The selected acquisition does not expose a risk-aware objective."
            )
        return replace(
            resolved,
            objective_config=replace(
                objective_config,
                risk_type=str(state["risk_type"]),
                alpha=float(state["risk_alpha"]),
            ),
        )

    engine._resolve_objective_config_n_w_from_input_transform = resolve_objective_config


def install_web_risk_adapters(workflows_module: Any) -> None:
    """Install request-local Web adapters without changing the public API schema."""

    global _INSTALLED

    if _INSTALLED:
        workflows_module._as_2d = _risk_aware_objective_rows
        return
    _install_acquisition_kwargs_adapter(workflows_module)
    _install_objective_adapter()
    workflows_module._as_2d = _risk_aware_objective_rows
    _INSTALLED = True


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
    "attach_web_risk_metadata",
    "current_web_risk_report",
    "install_web_risk_adapters",
    "web_risk_run",
]
