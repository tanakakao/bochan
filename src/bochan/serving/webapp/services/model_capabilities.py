"""Web-facing model capability checks shared across workflow entry points."""

from __future__ import annotations

from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_name(value: Any) -> str:
    return str(value or "").replace("-", "_").lower()


def validate_web_model_acquisition_compatibility(
    request: Any,
    target_settings: list[dict[str, Any]],
) -> None:
    """Reject model/acquisition combinations that are numerically valid but meaningless.

    TabPFN classification intentionally exposes the public ``predict_proba``
    distribution as one probability-posterior member. Its internal inference
    ensemble is not reinterpreted as independent epistemic model samples. BALD,
    however, measures disagreement across epistemic samples, so it collapses to
    zero for this bridge. Predictive entropy and probability/label variance remain
    meaningful and are therefore left available.
    """

    if _normalized_name(_field(request, "model_type")) != "tabpfn":
        return

    acquisition = _field(request, "acquisition")
    acquisition_name = _normalized_name(_field(acquisition, "name"))
    if acquisition_name != "bald":
        return

    has_classification_target = any(
        _normalized_name(setting.get("task_type")) == "classification"
        for setting in target_settings
    )
    if not has_classification_target:
        return

    raise ValueError(
        "TabPFN classification does not expose independent epistemic ensemble "
        "members for BALD. Use predictive_entropy, variance, or NIPV instead."
    )


__all__ = ["validate_web_model_acquisition_compatibility"]
