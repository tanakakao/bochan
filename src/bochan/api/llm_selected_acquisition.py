"""Execution-time LLM resolution for ``AcquisitionConfig``.

``AcquisitionConfig(name="llm_selected")`` is resolved explicitly by the
canonical acquisition service.  This module contains pure selection helpers and
does not replace ``BayesianOptimizer.acquisition`` or ``candidate`` at runtime.
"""

# ruff: noqa: I001

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from enum import Enum
from typing import Any

from .acquisition_config import AcquisitionConfig


_LLM_SELECTED_NAMES = {
    "llm",
    "llmselected",
    "llmacquisitionselect",
    "llmacquisitionselected",
    "llmplanned",
    "llmplanner",
}


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def is_llm_selected_acquisition(config: AcquisitionConfig) -> bool:
    """Return whether ``config`` requests execution-time LLM selection."""

    if config.acqf_cls is not None or config.acqf_factory is not None:
        return False
    return _normalize_name(config.name) in _LLM_SELECTED_NAMES


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _selection_prompt(config: AcquisitionConfig) -> str:
    requested = json.dumps(
        _to_jsonable(config),
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        "Resolve AcquisitionConfig(name='llm_selected') to one concrete "
        "acquisition_config for the fitted model and current optimization goal. "
        "Do not return llm_selected. Treat explicitly supplied objective_config, "
        "objective, constraints, sampler, objective_kwargs, and acqf_kwargs as "
        "authoritative unless they are structurally incompatible. Requested "
        f"configuration: {requested}"
    )


def _merge_requested_config(
    requested: AcquisitionConfig,
    suggested: AcquisitionConfig | None,
) -> AcquisitionConfig:
    """Overlay explicit user values on the LLM-selected configuration."""

    if suggested is None:
        raise ValueError(
            "LLM acquisition selector response must include 'acquisition_config'."
        )
    if is_llm_selected_acquisition(suggested):
        raise ValueError(
            "LLM acquisition selector must return a concrete acquisition name, "
            "not 'llm_selected'."
        )

    updates: dict[str, Any] = {
        "context_fields": requested.context_fields,
        "filter_kwargs_by_signature": requested.filter_kwargs_by_signature,
    }
    for name in (
        "objective",
        "objective_config",
        "objective_factory",
        "sampler",
    ):
        value = getattr(requested, name)
        if value is not None:
            updates[name] = value

    if requested.objective_kwargs:
        updates["objective_kwargs"] = {
            **dict(suggested.objective_kwargs or {}),
            **dict(requested.objective_kwargs),
        }

    suggested_acqf_kwargs = {
        key: value
        for key, value in dict(suggested.acqf_kwargs or {}).items()
        if key != "constraints"
    }
    requested_acqf_kwargs = {
        key: value
        for key, value in dict(requested.acqf_kwargs or {}).items()
        if key != "constraints"
    }
    merged_acqf_kwargs = {
        **suggested_acqf_kwargs,
        **requested_acqf_kwargs,
    }

    if requested.outcome_constraint_config is not None:
        updates["outcome_constraint_config"] = requested.outcome_constraint_config
        updates["constraints"] = None
    elif requested.constraints is not None:
        updates["constraints"] = requested.constraints
        updates["outcome_constraint_config"] = None
        merged_acqf_kwargs["constraints"] = requested.constraints
    elif suggested.constraints is not None:
        merged_acqf_kwargs["constraints"] = suggested.constraints

    updates["acqf_kwargs"] = merged_acqf_kwargs
    return replace(suggested, **updates)


def resolve_llm_selected_acquisition(
    optimizer: Any,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Resolve one selector configuration through the fitted optimizer."""

    if not is_llm_selected_acquisition(config):
        return config
    if getattr(optimizer, "llm_settings", None) is None:
        raise ValueError(
            "AcquisitionConfig(name='llm_selected') requires llm_settings "
            "on BayesianOptimizer or configure_llm(...)."
        )
    if getattr(optimizer, "_llm_refit_required", False):
        raise RuntimeError(
            "The LLM changed model or fit settings after fitting. Call fit() "
            "or refit() before resolving an acquisition."
        )

    optimizer._check_fitted()
    suggestion = optimizer.suggest_acquisition(
        prompt=_selection_prompt(config),
        apply=False,
    )
    resolved = _merge_requested_config(config, suggestion.acq_config)
    optimizer.acq_config = resolved
    optimizer.last_acquisition_suggestion = suggestion
    return resolved


def install_llm_selected_acquisition_api(
    optimizer_cls: type[Any] | None = None,
) -> None:
    """Deprecated no-op retained for import compatibility.

    LLM-selected acquisitions are resolved by
    :func:`resolve_llm_selected_acquisition` from the canonical acquisition
    service.  No optimizer methods are installed or replaced.
    """

    del optimizer_cls


__all__ = [
    "install_llm_selected_acquisition_api",
    "is_llm_selected_acquisition",
    "resolve_llm_selected_acquisition",
]
