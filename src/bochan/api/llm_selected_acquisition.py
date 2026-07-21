"""Execution-time LLM resolution for ``AcquisitionConfig``.

``BayesianOptimizer.suggest_acquisition()`` remains the review-first API. This
module adds the declarative counterpart: ``AcquisitionConfig(name="llm_selected")``
is resolved to a concrete acquisition immediately before acquisition construction
or candidate generation.
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


def install_llm_selected_acquisition_api(
    optimizer_cls: type[Any] | None = None,
) -> None:
    """Install execution-time acquisition selection on ``BayesianOptimizer``."""

    if optimizer_cls is None:
        from .engine_defaults import BayesianOptimizer

        optimizer_cls = BayesianOptimizer

    if not hasattr(optimizer_cls, "suggest_acquisition"):
        from .llm_suggestion import install_bayesian_optimizer_llm_api

        install_bayesian_optimizer_llm_api(optimizer_cls)

    if getattr(
        optimizer_cls,
        "_bochan_llm_selected_acquisition_api_installed",
        False,
    ):
        return

    original_acquisition = optimizer_cls.acquisition
    original_candidate = optimizer_cls.candidate

    def resolve_llm_selected_acquisition(
        self: Any,
        config: AcquisitionConfig,
    ) -> AcquisitionConfig:
        """Resolve a selector config and cache the concrete result."""

        if not is_llm_selected_acquisition(config):
            return config
        if getattr(self, "llm_settings", None) is None:
            raise ValueError(
                "AcquisitionConfig(name='llm_selected') requires llm_settings "
                "on BayesianOptimizer or configure_llm(...)."
            )
        if getattr(self, "_llm_refit_required", False):
            raise RuntimeError(
                "The LLM changed model or fit settings after fitting. Call fit() "
                "or refit() before resolving an acquisition."
            )

        check_fitted = getattr(self, "_check_fitted", None)
        if callable(check_fitted):
            check_fitted()

        suggestion = self.suggest_acquisition(
            prompt=_selection_prompt(config),
            apply=False,
        )
        resolved = _merge_requested_config(config, suggestion.acq_config)
        self.acq_config = resolved
        self.last_acquisition_suggestion = suggestion
        return resolved

    def acquisition_with_llm_selected(
        self: Any,
        acq_config: AcquisitionConfig | None = None,
        *,
        data_context: Any | None = None,
    ) -> Any:
        resolved = acq_config
        if resolved is None:
            resolved = getattr(self, "acq_config", None)
        if resolved is not None and is_llm_selected_acquisition(resolved):
            resolved = resolve_llm_selected_acquisition(self, resolved)
        return original_acquisition(
            self,
            resolved,
            data_context=data_context,
        )

    def candidate_with_llm_selected(
        self: Any,
        acq_config: AcquisitionConfig | None = None,
        opt_config: Any | None = None,
        *,
        data_context: Any | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> Any:
        resolved = acq_config
        if resolved is None:
            resolved = getattr(self, "acq_config", None)
        if resolved is not None and is_llm_selected_acquisition(resolved):
            resolved = resolve_llm_selected_acquisition(self, resolved)
        return original_candidate(
            self,
            resolved,
            opt_config,
            data_context=data_context,
            bounds=bounds,
            return_result=return_result,
        )

    optimizer_cls._resolve_llm_selected_acquisition = (
        resolve_llm_selected_acquisition
    )
    optimizer_cls.acquisition = acquisition_with_llm_selected
    optimizer_cls.candidate = candidate_with_llm_selected
    optimizer_cls._bochan_llm_selected_acquisition_api_installed = True


__all__ = [
    "install_llm_selected_acquisition_api",
    "is_llm_selected_acquisition",
]
