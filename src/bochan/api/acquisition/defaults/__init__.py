"""Automatic acquisition defaults for the high-level API."""

from .observations import resolve_observation_aware_baselines
from .resolver import (
    resolve_acquisition_data_context as _resolve_acquisition_data_context,
    resolve_acquisition_defaults as _resolve_acquisition_defaults,
    resolve_llm_selected_model_config,
    resolve_multi_output_model_config,
)


def resolve_acquisition_defaults(bundle, config, context):
    """Resolve observation semantics before acquisition-specific defaults."""

    context = resolve_observation_aware_baselines(bundle, config, context)
    return _resolve_acquisition_defaults(bundle, config, context)


def resolve_acquisition_data_context(bundle, config, context):
    """Return data context with observation-aware baseline semantics applied."""

    context = resolve_observation_aware_baselines(bundle, config, context)
    return _resolve_acquisition_data_context(bundle, config, context)


__all__ = [
    "resolve_acquisition_data_context",
    "resolve_acquisition_defaults",
    "resolve_llm_selected_model_config",
    "resolve_multi_output_model_config",
    "resolve_observation_aware_baselines",
]
