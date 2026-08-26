"""Automatic acquisition defaults for the high-level API."""

from __future__ import annotations

from typing import Any

from .resolver import (
    resolve_acquisition_data_context,
    resolve_acquisition_defaults,
    resolve_llm_selected_model_config,
    resolve_multi_output_model_config as _resolve_multi_output_model_config,
)

_ALIGNN_WIDE_MULTITASK_MODEL_TYPES = frozenset(
    {
        "alignnmultitask",
        "alignnmultitaskdkl",
    }
)


def _normalize_model_name(value: Any) -> str:
    """Normalize public model names for routing comparisons."""

    return "".join(character for character in str(value).lower() if character.isalnum())


def resolve_multi_output_model_config(model_config: Any, train_Y: Any) -> Any:
    """Keep correlated ALIGNN multitask targets wide in one shared model.

    ALIGNN correlated multitask models consume ``train_Y`` with shape ``[n, m]``
    directly. They therefore must bypass the automatic independent-output
    ``MultiOutputConfig`` wrapper in the same way as the canonical multitask,
    Kronecker, multi-fidelity, and CrabNet correlated multitask families.
    """

    if _normalize_model_name(model_config.model_type) in _ALIGNN_WIDE_MULTITASK_MODEL_TYPES:
        return model_config
    return _resolve_multi_output_model_config(model_config, train_Y)


__all__ = [
    "resolve_acquisition_data_context",
    "resolve_acquisition_defaults",
    "resolve_llm_selected_model_config",
    "resolve_multi_output_model_config",
]
