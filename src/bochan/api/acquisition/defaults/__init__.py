"""Automatic acquisition defaults for the high-level API."""

from typing import Any

from .resolver import (
    resolve_acquisition_data_context,
    resolve_acquisition_defaults,
    resolve_llm_selected_model_config,
)
from .resolver import resolve_multi_output_model_config as _resolve_multi_output_model_config

_MACE_WIDE_MODEL_TYPES = frozenset({"macemultitask", "macemultitaskdkl"})


def _normalize_model_type(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_multi_output_model_config(model_config: Any, train_Y: Any) -> Any:
    """Preserve correlated MACE wide targets before generic output splitting."""

    if _normalize_model_type(model_config.model_type) in _MACE_WIDE_MODEL_TYPES:
        return model_config
    return _resolve_multi_output_model_config(model_config, train_Y)


__all__ = [
    "resolve_acquisition_data_context",
    "resolve_acquisition_defaults",
    "resolve_llm_selected_model_config",
    "resolve_multi_output_model_config",
]
