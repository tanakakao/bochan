"""Public acquisition-registry entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ._acquisition_impl import (
    available_acqf_names as _available_acqf_names,
)
from ._acquisition_impl import resolve_acqf_cls as _resolve_acqf_cls


def resolve_acqf_cls(
    name: str,
    acquisition_registry: Mapping[str, Any] | None = None,
    *,
    task_type: str | None = None,
    model_type: str | None = None,
    multi_output: bool = False,
) -> type | Callable[..., Any]:
    """Resolve a canonical or contextual acquisition name."""
    return _resolve_acqf_cls(
        name,
        acquisition_registry,
        task_type=task_type,
        model_type=model_type,
        multi_output=multi_output,
    )


def available_acqf_names() -> list[str]:
    """Return canonical and contextual acquisition names."""
    return _available_acqf_names()


__all__ = ["available_acqf_names", "resolve_acqf_cls"]
