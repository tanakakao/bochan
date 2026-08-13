"""Lazy model and acquisition registries for the high-level API."""

from __future__ import annotations

from .acquisition import available_acqf_names, resolve_acqf_cls
from .model import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY, LazyModelRegistry

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
    "available_acqf_names",
    "resolve_acqf_cls",
]
