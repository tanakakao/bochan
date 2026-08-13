"""Deprecated flat import facade for :mod:`bochan.api.registry.model`."""

from __future__ import annotations

from .registry.model import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY, LazyModelRegistry

__all__ = ["DEFAULT_MODEL_REGISTRY", "LazyModelRegistry", "MODEL_REGISTRY"]
