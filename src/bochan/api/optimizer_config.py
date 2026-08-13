"""Deprecated flat import facade for :mod:`bochan.api.config.optimize`."""

from __future__ import annotations

from .config.optimize import (
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)

__all__ = [
    "OptimizeConfig",
    "resolve_optimizer_from_cat_dims",
    "uses_mixed_fixed_features",
]
