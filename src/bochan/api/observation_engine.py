"""Observation compatibility helpers for the canonical optimizer.

The public optimizer is defined only in :mod:`bochan.api.optimizer`.
Observation-specific model construction lives in :mod:`bochan.api.observation_service`.
Compatibility attributes resolve to the same canonical implementations without
creating another optimizer subclass or installing runtime patches.
"""

from __future__ import annotations

from typing import Any

from .observation_service import build_objective_bundle

_build_partial_objective_bundle = build_objective_bundle


def __getattr__(name: str) -> Any:
    if name == "BayesianOptimizer":
        from .optimizer import BayesianOptimizer

        return BayesianOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_objective_bundle"]
