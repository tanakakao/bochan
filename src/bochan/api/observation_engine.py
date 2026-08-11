"""Observation model-building compatibility module.

The public optimizer now lives exclusively in :mod:`bochan.api.optimizer`.
Observation-specific model construction is implemented by
:mod:`bochan.api.observation_service`; this module no longer defines another
``BayesianOptimizer`` subclass.
"""

from .observation_service import (
    _build_partial_objective_bundle,
    build_objective_bundle,
)

__all__ = ["build_objective_bundle"]
