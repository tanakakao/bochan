"""Store protocol for FastAPI-managed tensor optimizers."""

from __future__ import annotations

from typing import Protocol

from bochan.api import BayesianOptimizer

from .object import ObjectStore


class OptimizerStore(ObjectStore[BayesianOptimizer], Protocol):
    """Typed store contract for canonical :class:`BayesianOptimizer` objects."""


__all__ = ["OptimizerStore"]
