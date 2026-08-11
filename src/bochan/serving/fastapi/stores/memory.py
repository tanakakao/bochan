"""In-memory store for canonical Bayesian optimizers."""

from __future__ import annotations

from bochan.api import BayesianOptimizer

from .object import InMemoryObjectStore


class InMemoryOptimizerStore(InMemoryObjectStore[BayesianOptimizer]):
    """Thread-safe process-local store for fitted tensor optimizers."""

    def __init__(self) -> None:
        super().__init__(id_name="model_id")


__all__ = ["InMemoryOptimizerStore"]
