"""Store contracts for FastAPI-managed tabular optimizer adapters."""

from __future__ import annotations

from typing import Protocol

from bochan.tabular import TabularBayesianOptimizer

from .object import InMemoryObjectStore, ObjectStore


class TabularOptimizerStore(ObjectStore[TabularBayesianOptimizer], Protocol):
    """Typed store contract for tabular optimizer adapters."""


class InMemoryTabularOptimizerStore(
    InMemoryObjectStore[TabularBayesianOptimizer]
):
    """Thread-safe process-local store for tabular optimizer adapters."""

    def __init__(self) -> None:
        super().__init__(id_name="tabular model_id")


__all__ = ["InMemoryTabularOptimizerStore", "TabularOptimizerStore"]
