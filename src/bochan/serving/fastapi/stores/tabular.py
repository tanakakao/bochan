"""In-memory store for FastAPI-managed tabular optimizers."""

from __future__ import annotations

from threading import Lock
from typing import Protocol
from uuid import uuid4

from bochan.tabular import TabularBayesianOptimizer


class TabularOptimizerStore(Protocol):
    """Protocol for registries containing tabular optimizer instances."""

    def add(self, optimizer: TabularBayesianOptimizer) -> str:
        """Register an optimizer and return a generated model id."""
        ...

    def get(self, model_id: str) -> TabularBayesianOptimizer:
        """Return an optimizer by model id."""
        ...

    def delete(self, model_id: str) -> None:
        """Delete an optimizer by model id."""
        ...

    def list_ids(self) -> list[str]:
        """Return registered model ids in stable order."""
        ...


class InMemoryTabularOptimizerStore:
    """Thread-safe process-local store for tabular optimizers."""

    def __init__(self) -> None:
        self._items: dict[str, TabularBayesianOptimizer] = {}
        self._lock = Lock()

    def add(self, optimizer: TabularBayesianOptimizer) -> str:
        model_id = uuid4().hex
        with self._lock:
            self._items[model_id] = optimizer
        return model_id

    def get(self, model_id: str) -> TabularBayesianOptimizer:
        with self._lock:
            try:
                return self._items[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown tabular model_id: {model_id}") from exc

    def delete(self, model_id: str) -> None:
        with self._lock:
            try:
                del self._items[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown tabular model_id: {model_id}") from exc

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._items)


__all__ = ["InMemoryTabularOptimizerStore", "TabularOptimizerStore"]
