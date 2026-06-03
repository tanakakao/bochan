"""FastAPI dependencies and in-memory state for bochan serving."""

from __future__ import annotations

from threading import Lock
from uuid import uuid4

from bochan.api import BayesianOptimizer


class InMemoryOptimizerStore:
    """Thread-safe in-memory store for fitted BayesianOptimizer instances.

    This store is intentionally simple. It is suitable for local development,
    demos, and tests. For production, provide a different dependency that
    persists metadata and model artifacts outside the FastAPI process.
    """

    def __init__(self) -> None:
        self._items: dict[str, BayesianOptimizer] = {}
        self._lock = Lock()

    def add(self, optimizer: BayesianOptimizer) -> str:
        model_id = uuid4().hex
        with self._lock:
            self._items[model_id] = optimizer
        return model_id

    def get(self, model_id: str) -> BayesianOptimizer:
        with self._lock:
            try:
                return self._items[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model_id: {model_id}") from exc

    def delete(self, model_id: str) -> None:
        with self._lock:
            try:
                del self._items[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model_id: {model_id}") from exc

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._items)


_DEFAULT_STORE = InMemoryOptimizerStore()


def get_optimizer_store() -> InMemoryOptimizerStore:
    """Return the default optimizer store used by FastAPI dependencies."""

    return _DEFAULT_STORE
