"""In-memory optimizer store implementation."""

from __future__ import annotations

from threading import Lock
from uuid import uuid4

from bochan.api import BayesianOptimizer


class InMemoryOptimizerStore:
    """Thread-safe in-memory store for fitted optimizers.

    This implementation is intended for tests, demos, and single-process local
    serving. Production deployments should replace the FastAPI dependency with a
    database, object store, or model registry backed implementation.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory optimizer registry."""
        self._items: dict[str, BayesianOptimizer] = {}
        self._lock = Lock()

    def add(self, optimizer: BayesianOptimizer) -> str:
        """Register an optimizer and return a new model id.

        Args:
            optimizer: Optimizer instance to store.

        Returns:
            Hexadecimal model id.
        """
        model_id = uuid4().hex
        with self._lock:
            self._items[model_id] = optimizer
        return model_id

    def get(self, model_id: str) -> BayesianOptimizer:
        """Return an optimizer by id.

        Args:
            model_id: Stored optimizer id.

        Returns:
            Stored optimizer.

        Raises:
            KeyError: If the id is unknown.
        """
        with self._lock:
            try:
                return self._items[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model_id: {model_id}") from exc

    def delete(self, model_id: str) -> None:
        """Delete an optimizer by id.

        Args:
            model_id: Stored optimizer id.

        Raises:
            KeyError: If the id is unknown.
        """
        with self._lock:
            try:
                del self._items[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model_id: {model_id}") from exc

    def list_ids(self) -> list[str]:
        """Return sorted model ids currently registered in memory."""
        with self._lock:
            return sorted(self._items)
