"""Store protocols for FastAPI-managed optimizers."""

from __future__ import annotations

from typing import Protocol

from bochan.api import BayesianOptimizer


class OptimizerStore(Protocol):
    """Protocol for process-local or persistent optimizer registries.

    Implementations manage fitted :class:`bochan.api.BayesianOptimizer`
    instances by opaque model id.
    """

    def add(self, optimizer: BayesianOptimizer) -> str:
        """Register an optimizer and return its generated model id.

        Args:
            optimizer: Fitted or loadable optimizer instance.

        Returns:
            Generated model id.
        """
        ...

    def get(self, model_id: str) -> BayesianOptimizer:
        """Return an optimizer by model id.

        Args:
            model_id: Model id returned by :meth:`add`.

        Returns:
            Registered optimizer.

        Raises:
            KeyError: If ``model_id`` is unknown.
        """
        ...

    def delete(self, model_id: str) -> None:
        """Delete an optimizer by model id.

        Args:
            model_id: Model id to delete.

        Raises:
            KeyError: If ``model_id`` is unknown.
        """
        ...

    def list_ids(self) -> list[str]:
        """Return registered model ids sorted for stable API responses."""
        ...
