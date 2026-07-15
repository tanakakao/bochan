"""FastAPI dependency providers for bochan serving."""

from __future__ import annotations

from .stores import FileOptimizerStore, InMemoryOptimizerStore, OptimizerStore

_DEFAULT_STORE = InMemoryOptimizerStore()
_DEFAULT_FILE_STORE = FileOptimizerStore()


def get_optimizer_store() -> OptimizerStore:
    """Return the default optimizer store used by API endpoints.

    Returns:
        Process-local optimizer store. Tests and applications can override this
        dependency with FastAPI's ``dependency_overrides``.
    """
    return _DEFAULT_STORE


def get_file_optimizer_store() -> FileOptimizerStore:
    """Return the default file artifact store used by API endpoints.

    Returns:
        File artifact store rooted at ``BOCHAN_API_MODEL_DIR`` when configured.
    """
    return _DEFAULT_FILE_STORE


__all__ = [
    "FileOptimizerStore",
    "InMemoryOptimizerStore",
    "OptimizerStore",
    "get_file_optimizer_store",
    "get_optimizer_store",
]
