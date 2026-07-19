"""FastAPI dependency providers for bochan serving."""

from __future__ import annotations

from .stores import (
    FileOptimizerStore,
    InMemoryOptimizerStore,
    InMemoryStudyStore,
    InMemoryTabularOptimizerStore,
    OptimizerStore,
    StudyStore,
    TabularOptimizerStore,
)

_DEFAULT_STORE = InMemoryOptimizerStore()
_DEFAULT_TABULAR_STORE = InMemoryTabularOptimizerStore()
_DEFAULT_STUDY_STORE = InMemoryStudyStore()
_DEFAULT_FILE_STORE = FileOptimizerStore()


def get_optimizer_store() -> OptimizerStore:
    """Return the default tensor optimizer store used by API endpoints."""

    return _DEFAULT_STORE


def get_tabular_optimizer_store() -> TabularOptimizerStore:
    """Return the process-local store used by tabular API endpoints."""

    return _DEFAULT_TABULAR_STORE


def get_study_store() -> StudyStore:
    """Return the process-local store used by BochanStudy endpoints."""

    return _DEFAULT_STUDY_STORE


def get_file_optimizer_store() -> FileOptimizerStore:
    """Return the default file artifact store used by API endpoints."""

    return _DEFAULT_FILE_STORE


__all__ = [
    "FileOptimizerStore",
    "InMemoryOptimizerStore",
    "InMemoryStudyStore",
    "InMemoryTabularOptimizerStore",
    "OptimizerStore",
    "StudyStore",
    "TabularOptimizerStore",
    "get_file_optimizer_store",
    "get_optimizer_store",
    "get_study_store",
    "get_tabular_optimizer_store",
]
