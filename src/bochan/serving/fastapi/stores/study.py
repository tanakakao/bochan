"""Thread-safe in-memory store for FastAPI-managed BochanStudy instances."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Protocol, TypeVar
from uuid import uuid4

from bochan.api import BochanStudy

T = TypeVar("T")


class StudyStore(Protocol):
    """Protocol for registries containing mutable study instances."""

    def add(self, study: BochanStudy, *, study_id: str | None = None) -> str:
        """Register a study and return its identifier."""
        ...

    def get(self, study_id: str) -> BochanStudy:
        """Return a study by identifier."""
        ...

    def delete(self, study_id: str) -> None:
        """Delete a study by identifier."""
        ...

    def list_ids(self) -> list[str]:
        """Return registered identifiers in stable order."""
        ...

    def call(self, study_id: str, operation: Callable[[BochanStudy], T]) -> T:
        """Run one operation while holding the study-specific lock."""
        ...


class InMemoryStudyStore:
    """Process-local study store with per-study operation serialization.

    BochanStudy mutates trial state during ``ask()``, ``tell()``, and status
    updates. FastAPI executes synchronous endpoints in a thread pool, so each
    study receives its own re-entrant lock to avoid concurrent state races.
    """

    def __init__(self) -> None:
        self._items: dict[str, BochanStudy] = {}
        self._item_locks: dict[str, RLock] = {}
        self._lock = RLock()

    def add(self, study: BochanStudy, *, study_id: str | None = None) -> str:
        resolved_id = str(study_id or uuid4().hex)
        with self._lock:
            if resolved_id in self._items:
                raise KeyError(f"Study id already exists: {resolved_id}")
            self._items[resolved_id] = study
            self._item_locks[resolved_id] = RLock()
        return resolved_id

    def get(self, study_id: str) -> BochanStudy:
        with self._lock:
            try:
                return self._items[study_id]
            except KeyError as exc:
                raise KeyError(f"Unknown study_id: {study_id}") from exc

    @contextmanager
    def _locked_study(self, study_id: str) -> Iterator[BochanStudy]:
        with self._lock:
            try:
                study = self._items[study_id]
                item_lock = self._item_locks[study_id]
            except KeyError as exc:
                raise KeyError(f"Unknown study_id: {study_id}") from exc
        with item_lock:
            yield study

    def call(self, study_id: str, operation: Callable[[BochanStudy], T]) -> T:
        with self._locked_study(study_id) as study:
            return operation(study)

    def delete(self, study_id: str) -> None:
        with self._lock:
            try:
                item_lock = self._item_locks[study_id]
            except KeyError as exc:
                raise KeyError(f"Unknown study_id: {study_id}") from exc
            with item_lock:
                del self._items[study_id]
                del self._item_locks[study_id]

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._items)


__all__ = ["InMemoryStudyStore", "StudyStore"]
