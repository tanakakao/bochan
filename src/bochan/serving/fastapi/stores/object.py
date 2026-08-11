"""Generic object-store contracts for FastAPI-managed runtime sessions."""

from __future__ import annotations

from threading import Lock
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

T = TypeVar("T")


class ObjectStore(Protocol[T]):
    """Minimal CRUD contract for process-local or persistent runtime objects."""

    def add(self, value: T) -> str: ...

    def get(self, object_id: str) -> T: ...

    def delete(self, object_id: str) -> None: ...

    def list_ids(self) -> list[str]: ...


class InMemoryObjectStore(Generic[T]):
    """Thread-safe in-memory implementation shared by serving object types."""

    def __init__(self, *, id_name: str = "object_id") -> None:
        self._items: dict[str, T] = {}
        self._lock = Lock()
        self._id_name = str(id_name)

    def add(self, value: T) -> str:
        object_id = uuid4().hex
        with self._lock:
            self._items[object_id] = value
        return object_id

    def get(self, object_id: str) -> T:
        with self._lock:
            try:
                return self._items[object_id]
            except KeyError as exc:
                raise KeyError(f"Unknown {self._id_name}: {object_id}") from exc

    def delete(self, object_id: str) -> None:
        with self._lock:
            try:
                del self._items[object_id]
            except KeyError as exc:
                raise KeyError(f"Unknown {self._id_name}: {object_id}") from exc

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._items)


__all__ = ["InMemoryObjectStore", "ObjectStore"]
