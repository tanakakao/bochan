"""Request-local progress hooks for optional application telemetry.

The core API never requires a progress callback. Applications may activate one
around a synchronous operation to observe events without changing modeling
semantics or serializing callback objects into configuration dataclasses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

ProgressCallback = Callable[[str, Mapping[str, Any]], None]

_CALLBACK: ContextVar[ProgressCallback | None] = ContextVar(
    "bochan_execution_progress_callback",
    default=None,
)


def emit_progress(event: str, **payload: Any) -> None:
    """Emit one supplemental progress event when a callback is active.

    Progress reporting must never make a model fit fail, so callback exceptions
    are deliberately isolated from the modeling path.
    """

    callback = _CALLBACK.get()
    if callback is None:
        return
    try:
        callback(str(event), dict(payload))
    except Exception:
        return


@contextmanager
def progress_reporting(callback: ProgressCallback | None) -> Iterator[None]:
    """Activate ``callback`` only for the current context/request."""

    token = _CALLBACK.set(callback)
    try:
        yield
    finally:
        _CALLBACK.reset(token)


__all__ = ["ProgressCallback", "emit_progress", "progress_reporting"]
