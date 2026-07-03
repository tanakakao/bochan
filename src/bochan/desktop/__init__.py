"""Desktop application helpers for bochan."""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create the desktop FastAPI app without importing it at package import time."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def run_desktop(*args: Any, **kwargs: Any) -> None:
    """Launch the desktop application using a lazy import."""

    from .app import run_desktop as _run_desktop

    _run_desktop(*args, **kwargs)


__all__ = ["create_app", "run_desktop"]
