"""FastAPI serving integration for bochan.

The FastAPI layer is intentionally separated from :mod:`bochan.api`.
Import from this package only when the optional ``api`` dependencies are
installed.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
