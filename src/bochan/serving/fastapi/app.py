"""FastAPI application factory for bochan."""

from __future__ import annotations

from fastapi import FastAPI

from .router import create_api_router


def create_app(*, title: str = "bochan API", version: str = "0.1.0") -> FastAPI:
    """Create a FastAPI app exposing the versioned bochan API.

    Args:
        title: OpenAPI application title.
        version: OpenAPI application version.

    Returns:
        Configured FastAPI application with endpoints mounted under
        ``/api/v1``.
    """
    app = FastAPI(title=title, version=version)
    app.include_router(create_api_router(prefix="/api/v1"))
    return app


app = create_app()
