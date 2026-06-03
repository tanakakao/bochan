"""FastAPI application factory for bochan."""

from __future__ import annotations

from fastapi import FastAPI

from .routers import acquisitions, candidates, health, models, predictions


def create_app(*, title: str = "bochan API", version: str = "0.1.0") -> FastAPI:
    """Create a FastAPI app exposing the bochan high-level API.

    The app keeps model state in an in-memory store by default. This is useful
    for local development and prototyping. Production deployments should
    replace the store dependency with a database, object store, or model registry.
    """

    app = FastAPI(title=title, version=version)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predictions.router)
    app.include_router(candidates.router)
    app.include_router(acquisitions.router)
    return app


app = create_app()
