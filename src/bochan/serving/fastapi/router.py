"""Shared API router factory for bochan FastAPI serving."""

from __future__ import annotations

from fastapi import APIRouter

from .routers import acquisitions, artifacts, candidates, health, models, predictions, suggestions


def create_api_router(*, prefix: str = "") -> APIRouter:
    """Create the common bochan API router.

    Args:
        prefix: Optional path prefix, for example ``/api/v1``.

    Returns:
        Router containing health, model lifecycle, prediction, candidate,
        acquisition, and artifact endpoints.
    """
    router = APIRouter(prefix=prefix)
    router.include_router(health.router)
    router.include_router(models.router)
    router.include_router(suggestions.router)
    router.include_router(predictions.router)
    router.include_router(candidates.router)
    router.include_router(acquisitions.router)
    router.include_router(artifacts.router)
    return router
