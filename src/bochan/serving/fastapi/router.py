"""Shared API router factory for bochan FastAPI serving."""

from __future__ import annotations

from fastapi import APIRouter

from .routers import (
    acquisitions,
    artifacts,
    candidates,
    health,
    models,
    predictions,
    studies,
    suggestions,
    tabular,
    tabular_artifacts,
)


def create_api_router(*, prefix: str = "") -> APIRouter:
    """Create the common bochan API router."""

    router = APIRouter(prefix=prefix)
    router.include_router(health.router)
    router.include_router(models.router)
    router.include_router(tabular.router)
    router.include_router(tabular_artifacts.router)
    router.include_router(studies.router)
    router.include_router(suggestions.router)
    router.include_router(predictions.router)
    router.include_router(candidates.router)
    router.include_router(acquisitions.router)
    router.include_router(artifacts.router)
    return router
