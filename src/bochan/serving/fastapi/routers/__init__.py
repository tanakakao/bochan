"""Router composition for the bochan FastAPI app."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    acquisitions,
    alignn_tabular,
    artifacts,
    candidates,
    chgnet_tabular,
    health,
    m3gnet_tabular,
    mace_tabular,
    material_residual,
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
    router.include_router(alignn_tabular.router)
    router.include_router(chgnet_tabular.router)
    router.include_router(m3gnet_tabular.router)
    router.include_router(mace_tabular.router)
    router.include_router(material_residual.router)
    router.include_router(tabular_artifacts.router)
    router.include_router(studies.router)
    router.include_router(suggestions.router)
    router.include_router(predictions.router)
    router.include_router(candidates.router)
    router.include_router(acquisitions.router)
    router.include_router(artifacts.router)
    return router


__all__ = ["create_api_router"]
