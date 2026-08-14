"""HTTP routers for the React-oriented Web API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..services.model_artifacts import create_model_artifact_router
from .capabilities import create_capabilities_router
from .composition import create_composition_router
from .datasets import create_datasets_router
from .logs import create_logs_router
from .regression import create_regression_router, run_regression_request
from .visualizations import create_visualization_router


def create_web_router(
    *,
    api_prefix: str,
    dataset_store: Any,
    logger: Any,
) -> APIRouter:
    """Create the complete React Web API router."""

    router = APIRouter(prefix=api_prefix, tags=["web"])
    router.include_router(create_capabilities_router(api_prefix=api_prefix))
    router.include_router(create_logs_router())
    router.include_router(create_datasets_router(dataset_store=dataset_store, logger=logger))
    router.include_router(create_regression_router(dataset_store=dataset_store, logger=logger))
    router.include_router(create_visualization_router(logger=logger))
    router.include_router(create_model_artifact_router(dataset_store))

    def run_regression(request: Any) -> dict[str, Any]:
        return run_regression_request(request, dataset_store=dataset_store, logger=logger)

    router.include_router(create_composition_router(run_regression=run_regression))
    return router


__all__ = ["create_web_router"]
