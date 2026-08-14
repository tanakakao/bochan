"""Result visualization routes for the Web API."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ..logging import log_event
from ..schemas.visualization import VisualizationRequestSchema
from ..visualization_sessions import build_visualization, get_visualization_session


def create_visualization_router(*, logger: Any) -> APIRouter:
    """Create routes for result visualizations backed by retained run sessions."""

    router = APIRouter()

    @router.post("/runs/{run_id}/visualizations")
    def result_visualization(
        run_id: str,
        request: VisualizationRequestSchema,
    ) -> dict[str, Any]:
        try:
            get_visualization_session(run_id)
        except KeyError as exc:
            log_event(
                logger,
                logging.WARNING,
                "visualization_session_not_found",
                "Visualization session was not found",
                visualization_run_id=run_id,
                visualization_kind=request.kind,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            return build_visualization(run_id, request.model_dump())
        except Exception as exc:
            logger.exception(
                "Result visualization failed",
                extra={
                    "event": "visualization_failed",
                    "visualization_run_id": run_id,
                    "visualization_kind": request.kind,
                },
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


__all__ = ["create_visualization_router"]
