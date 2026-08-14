"""Structured log routes for the Web API."""

from typing import Any

from fastapi import APIRouter

from ..logging import log_file_path, read_recent_logs


def create_logs_router() -> APIRouter:
    """Create routes for reading recent Web application logs."""

    router = APIRouter()

    @router.get("/logs")
    def recent_logs(
        limit: int = 200,
        level: str | None = None,
        event: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        entries = read_recent_logs(
            limit=limit,
            level=level,
            event=event,
            request_id=request_id,
        )
        return {
            "entries": entries,
            "count": len(entries),
            "log_file": str(log_file_path()),
        }

    return router


__all__ = ["create_logs_router"]
