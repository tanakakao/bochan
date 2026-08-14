"""React-oriented FastAPI application for the bochan web workbench."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from bochan.serving.fastapi import create_api_router
from bochan.serving.workbench.datasets import DatasetStore

from .logging import (
    configure_logging,
    current_request_id,
    get_logger,
    log_event,
    reset_request_id,
    set_request_id,
)
from .routers import create_web_router


def create_app(
    *,
    title: str = "bochan Web API",
    version: str = "0.3.0",
    api_prefix: str = "/api/v1",
    cors_origins: Sequence[str] | None = None,
    include_core_api: bool = True,
) -> FastAPI:
    """Create the FastAPI application used by the React workbench."""

    configured_log_path = configure_logging()
    logger = get_logger("api")
    app = FastAPI(title=title, version=version)
    allowed_origins = list(cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Content-Disposition", "X-Model-Artifact-Version"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        token = set_request_id(request_id)
        started = perf_counter()
        log_event(
            logger,
            logging.INFO,
            "http_request_started",
            "HTTP request started",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            client=request.client.host if request.client else None,
        )
        try:
            response = await call_next(request)
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "http_request_failed",
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                exc_info=True,
            )
            raise
        else:
            duration_ms = round((perf_counter() - started) * 1000, 3)
            response.headers["X-Request-ID"] = request_id
            response_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            log_event(
                logger,
                response_level,
                "http_request_completed",
                "HTTP request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        finally:
            reset_request_id(token)

    if include_core_api:
        app.include_router(create_api_router(prefix=api_prefix))

    dataset_store = DatasetStore()
    app.include_router(
        create_web_router(
            api_prefix=api_prefix,
            dataset_store=dataset_store,
            logger=logger,
        )
    )
    log_event(
        logger,
        logging.INFO,
        "application_configured",
        "bochan web application configured",
        title=title,
        version=version,
        log_file=str(configured_log_path),
        request_id=current_request_id(),
    )
    return app


app = create_app()


__all__ = ["app", "create_app"]
