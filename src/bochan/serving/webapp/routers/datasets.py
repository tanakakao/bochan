"""Dataset routes for the Web API."""

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException

from bochan.serving.fastapi.converters import to_serializable
from bochan.serving.workbench.datasets import (
    build_dataset_record,
    dataframe_preview,
    load_dataframe_from_payload,
)

from ..logging import log_event
from ..schemas.dataset import DatasetLoadRequest


def _profile_with_category_values(record: Any) -> dict[str, Any]:
    """Add complete low-cardinality values, including numeric categories, for UI selects."""

    profile = {
        **record.profile,
        "columns": [dict(column) for column in record.profile["columns"]],
    }
    for column in profile["columns"]:
        if int(column.get("unique_count", 0)) > 30:
            continue
        series = record.data[column["name"]].dropna()
        column["values"] = to_serializable(series.unique().tolist())
    return profile


def create_datasets_router(*, dataset_store: Any, logger: Any) -> APIRouter:
    """Create dataset upload, listing, and retrieval routes."""

    router = APIRouter()

    @router.get("/datasets")
    def list_datasets() -> dict[str, Any]:
        return {"datasets": dataset_store.list()}

    @router.post("/datasets")
    def load_dataset(request: DatasetLoadRequest) -> dict[str, Any]:
        started = perf_counter()
        log_event(
            logger,
            logging.INFO,
            "dataset_load_started",
            "Dataset loading started",
            dataset_name=request.name,
            source_type=request.source_type,
        )
        try:
            data, metadata = load_dataframe_from_payload(
                source_type=request.source_type,
                content_base64=request.content_base64,
                name=request.name,
                encoding=request.encoding,
                sep=request.sep,
                sheet_name=request.sheet_name,
            )
            record = build_dataset_record(
                data=data,
                name=request.name or "dataset",
                source_type=request.source_type,
                metadata=metadata,
            )
            dataset_store.add(record)
            log_event(
                logger,
                logging.INFO,
                "dataset_load_completed",
                "Dataset loading completed",
                dataset_id=record.dataset_id,
                dataset_name=record.name,
                source_type=record.source_type,
                n_rows=record.profile["n_rows"],
                n_columns=record.profile["n_columns"],
                duration_ms=round((perf_counter() - started) * 1000, 3),
            )
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": _profile_with_category_values(record),
                "preview": dataframe_preview(record.data, limit=100),
            }
        except Exception as exc:
            logger.exception(
                "Dataset loading failed",
                extra={
                    "event": "dataset_load_failed",
                    "dataset_name": request.name,
                    "source_type": request.source_type,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, limit: int = 100) -> dict[str, Any]:
        try:
            record = dataset_store.get(dataset_id)
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": _profile_with_category_values(record),
                "preview": dataframe_preview(record.data, limit=limit),
            }
        except KeyError as exc:
            log_event(
                logger,
                logging.WARNING,
                "dataset_not_found",
                "Dataset was not found",
                dataset_id=dataset_id,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "Dataset retrieval failed",
                extra={"event": "dataset_get_failed", "dataset_id": dataset_id},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


__all__ = ["_profile_with_category_values", "create_datasets_router"]
