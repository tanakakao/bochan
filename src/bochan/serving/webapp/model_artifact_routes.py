"""FastAPI routes for downloading and restoring Web model artifacts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from bochan.desktop.services import (
    build_dataset_record,
    dataframe_preview,
)
from bochan.serving.fastapi.converters import to_serializable

from .model_artifacts import (
    deserialize_web_model_artifact,
    restore_web_model_artifact,
    serialize_web_model_artifact,
)
from .model_reuse import model_reuse_signature, register_model_signature


def _profile_with_category_values(record: Any) -> dict[str, Any]:
    """Return the imported dataset profile with low-cardinality values."""

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


def create_model_artifact_router(dataset_store: Any) -> APIRouter:
    """Create Web-only routes bound to the application's dataset store."""

    router = APIRouter(tags=["web-model-artifacts"])

    @router.get("/runs/{run_id}/model-artifact")
    def download_model_artifact(run_id: str) -> Response:
        """Download a fitted Web model, dataset, settings, and Results state."""

        try:
            content, filename = serialize_web_model_artifact(run_id)
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-Model-Artifact-Version": "1",
                },
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/model-artifacts/import")
    async def import_model_artifact(
        request: Request,
        trust_pickle: bool = False,
    ) -> dict[str, Any]:
        """Restore a trusted downloaded model artifact without retraining."""

        try:
            content = await request.body()
            payload = deserialize_web_model_artifact(
                content,
                trust_pickle=trust_pickle,
                map_location="cpu",
            )
            original_result = payload["result"]
            supplied_name = request.headers.get("X-Model-Filename", "").strip()
            dataset_name = str(
                original_result.get("dataset_name")
                or supplied_name.removesuffix(".bochan.pt")
                or "imported_model"
            )
            record = build_dataset_record(
                data=payload["data"].copy(),
                name=dataset_name,
                source_type="model_artifact",
                metadata={
                    "artifact_filename": supplied_name or None,
                    "artifact_version": payload.get("artifact_version"),
                    "artifact_bochan_version": payload.get("bochan_version"),
                },
            )
            dataset_store.add(record)
            run_id, result, request_payload = restore_web_model_artifact(
                payload,
                dataset_id=record.dataset_id,
                dataset_name=record.name,
            )
            if request_payload:
                signature = model_reuse_signature(SimpleNamespace(**request_payload))
                register_model_signature(run_id, signature)

            return {
                "dataset": {
                    "dataset_id": record.dataset_id,
                    "name": record.name,
                    "source_type": record.source_type,
                    "profile": _profile_with_category_values(record),
                    "preview": dataframe_preview(record.data, limit=100),
                },
                "result": result,
                "request": to_serializable(request_payload),
                "artifact": {
                    "filename": supplied_name or None,
                    "artifact_version": payload.get("artifact_version"),
                    "bochan_version": payload.get("bochan_version"),
                    "original_run_id": payload.get("original_run_id"),
                    "restored_run_id": run_id,
                    "pickle_warning": (
                        "Only import model artifacts created by a trusted bochan Web application."
                    ),
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


__all__ = ["create_model_artifact_router"]
