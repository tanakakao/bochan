"""FastAPI routes for Web artifacts, project archives, and experiment history."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response

from bochan.desktop.services import (
    build_dataset_record,
    dataframe_preview,
)
from bochan.serving.fastapi.converters import to_serializable

from .experiment_history import ExperimentCycleRequest, ExperimentHistoryStore
from .model_artifacts import (
    deserialize_web_model_artifact,
    restore_web_model_artifact,
    serialize_web_model_artifact,
)
from .model_reuse import model_reuse_signature, register_model_signature
from .project_archive import (
    PROJECT_ARCHIVE_VERSION,
    ExperimentProjectExportRequest,
    is_experiment_project_archive,
    restore_experiment_project,
    serialize_experiment_project,
)


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


def _dataset_response(record: Any) -> dict[str, Any]:
    """Build the common Web dataset response for restored artifacts."""

    return {
        "dataset_id": record.dataset_id,
        "name": record.name,
        "source_type": record.source_type,
        "profile": _profile_with_category_values(record),
        "preview": dataframe_preview(record.data, limit=100),
    }


def create_model_artifact_router(dataset_store: Any) -> APIRouter:
    """Create Web-only routes bound to the application's dataset store."""

    router = APIRouter(tags=["web-model-artifacts"])
    experiment_history = ExperimentHistoryStore()

    @router.get("/runs/{run_id}/model-artifact")
    def download_model_artifact(
        run_id: str,
        filename: str | None = None,
    ) -> Response:
        """Download a fitted Web model with an optional user-selected filename."""

        try:
            content, resolved_filename = serialize_web_model_artifact(
                run_id,
                filename=filename,
            )
            disposition = (
                'attachment; filename="bochan_model.bochan.pt"; '
                f"filename*=UTF-8''{quote(resolved_filename)}"
            )
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": disposition,
                    "X-Model-Artifact-Version": "1",
                },
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/experiment-projects/export", tags=["web-experiment-projects"])
    def export_experiment_project(request: ExperimentProjectExportRequest) -> Response:
        """Download the complete project with latest and optional historical models."""

        try:
            content, filename = serialize_experiment_project(
                dataset_store,
                experiment_history,
                request,
            )
            disposition = (
                'attachment; filename="bochan_project.bochan-project.zip"; '
                f"filename*=UTF-8''{quote(filename)}"
            )
            return Response(
                content=content,
                media_type="application/zip",
                headers={
                    "Content-Disposition": disposition,
                    "X-Project-Archive-Version": str(PROJECT_ARCHIVE_VERSION),
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
        """Restore either a trusted model artifact or a project ZIP archive."""

        try:
            content = await request.body()
            supplied_name = request.headers.get("X-Model-Filename", "").strip()
            expects_project = supplied_name.lower().endswith(".bochan-project.zip")

            if is_experiment_project_archive(content):
                imported = restore_experiment_project(
                    content,
                    dataset_store,
                    experiment_history,
                    trust_pickle=trust_pickle,
                )
                record = imported["record"]
                archive = imported["archive"]
                if archive["model_included"]:
                    if archive["model_restored"]:
                        pickle_warning = (
                            "Embedded project models were restored after explicit trust confirmation."
                        )
                    else:
                        pickle_warning = (
                            "The project contains model artifacts, but they were not loaded. "
                            "Import again with explicit pickle trust to restore them."
                        )
                else:
                    pickle_warning = (
                        "This project does not contain a trained model; retraining is required."
                    )
                return {
                    "dataset": _dataset_response(record),
                    "result": imported["result"],
                    "request": to_serializable(imported["request"]),
                    "history": imported["history"],
                    "artifact": {
                        "filename": supplied_name or None,
                        "artifact_version": archive["version"],
                        "bochan_version": None,
                        "original_run_id": archive.get("original_run_id"),
                        "restored_run_id": archive["restored_run_id"],
                        "project_archive": True,
                        "model_included": archive["model_included"],
                        "model_count": archive["model_count"],
                        "latest_model_included": archive["latest_model_included"],
                        "past_model_count": archive["past_model_count"],
                        "model_restored": archive["model_restored"],
                        "restored_model_count": archive["restored_model_count"],
                        "active_model_restored": archive["active_model_restored"],
                        "requires_pickle_trust": archive["requires_pickle_trust"],
                        "restored_models": archive["restored_models"],
                        "model_export_warnings": archive["model_export_warnings"],
                        "cycle_count": archive["cycle_count"],
                        "dataset_count": archive["dataset_count"],
                        "pickle_warning": pickle_warning,
                    },
                }

            if expects_project:
                raise ValueError(
                    "The selected .bochan-project.zip file is not a valid bochan project archive."
                )

            payload = deserialize_web_model_artifact(
                content,
                trust_pickle=trust_pickle,
                map_location="cpu",
            )
            original_result = payload["result"]
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
                "dataset": _dataset_response(record),
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

    @router.post("/experiment-cycles", tags=["web-experiment-history"])
    def record_experiment_cycle(request: ExperimentCycleRequest) -> dict[str, Any]:
        """Record actual conditions, outcomes, and optimization settings for one cycle."""

        try:
            parent_record = dataset_store.get(request.parent_dataset_id)
            updated_record = dataset_store.get(request.dataset_id)
            if int(parent_record.profile["n_rows"]) != request.n_rows_before:
                raise ValueError("n_rows_before does not match the parent dataset.")
            if int(updated_record.profile["n_rows"]) != request.n_rows_after:
                raise ValueError("n_rows_after does not match the updated dataset.")
            return {"cycle": experiment_history.add(request)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/experiment-cycles", tags=["web-experiment-history"])
    def list_experiment_cycles(dataset_id: str) -> dict[str, Any]:
        """Return the experiment lineage and objective-progress visualizations."""

        try:
            dataset_store.get(dataset_id)
            return experiment_history.response_for_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


__all__ = ["create_model_artifact_router"]
