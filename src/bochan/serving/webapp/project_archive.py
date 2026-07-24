"""Portable project archives containing datasets, history, settings, and models."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO, StringIO
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, is_zipfile

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bochan.desktop.services import build_dataset_record
from bochan.serving.fastapi.converters import to_serializable

from .experiment_history import ExperimentCycleRequest, ExperimentHistoryStore
from .model_artifacts import (
    deserialize_web_model_artifact,
    restore_web_model_artifact,
    serialize_web_model_artifact,
)

PROJECT_ARCHIVE_FORMAT = "bochan-experiment-project"
PROJECT_ARCHIVE_VERSION = 2
SUPPORTED_PROJECT_ARCHIVE_VERSIONS = {1, PROJECT_ARCHIVE_VERSION}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_FILES = 1000


class ExperimentProjectExportRequest(BaseModel):
    """Current Web state and model-retention policy for a project archive."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    include_latest_model: bool = True
    include_past_models: bool = False


def is_experiment_project_archive(content: bytes) -> bool:
    """Return whether raw bytes look like a supported ZIP project archive."""

    if not content or not is_zipfile(BytesIO(content)):
        return False
    try:
        with ZipFile(BytesIO(content)) as archive:
            if "manifest.json" not in archive.namelist():
                return False
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            return manifest.get("format") == PROJECT_ARCHIVE_FORMAT
    except (BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _json_bytes(value: Any) -> bytes:
    """Encode one JSON document using stable readable settings."""

    return json.dumps(
        to_serializable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _dataset_ids_for_export(
    dataset_id: str,
    cycles: list[dict[str, Any]],
) -> list[str]:
    """Return the ordered, unique dataset lineage required by the archive."""

    raw_ids: list[str] = []
    if cycles:
        raw_ids.append(str(cycles[0]["parent_dataset_id"]))
        raw_ids.extend(str(cycle["dataset_id"]) for cycle in cycles)
    else:
        raw_ids.append(dataset_id)
    if dataset_id not in raw_ids:
        raw_ids.append(dataset_id)

    ordered: list[str] = []
    for value in raw_ids:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _safe_filename(name: str) -> str:
    """Build a filesystem-friendly archive filename without losing Japanese text."""

    stem = re.sub(r"\.[^.]+$", "", name).strip() or "bochan_project"
    stem = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", stem)
    return f"{stem}.bochan-project.zip"


def _model_export_candidates(
    request: ExperimentProjectExportRequest,
    cycles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select latest and optional historical model runs without duplicates."""

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(
        run_id: Any,
        *,
        role: str,
        dataset_id: str,
        cycle_number: int | None,
    ) -> None:
        normalized = str(run_id or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(
            {
                "run_id": normalized,
                "role": role,
                "dataset_id": dataset_id,
                "cycle_number": cycle_number,
            }
        )

    result_run_id = request.result.get("visualization_run_id")
    if request.include_latest_model:
        if result_run_id:
            add_candidate(
                result_run_id,
                role="latest",
                dataset_id=request.dataset_id,
                cycle_number=None,
            )
        else:
            for cycle in reversed(cycles):
                if cycle.get("source_run_id"):
                    add_candidate(
                        cycle["source_run_id"],
                        role="latest",
                        dataset_id=str(cycle["parent_dataset_id"]),
                        cycle_number=int(cycle["cycle_number"]),
                    )
                    break

    if request.include_past_models:
        for cycle in cycles:
            add_candidate(
                cycle.get("source_run_id"),
                role="historical",
                dataset_id=str(cycle["parent_dataset_id"]),
                cycle_number=int(cycle["cycle_number"]),
            )

    return candidates


def _write_embedded_models(
    archive: ZipFile,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Write selected model artifacts and return manifest entries and warnings."""

    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        run_id = str(candidate["run_id"])
        try:
            content, artifact_filename = serialize_web_model_artifact(run_id)
        except (KeyError, RuntimeError, ValueError) as exc:
            warnings.append(f"Model run {run_id} could not be exported: {exc}")
            continue
        if len(content) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"Model artifact {run_id} exceeds the project archive size limit.")

        if candidate["role"] == "latest":
            path = "models/latest.bochan.pt"
        else:
            cycle_number = candidate.get("cycle_number")
            suffix = f"cycle_{int(cycle_number):04d}" if cycle_number is not None else f"history_{index:04d}"
            path = f"models/{suffix}.bochan.pt"
        archive.writestr(path, content)
        entries.append(
            {
                **candidate,
                "path": path,
                "artifact_filename": artifact_filename,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return entries, warnings


def serialize_experiment_project(
    dataset_store: Any,
    history_store: ExperimentHistoryStore,
    request: ExperimentProjectExportRequest,
) -> tuple[bytes, str]:
    """Serialize dataset lineage, cycle history, UI state, and selected models."""

    current_record = dataset_store.get(request.dataset_id)
    cycles = history_store.list_for_dataset(request.dataset_id)
    dataset_ids = _dataset_ids_for_export(request.dataset_id, cycles)
    dataset_entries: list[dict[str, Any]] = []

    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for index, dataset_id in enumerate(dataset_ids):
            record = dataset_store.get(dataset_id)
            data_path = f"datasets/{index:04d}.json"
            table_json = record.data.to_json(
                orient="table",
                date_format="iso",
                index=False,
                force_ascii=False,
            )
            archive.writestr(data_path, table_json.encode("utf-8"))
            dataset_entries.append(
                {
                    "dataset_id": record.dataset_id,
                    "name": record.name,
                    "source_type": record.source_type,
                    "metadata": to_serializable(getattr(record, "metadata", {})),
                    "data_path": data_path,
                    "n_rows": int(record.profile["n_rows"]),
                    "n_columns": int(record.profile["n_columns"]),
                }
            )

        archive.writestr("history.json", _json_bytes({"cycles": cycles}))
        archive.writestr(
            "workbench.json",
            _json_bytes({"request": request.request, "result": request.result}),
        )
        model_entries, model_warnings = _write_embedded_models(
            archive,
            _model_export_candidates(request, cycles),
        )
        manifest = {
            "format": PROJECT_ARCHIVE_FORMAT,
            "version": PROJECT_ARCHIVE_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "current_dataset_id": request.dataset_id,
            "datasets": dataset_entries,
            "history_path": "history.json",
            "workbench_path": "workbench.json",
            "models": model_entries,
            "model_included": bool(model_entries),
            "latest_model_included": any(entry["role"] == "latest" for entry in model_entries),
            "past_model_count": sum(entry["role"] == "historical" for entry in model_entries),
            "model_policy": {
                "include_latest_model": request.include_latest_model,
                "include_past_models": request.include_past_models,
                "default_include_past_models": False,
            },
            "model_export_warnings": model_warnings,
        }
        archive.writestr("manifest.json", _json_bytes(manifest))

    content = output.getvalue()
    with ZipFile(BytesIO(content)) as validation_archive:
        _validate_zip_limits(validation_archive)
    return content, _safe_filename(current_record.name)


def _read_json_file(archive: ZipFile, path: str) -> dict[str, Any]:
    """Read and validate one JSON object from an archive member."""

    if path not in archive.namelist():
        raise ValueError(f"Project archive member is missing: {path}")
    try:
        value = json.loads(archive.read(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Project archive member is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Project archive member must contain a JSON object: {path}")
    return value


def _validate_zip_limits(archive: ZipFile) -> None:
    """Reject unexpectedly large or encrypted archives before parsing contents."""

    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_FILES:
        raise ValueError("Project archive contains too many files.")
    total_size = sum(member.file_size for member in members)
    if total_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Project archive is too large after decompression.")
    if any(member.flag_bits & 0x1 for member in members):
        raise ValueError("Encrypted project archives are not supported.")


def _restore_cycles(
    history_store: ExperimentHistoryStore,
    archived_cycles: list[dict[str, Any]],
    dataset_id_map: dict[str, str],
) -> None:
    """Restore ordered cycles with remapped datasets and preserved timestamps."""

    request_fields = set(ExperimentCycleRequest.model_fields)
    for archived in archived_cycles:
        old_parent = str(archived.get("parent_dataset_id") or "")
        old_dataset = str(archived.get("dataset_id") or "")
        if old_parent not in dataset_id_map or old_dataset not in dataset_id_map:
            raise ValueError("Experiment history references a dataset missing from the archive.")

        payload = {
            key: deepcopy(value)
            for key, value in archived.items()
            if key in request_fields
        }
        payload["parent_dataset_id"] = dataset_id_map[old_parent]
        payload["dataset_id"] = dataset_id_map[old_dataset]
        validated = ExperimentCycleRequest.model_validate(payload)
        restored = history_store.add(validated)

        # Preserve archival identity and time while keeping newly generated IDs collision-free.
        with history_store._lock:  # noqa: SLF001
            stored = history_store._by_dataset[validated.dataset_id]  # noqa: SLF001
            stored["archived_cycle_id"] = archived.get("cycle_id")
            stored["archived_cycle_number"] = archived.get("cycle_number")
            stored["created_at"] = str(archived.get("created_at") or stored["created_at"])
            history_store._by_cycle_id[restored["cycle_id"]] = stored  # noqa: SLF001


def _sanitized_workbench(
    workbench: dict[str, Any],
    *,
    current_dataset_id: str,
    current_dataset_name: str,
    model_included: bool,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Remap imported workbench state and force retraining unless a current model loads."""

    request_payload = deepcopy(workbench.get("request") or {})
    result_payload = deepcopy(workbench.get("result") or {})
    if not isinstance(request_payload, dict) or not isinstance(result_payload, dict):
        raise ValueError("Project workbench state is invalid.")

    request_payload["dataset_id"] = current_dataset_id
    original_run_id = result_payload.pop("visualization_run_id", None)
    result_payload["dataset_id"] = current_dataset_id
    result_payload["dataset_name"] = current_dataset_name
    metadata = result_payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    result_payload["metadata"] = {
        **metadata,
        "stale_after_data_append": True,
        "restored_from_project_archive": True,
        "project_model_included": model_included,
        "project_model_restored": False,
    }
    return request_payload, result_payload, str(original_run_id) if original_run_id else None


def _model_manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the additive model manifest used by project archive version 2."""

    raw_entries = manifest.get("models") or []
    if not isinstance(raw_entries, list):
        raise ValueError("Project model manifest is invalid.")
    entries: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("Project model entry is invalid.")
        path = str(raw.get("path") or "")
        dataset_id = str(raw.get("dataset_id") or "")
        role = str(raw.get("role") or "")
        if not path.startswith("models/") or not path.endswith(".bochan.pt"):
            raise ValueError("Project model path is invalid.")
        if path in paths or not dataset_id or role not in {"latest", "historical"}:
            raise ValueError("Project model entry is incomplete or duplicated.")
        paths.add(path)
        entries.append(deepcopy(raw))
    return entries


def _read_model_member(archive: ZipFile, entry: dict[str, Any]) -> bytes:
    """Read and checksum one embedded model without deserializing pickle."""

    path = str(entry["path"])
    if path not in archive.namelist():
        raise ValueError(f"Project model member is missing: {path}")
    content = archive.read(path)
    expected_hash = str(entry.get("sha256") or "")
    if expected_hash and hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"Project model checksum does not match: {path}")
    return content


def _restore_embedded_models(
    archive: ZipFile,
    entries: list[dict[str, Any]],
    *,
    dataset_store: Any,
    dataset_id_map: dict[str, str],
    trust_pickle: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Validate all embedded models and optionally restore their fitted sessions."""

    restored_models: list[dict[str, Any]] = []
    latest_model: dict[str, Any] | None = None
    for entry in entries:
        content = _read_model_member(archive, entry)
        old_dataset_id = str(entry["dataset_id"])
        if old_dataset_id not in dataset_id_map:
            raise ValueError("Project model references a dataset missing from the archive.")
        new_dataset_id = dataset_id_map[old_dataset_id]
        descriptor = {
            "role": entry["role"],
            "cycle_number": entry.get("cycle_number"),
            "original_run_id": entry.get("run_id"),
            "dataset_id": new_dataset_id,
            "artifact_filename": entry.get("artifact_filename"),
            "restored": False,
        }
        if not trust_pickle:
            restored_models.append(descriptor)
            continue

        payload = deserialize_web_model_artifact(
            content,
            trust_pickle=True,
            map_location="cpu",
        )
        payload_result = payload.get("result") or {}
        payload_dataset_id = str(payload_result.get("dataset_id") or "")
        if payload_dataset_id and payload_dataset_id != old_dataset_id:
            raise ValueError("Embedded model dataset does not match the project manifest.")
        record = dataset_store.get(new_dataset_id)
        if len(payload["data"]) != int(record.profile["n_rows"]):
            raise ValueError("Embedded model training rows do not match the archived dataset.")
        run_id, model_result, model_request = restore_web_model_artifact(
            payload,
            dataset_id=new_dataset_id,
            dataset_name=record.name,
        )
        descriptor.update({"restored": True, "restored_run_id": run_id})
        restored_models.append(descriptor)
        if entry["role"] == "latest":
            latest_model = {
                **descriptor,
                "result": model_result,
                "request": model_request,
            }
    return restored_models, latest_model


def restore_experiment_project(
    content: bytes,
    dataset_store: Any,
    history_store: ExperimentHistoryStore,
    *,
    trust_pickle: bool = False,
) -> dict[str, Any]:
    """Restore datasets, history, settings, and trusted embedded model artifacts."""

    if not is_experiment_project_archive(content):
        raise ValueError("The selected file is not a bochan experiment project archive.")

    try:
        with ZipFile(BytesIO(content)) as archive:
            _validate_zip_limits(archive)
            manifest = _read_json_file(archive, "manifest.json")
            if manifest.get("format") != PROJECT_ARCHIVE_FORMAT:
                raise ValueError("Unsupported project archive format.")
            archive_version = int(manifest.get("version", 0))
            if archive_version not in SUPPORTED_PROJECT_ARCHIVE_VERSIONS:
                raise ValueError("Unsupported project archive version.")

            raw_entries = manifest.get("datasets")
            if not isinstance(raw_entries, list) or not raw_entries:
                raise ValueError("Project archive does not contain datasets.")

            dataset_id_map: dict[str, str] = {}
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    raise ValueError("Project dataset manifest is invalid.")
                old_dataset_id = str(entry.get("dataset_id") or "")
                data_path = str(entry.get("data_path") or "")
                if not old_dataset_id or data_path not in archive.namelist():
                    raise ValueError("Project dataset entry is incomplete.")
                try:
                    data = pd.read_json(
                        StringIO(archive.read(data_path).decode("utf-8")),
                        orient="table",
                    )
                except Exception as exc:
                    raise ValueError(f"Could not restore project dataset: {data_path}") from exc

                archived_metadata = entry.get("metadata")
                archived_metadata = archived_metadata if isinstance(archived_metadata, dict) else {}
                record = build_dataset_record(
                    data=data,
                    name=str(entry.get("name") or "imported_project"),
                    source_type="project_archive",
                    metadata={
                        **archived_metadata,
                        "project_archive_version": archive_version,
                        "archived_dataset_id": old_dataset_id,
                        "archived_source_type": entry.get("source_type"),
                    },
                )
                dataset_store.add(record)
                dataset_id_map[old_dataset_id] = record.dataset_id

            history_path = str(manifest.get("history_path") or "history.json")
            history_payload = _read_json_file(archive, history_path)
            archived_cycles = history_payload.get("cycles") or []
            if not isinstance(archived_cycles, list):
                raise ValueError("Project experiment history is invalid.")
            _restore_cycles(history_store, archived_cycles, dataset_id_map)

            old_current_id = str(manifest.get("current_dataset_id") or "")
            if old_current_id not in dataset_id_map:
                raise ValueError("Current dataset is missing from the project archive.")
            current_record = dataset_store.get(dataset_id_map[old_current_id])

            model_entries = _model_manifest_entries(manifest)
            restored_models, latest_model = _restore_embedded_models(
                archive,
                model_entries,
                dataset_store=dataset_store,
                dataset_id_map=dataset_id_map,
                trust_pickle=trust_pickle,
            )

            workbench_path = str(manifest.get("workbench_path") or "workbench.json")
            workbench = _read_json_file(archive, workbench_path)
            request_payload, result_payload, original_run_id = _sanitized_workbench(
                workbench,
                current_dataset_id=current_record.dataset_id,
                current_dataset_name=current_record.name,
                model_included=bool(model_entries),
            )

            active_model_restored = bool(
                latest_model
                and latest_model["dataset_id"] == current_record.dataset_id
            )
            if active_model_restored and latest_model is not None:
                model_request = latest_model.get("request")
                if isinstance(model_request, dict) and model_request:
                    request_payload = deepcopy(model_request)
                    request_payload["dataset_id"] = current_record.dataset_id
                result_payload = deepcopy(latest_model["result"])
                metadata = result_payload.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                metadata.pop("stale_after_data_append", None)
                result_payload["metadata"] = {
                    **metadata,
                    "restored_from_project_archive": True,
                    "project_model_included": True,
                    "project_model_restored": True,
                }

            restored_count = sum(bool(item.get("restored")) for item in restored_models)
            restored_run_id = (
                str(latest_model["restored_run_id"])
                if active_model_restored and latest_model is not None
                else f"project-{uuid4().hex}"
            )
            return {
                "record": current_record,
                "request": request_payload,
                "result": result_payload,
                "history": history_store.response_for_dataset(current_record.dataset_id),
                "archive": {
                    "format": PROJECT_ARCHIVE_FORMAT,
                    "version": archive_version,
                    "exported_at": manifest.get("exported_at"),
                    "imported_at": datetime.now(UTC).isoformat(),
                    "dataset_count": len(dataset_id_map),
                    "cycle_count": len(archived_cycles),
                    "original_run_id": original_run_id,
                    "restored_run_id": restored_run_id,
                    "model_included": bool(model_entries),
                    "model_count": len(model_entries),
                    "latest_model_included": any(
                        entry.get("role") == "latest" for entry in model_entries
                    ),
                    "past_model_count": sum(
                        entry.get("role") == "historical" for entry in model_entries
                    ),
                    "model_restored": restored_count > 0,
                    "restored_model_count": restored_count,
                    "active_model_restored": active_model_restored,
                    "requires_pickle_trust": bool(model_entries) and not trust_pickle,
                    "restored_models": restored_models,
                    "model_export_warnings": manifest.get("model_export_warnings") or [],
                },
            }
    except BadZipFile as exc:
        raise ValueError("The project archive ZIP is damaged.") from exc


__all__ = [
    "ExperimentProjectExportRequest",
    "PROJECT_ARCHIVE_FORMAT",
    "PROJECT_ARCHIVE_VERSION",
    "SUPPORTED_PROJECT_ARCHIVE_VERSIONS",
    "is_experiment_project_archive",
    "restore_experiment_project",
    "serialize_experiment_project",
]
