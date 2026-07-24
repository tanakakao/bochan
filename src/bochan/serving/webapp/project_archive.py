"""Portable project archives containing datasets, experiment history, and Web settings."""

from __future__ import annotations

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

PROJECT_ARCHIVE_FORMAT = "bochan-experiment-project"
PROJECT_ARCHIVE_VERSION = 1
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_FILES = 1000


class ExperimentProjectExportRequest(BaseModel):
    """Current Web state to store alongside the complete dataset lineage."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


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


def serialize_experiment_project(
    dataset_store: Any,
    history_store: ExperimentHistoryStore,
    request: ExperimentProjectExportRequest,
) -> tuple[bytes, str]:
    """Serialize the current dataset lineage, cycle history, and UI state to ZIP."""

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
        manifest = {
            "format": PROJECT_ARCHIVE_FORMAT,
            "version": PROJECT_ARCHIVE_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "current_dataset_id": request.dataset_id,
            "datasets": dataset_entries,
            "history_path": "history.json",
            "workbench_path": "workbench.json",
            "model_included": False,
        }
        archive.writestr("manifest.json", _json_bytes(manifest))

    return output.getvalue(), _safe_filename(current_record.name)


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
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Remap imported workbench state and force retraining of non-exported models."""

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
        "project_model_included": False,
    }
    return request_payload, result_payload, str(original_run_id) if original_run_id else None


def restore_experiment_project(
    content: bytes,
    dataset_store: Any,
    history_store: ExperimentHistoryStore,
) -> dict[str, Any]:
    """Restore datasets, cycle history, and compatible UI state from a project ZIP."""

    if not is_experiment_project_archive(content):
        raise ValueError("The selected file is not a bochan experiment project archive.")

    try:
        with ZipFile(BytesIO(content)) as archive:
            _validate_zip_limits(archive)
            manifest = _read_json_file(archive, "manifest.json")
            if manifest.get("format") != PROJECT_ARCHIVE_FORMAT:
                raise ValueError("Unsupported project archive format.")
            if int(manifest.get("version", 0)) != PROJECT_ARCHIVE_VERSION:
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
                        "project_archive_version": PROJECT_ARCHIVE_VERSION,
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

            workbench_path = str(manifest.get("workbench_path") or "workbench.json")
            workbench = _read_json_file(archive, workbench_path)
            request_payload, result_payload, original_run_id = _sanitized_workbench(
                workbench,
                current_dataset_id=current_record.dataset_id,
                current_dataset_name=current_record.name,
            )
            return {
                "record": current_record,
                "request": request_payload,
                "result": result_payload,
                "history": history_store.response_for_dataset(current_record.dataset_id),
                "archive": {
                    "format": PROJECT_ARCHIVE_FORMAT,
                    "version": PROJECT_ARCHIVE_VERSION,
                    "exported_at": manifest.get("exported_at"),
                    "imported_at": datetime.now(UTC).isoformat(),
                    "dataset_count": len(dataset_id_map),
                    "cycle_count": len(archived_cycles),
                    "original_run_id": original_run_id,
                    "restored_run_id": f"project-{uuid4().hex}",
                    "model_included": False,
                },
            }
    except BadZipFile as exc:
        raise ValueError("The project archive ZIP is damaged.") from exc


__all__ = [
    "ExperimentProjectExportRequest",
    "PROJECT_ARCHIVE_FORMAT",
    "PROJECT_ARCHIVE_VERSION",
    "is_experiment_project_archive",
    "restore_experiment_project",
    "serialize_experiment_project",
]
