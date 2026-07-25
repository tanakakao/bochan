"""Round-trip tests for portable Web experiment project archives."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient

import bochan.serving.webapp.project_archive as project_archive_module
from bochan.serving.webapp.app import create_app


def _load_csv(client: TestClient, name: str, text: str) -> dict[str, object]:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    response = client.post(
        "/api/v1/datasets",
        json={
            "source_type": "csv",
            "name": name,
            "content_base64": f"data:text/csv;base64,{encoded}",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _result(dataset: dict[str, object]) -> dict[str, object]:
    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["name"],
        "task_type": "regression",
        "model_type": "base",
        "n_train": 2,
        "n_features": 1,
        "feature_columns": ["x"],
        "target_columns": ["y"],
        "target_column": "y",
        "target_settings": [
            {
                "target": "y",
                "task_type": "regression",
                "optimize": True,
                "direction": "maximize",
                "goal": "none",
                "value": None,
            }
        ],
        "directions": {"y": "maximize"},
        "direction": "maximize",
        "best_observed": 2.0,
        "candidates": [
            {
                "rank": 1,
                "values": {"x": 3.0},
                "acq_value": 0.4,
                "predictions": {"y": {"mean": 3.0, "std": 0.2}},
                "predicted_target_mean": 3.0,
                "predicted_target_std": 0.2,
                "constraints_ok": True,
            }
        ],
        "visualizations": [],
        "visualization_warnings": [],
        "visualization_run_id": "archived-run-id",
        "metadata": {"model_details": {"effective_acquisition": "EI"}},
    }


def _request(dataset: dict[str, object]) -> dict[str, object]:
    target_settings = [
        {
            "target": "y",
            "task_type": "regression",
            "optimize": True,
            "direction": "maximize",
            "goal": "none",
            "value": None,
        }
    ]
    return {
        "dataset_id": dataset["dataset_id"],
        "feature_columns": ["x"],
        "target_column": "y",
        "target_columns": ["y"],
        "direction": "maximize",
        "directions": {"y": "maximize"},
        "model_type": "base",
        "model_kwargs": {
            "web_target_settings": target_settings,
            "web_feature_missing": {"strategy": "drop"},
        },
        "fit_maxiter": 128,
        "normalize": True,
        "outcome_transform": True,
        "input_perturbation": False,
        "n_w": 16,
        "perturbation_std": 0.1,
        "search_space": [
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 4.0,
                "fixed": False,
            }
        ],
        "constraints": [],
        "outcome_constraints": [],
        "k_sparse": None,
        "acquisition": {
            "name": "EI",
            "beta": 2.0,
            "acqf_kwargs": {"web_family": "bayesian_optimization"},
        },
        "optimizer": {
            "name": "normal",
            "q": 1,
            "num_restarts": 10,
            "raw_samples": 256,
            "sequential": True,
        },
        "drop_missing": True,
    }


def _record_cycle(
    client: TestClient,
    parent: dict[str, object],
    updated: dict[str, object],
    *,
    source_run_id: str,
) -> dict[str, object]:
    response = client.post(
        "/api/v1/experiment-cycles",
        json={
            "parent_dataset_id": parent["dataset_id"],
            "dataset_id": updated["dataset_id"],
            "dataset_name": updated["name"],
            "source_run_id": source_run_id,
            "append_mode": "manual",
            "n_rows_before": 2,
            "n_rows_after": 3,
            "rows": [{"x": 3.0, "y": 3.0}],
            "feature_columns": ["x"],
            "target_columns": ["y"],
            "target_settings": _request(updated)["model_kwargs"]["web_target_settings"],
            "model": {"type": "base", "n_train": 2},
            "acquisition": {"name": "EI", "family": "bayesian_optimization", "beta": 2.0},
            "optimizer": {"backend": "normal", "q": 1},
            "best_observed_before": {"y": 2.0},
            "candidate_count": 1,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["cycle"]


def test_project_archive_restores_dataset_history_and_settings() -> None:
    source_client = TestClient(create_app(include_core_api=False))
    parent = _load_csv(source_client, "experiment.csv", "x,y\n1,1\n2,2\n")
    updated = _load_csv(source_client, "experiment_updated.csv", "x,y\n1,1\n2,2\n3,3\n")
    original_cycle = _record_cycle(
        source_client,
        parent,
        updated,
        source_run_id="archived-run-id",
    )

    export_response = source_client.post(
        "/api/v1/experiment-projects/export",
        json={
            "dataset_id": updated["dataset_id"],
            "request": _request(updated),
            "result": _result(updated),
        },
    )
    assert export_response.status_code == 200, export_response.text
    assert export_response.headers["content-type"].startswith("application/zip")
    assert ".bochan-project.zip" in export_response.headers["content-disposition"]

    archive_bytes = export_response.content
    with ZipFile(BytesIO(archive_bytes)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "bochan-experiment-project"
        assert manifest["version"] == 2
        assert manifest["model_included"] is False
        assert manifest["model_policy"]["include_latest_model"] is True
        assert manifest["model_policy"]["include_past_models"] is False
        assert len(manifest["datasets"]) == 2
        assert "history.json" in archive.namelist()
        assert "workbench.json" in archive.namelist()

    restored_client = TestClient(create_app(include_core_api=False))
    import_response = restored_client.post(
        "/api/v1/model-artifacts/import?trust_pickle=false",
        content=archive_bytes,
        headers={
            "Content-Type": "application/zip",
            "X-Model-Filename": "experiment.bochan-project.zip",
        },
    )
    assert import_response.status_code == 200, import_response.text
    imported = import_response.json()
    assert imported["artifact"]["project_archive"] is True
    assert imported["artifact"]["model_included"] is False
    assert imported["artifact"]["cycle_count"] == 1
    assert imported["dataset"]["profile"]["n_rows"] == 3
    assert imported["request"]["dataset_id"] == imported["dataset"]["dataset_id"]
    assert imported["request"]["model_type"] == "base"
    assert imported["request"]["acquisition"]["name"] == "EI"
    assert "visualization_run_id" not in imported["result"]
    assert imported["result"]["metadata"]["restored_from_project_archive"] is True
    assert imported["result"]["metadata"]["stale_after_data_append"] is True

    history_response = restored_client.get(
        "/api/v1/experiment-cycles",
        params={"dataset_id": imported["dataset"]["dataset_id"]},
    )
    assert history_response.status_code == 200, history_response.text
    history = history_response.json()
    assert history["count"] == 1
    restored_cycle = history["cycles"][0]
    assert restored_cycle["created_at"] == original_cycle["created_at"]
    assert restored_cycle["rows"] == [{"x": 3.0, "y": 3.0}]
    assert restored_cycle["model"]["type"] == "base"
    assert restored_cycle["acquisition"]["name"] == "EI"
    assert restored_cycle["target_summary"]["y"]["best"] == 3.0
    assert len(history["visualizations"]) == 1


def test_project_archive_defaults_to_latest_model_only(monkeypatch) -> None:
    source_client = TestClient(create_app(include_core_api=False))
    parent = _load_csv(source_client, "experiment.csv", "x,y\n1,1\n2,2\n")
    updated = _load_csv(source_client, "experiment_updated.csv", "x,y\n1,1\n2,2\n3,3\n")
    _record_cycle(source_client, parent, updated, source_run_id="past-run")

    serialized_run_ids: list[str] = []

    def fake_serialize(run_id: str, *, filename: str | None = None) -> tuple[bytes, str]:
        del filename
        serialized_run_ids.append(run_id)
        return f"model:{run_id}".encode(), f"{run_id}.bochan.pt"

    monkeypatch.setattr(
        project_archive_module,
        "serialize_web_model_artifact",
        fake_serialize,
    )
    latest_result = _result(updated)
    latest_result["visualization_run_id"] = "latest-run"
    export_response = source_client.post(
        "/api/v1/experiment-projects/export",
        json={
            "dataset_id": updated["dataset_id"],
            "request": _request(updated),
            "result": latest_result,
        },
    )
    assert export_response.status_code == 200, export_response.text
    assert serialized_run_ids == ["latest-run"]

    archive_bytes = export_response.content
    with ZipFile(BytesIO(archive_bytes)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["model_included"] is True
        assert manifest["latest_model_included"] is True
        assert manifest["past_model_count"] == 0
        assert len(manifest["models"]) == 1
        assert manifest["models"][0]["role"] == "latest"
        assert manifest["models"][0]["run_id"] == "latest-run"
        assert "models/latest.bochan.pt" in archive.namelist()

    untrusted_client = TestClient(create_app(include_core_api=False))
    untrusted_response = untrusted_client.post(
        "/api/v1/model-artifacts/import?trust_pickle=false",
        content=archive_bytes,
        headers={"X-Model-Filename": "experiment.bochan-project.zip"},
    )
    assert untrusted_response.status_code == 200, untrusted_response.text
    untrusted = untrusted_response.json()
    assert untrusted["artifact"]["model_included"] is True
    assert untrusted["artifact"]["model_restored"] is False
    assert untrusted["artifact"]["requires_pickle_trust"] is True
    assert "visualization_run_id" not in untrusted["result"]

    def fake_deserialize(
        content: bytes,
        *,
        trust_pickle: bool,
        map_location: str = "cpu",
    ) -> dict[str, object]:
        del map_location
        assert trust_pickle is True
        run_id = content.decode().split(":", 1)[1]
        return {
            "run_id": run_id,
            "result": {"dataset_id": updated["dataset_id"]},
            "data": [1, 2, 3],
        }

    def fake_restore(
        payload: dict[str, object],
        *,
        dataset_id: str,
        dataset_name: str,
    ) -> tuple[str, dict[str, object], dict[str, object]]:
        restored_result = _result({"dataset_id": dataset_id, "name": dataset_name})
        restored_result["visualization_run_id"] = f"restored-{payload['run_id']}"
        restored_result["metadata"] = {"model_artifact_loaded": True}
        restored_request = _request({"dataset_id": dataset_id, "name": dataset_name})
        return str(restored_result["visualization_run_id"]), restored_result, restored_request

    monkeypatch.setattr(
        project_archive_module,
        "deserialize_web_model_artifact",
        fake_deserialize,
    )
    monkeypatch.setattr(
        project_archive_module,
        "restore_web_model_artifact",
        fake_restore,
    )
    trusted_client = TestClient(create_app(include_core_api=False))
    trusted_response = trusted_client.post(
        "/api/v1/model-artifacts/import?trust_pickle=true",
        content=archive_bytes,
        headers={"X-Model-Filename": "experiment.bochan-project.zip"},
    )
    assert trusted_response.status_code == 200, trusted_response.text
    trusted = trusted_response.json()
    assert trusted["artifact"]["model_restored"] is True
    assert trusted["artifact"]["restored_model_count"] == 1
    assert trusted["artifact"]["active_model_restored"] is True
    assert trusted["result"]["visualization_run_id"] == "restored-latest-run"
    assert "stale_after_data_append" not in trusted["result"]["metadata"]

    serialized_run_ids.clear()
    complete_response = source_client.post(
        "/api/v1/experiment-projects/export",
        json={
            "dataset_id": updated["dataset_id"],
            "request": _request(updated),
            "result": latest_result,
            "include_past_models": True,
        },
    )
    assert complete_response.status_code == 200, complete_response.text
    assert serialized_run_ids == ["latest-run", "past-run"]
    with ZipFile(BytesIO(complete_response.content)) as archive:
        complete_manifest = json.loads(archive.read("manifest.json"))
        assert complete_manifest["past_model_count"] == 1
        assert len(complete_manifest["models"]) == 2


def test_model_import_rejects_invalid_project_zip() -> None:
    client = TestClient(create_app(include_core_api=False))
    response = client.post(
        "/api/v1/model-artifacts/import?trust_pickle=false",
        content=b"not-a-project-or-model",
        headers={"X-Model-Filename": "broken.bochan-project.zip"},
    )
    assert response.status_code == 400
