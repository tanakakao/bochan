"""Tests for Web experiment-cycle history and objective-progress plots."""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pandas")
pytest.importorskip("plotly")

from fastapi.testclient import TestClient

from bochan.serving.webapp.app import create_app


def _upload_csv(client: TestClient, name: str, csv_text: str) -> dict:
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    response = client.post(
        "/api/v1/datasets",
        json={
            "source_type": "csv",
            "name": name,
            "content_base64": encoded,
            "encoding": "utf-8",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _cycle_payload(
    *,
    parent: dict,
    updated: dict,
    value: float,
    model_type: str,
    acquisition: str,
) -> dict:
    return {
        "parent_dataset_id": parent["dataset_id"],
        "dataset_id": updated["dataset_id"],
        "dataset_name": updated["name"],
        "source_run_id": f"run-{model_type}-{acquisition}",
        "append_mode": "manual",
        "n_rows_before": parent["profile"]["n_rows"],
        "n_rows_after": updated["profile"]["n_rows"],
        "rows": [{"x": 0.5, "y": value}],
        "feature_columns": ["x"],
        "target_columns": ["y"],
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
        "model": {"type": model_type, "n_train": parent["profile"]["n_rows"]},
        "acquisition": {"name": acquisition, "family": "bayesian_optimization"},
        "optimizer": {"backend": "optimize_acqf", "q": 1},
        "best_observed_before": {"y": 2.0},
        "candidate_count": 1,
    }


def test_experiment_history_tracks_dataset_lineage_and_builds_plot() -> None:
    client = TestClient(create_app())
    initial = _upload_csv(client, "initial.csv", "x,y\n0,1\n1,2\n")
    cycle_one_data = _upload_csv(client, "cycle1.csv", "x,y\n0,1\n1,2\n0.5,3\n")
    cycle_two_data = _upload_csv(client, "cycle2.csv", "x,y\n0,1\n1,2\n0.5,3\n0.7,4\n")

    first = client.post(
        "/api/v1/experiment-cycles",
        json=_cycle_payload(
            parent=initial,
            updated=cycle_one_data,
            value=3.0,
            model_type="base",
            acquisition="EI",
        ),
    )
    assert first.status_code == 200, first.text
    assert first.json()["cycle"]["cycle_number"] == 1
    assert first.json()["cycle"]["target_summary"]["y"]["best"] == 3.0

    second = client.post(
        "/api/v1/experiment-cycles",
        json=_cycle_payload(
            parent=cycle_one_data,
            updated=cycle_two_data,
            value=4.0,
            model_type="saas",
            acquisition="UCB",
        ),
    )
    assert second.status_code == 200, second.text
    assert second.json()["cycle"]["cycle_number"] == 2

    response = client.get(
        "/api/v1/experiment-cycles",
        params={"dataset_id": cycle_two_data["dataset_id"]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 3
    assert payload["targets"] == ["y"]
    assert [cycle["cycle_number"] for cycle in payload["cycles"]] == [0, 1, 2]
    assert [cycle["model"]["type"] for cycle in payload["cycles"]] == ["initial_data", "base", "saas"]
    assert [cycle["acquisition"].get("name") for cycle in payload["cycles"]] == [None, "EI", "UCB"]
    assert payload["cycles"][2]["rows"] == [{"x": 0.5, "y": 4.0}]

    visualization = payload["visualizations"][0]
    assert visualization["target"] == "y"
    assert visualization["figure"]["data"][0]["name"] == "サイクル内ベスト"
    assert visualization["figure"]["data"][0]["y"] == [2.0, 3.0, 4.0]
    assert visualization["figure"]["data"][2]["name"] == "累積ベスト"
    assert visualization["figure"]["data"][2]["y"] == [2.0, 3.0, 4.0]
    assert visualization["figure"]["data"][3]["name"] == "各データ"
    assert visualization["figure"]["data"][3]["y"] == [1.0, 2.0, 3.0, 4.0]
    assert visualization["figure"]["data"][3]["x"] == [-0.18, 0.18, 1.0, 2.0]


def test_experiment_history_rejects_dataset_row_count_mismatch() -> None:
    client = TestClient(create_app())
    initial = _upload_csv(client, "initial.csv", "x,y\n0,1\n")
    updated = _upload_csv(client, "updated.csv", "x,y\n0,1\n1,2\n")
    payload = _cycle_payload(
        parent=initial,
        updated=updated,
        value=2.0,
        model_type="base",
        acquisition="EI",
    )
    payload["n_rows_after"] = 999

    response = client.post("/api/v1/experiment-cycles", json=payload)

    assert response.status_code == 400
    assert "n_rows_after" in response.json()["detail"]
