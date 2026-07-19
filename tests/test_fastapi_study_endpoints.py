"""FastAPI coverage for stateful BochanStudy workflows."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.dependencies import get_study_store
from bochan.serving.fastapi.stores import InMemoryStudyStore


@pytest.fixture
def client_and_store():
    store = InMemoryStudyStore()
    app = create_app(title="study test")
    app.dependency_overrides[get_study_store] = lambda: store
    return TestClient(app), store


def _create_study(client: TestClient, **overrides) -> str:
    payload = {
        "bounds": [[0.0, 0.0], [1.0, 1.0]],
        "n_initial_random": 10,
        "metadata": {"feature_names": ["temperature", "pressure"]},
    }
    payload.update(overrides)
    response = client.post("/api/v1/studies", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["study_id"]


def test_study_ask_tell_best_history_and_restore(client_and_store) -> None:
    client, _ = client_and_store
    study_id = _create_study(client)

    ask_response = client.post(
        f"/api/v1/studies/{study_id}/ask",
        json={"q": 2, "mark_running": True},
    )
    assert ask_response.status_code == 200, ask_response.text
    ask_body = ask_response.json()
    assert len(ask_body["trial_ids"]) == 2
    assert len(ask_body["candidates"]) == 2

    tell_response = client.post(
        f"/api/v1/studies/{study_id}/tell",
        json={
            "trial_ids": ask_body["trial_ids"],
            "values": [0.25, 0.75],
            "metadata": [{"cycle": 0}, {"cycle": 1}],
        },
    )
    assert tell_response.status_code == 200, tell_response.text
    assert tell_response.json()["n_completed"] == 2
    assert tell_response.json()["n_pending"] == 0

    best_response = client.get(f"/api/v1/studies/{study_id}/best")
    assert best_response.status_code == 200, best_response.text
    best = best_response.json()["result"]
    assert best["value"] == pytest.approx(0.75)
    assert set(best["params"]) == {"temperature", "pressure"}
    assert len(best["x"]) == 2

    history_response = client.get(f"/api/v1/studies/{study_id}/history")
    assert history_response.status_code == 200, history_response.text
    history = history_response.json()
    assert history["direction"] == "maximize"
    assert [row["best_value"] for row in history["records"]] == [0.25, 0.75]
    assert [row["is_best"] for row in history["records"]] == [True, True]

    snapshot_response = client.get(f"/api/v1/studies/{study_id}/snapshot")
    assert snapshot_response.status_code == 200, snapshot_response.text
    snapshot = snapshot_response.json()["snapshot"]

    restore_response = client.post(
        "/api/v1/studies/restore",
        json={
            "snapshot": snapshot,
            "bounds": [[0.0, 0.0], [1.0, 1.0]],
            "n_initial_random": 10,
        },
    )
    assert restore_response.status_code == 200, restore_response.text
    restored_id = restore_response.json()["study_id"]
    assert restore_response.json()["n_completed"] == 2
    assert client.get(f"/api/v1/studies/{restored_id}/best").json()["result"]["value"] == pytest.approx(0.75)

    list_response = client.get("/api/v1/studies")
    assert sorted(list_response.json()["study_ids"]) == sorted([study_id, restored_id])


def test_study_observations_pareto_and_trial_states(client_and_store) -> None:
    client, _ = client_and_store
    study_id = _create_study(
        client,
        bounds=[[0.0], [2.0]],
        model_config={"task_type": "multi_objective", "model_type": "base"},
        metadata={"feature_names": ["x"]},
    )

    observations = client.post(
        f"/api/v1/studies/{study_id}/observations",
        json={
            "X": [[0.0], [1.0], [2.0]],
            "Y": [[1.0, 5.0], [2.0, 4.0], [3.0, 6.0]],
        },
    )
    assert observations.status_code == 200, observations.text
    assert observations.json()["n_completed"] == 3

    pareto_response = client.post(
        f"/api/v1/studies/{study_id}/pareto",
        json={
            "output_indices": [0, 1],
            "directions": ["maximize", "minimize"],
        },
    )
    assert pareto_response.status_code == 200, pareto_response.text
    pareto = pareto_response.json()
    assert [trial["trial_id"] for trial in pareto["pareto_trials"]] == [1, 2]
    assert [trial["is_pareto"] for trial in pareto["trials"]] == [False, True, True]

    best_cost = client.get(
        f"/api/v1/studies/{study_id}/best",
        params={"output_index": 1, "direction": "minimize"},
    )
    assert best_cost.status_code == 200, best_cost.text
    assert best_cost.json()["result"]["trial_id"] == 1
    assert best_cost.json()["result"]["value"] == pytest.approx(4.0)

    state_study_id = _create_study(client, bounds=[[0.0], [1.0]])
    ask_response = client.post(
        f"/api/v1/studies/{state_study_id}/ask",
        json={"q": 1},
    )
    trial_id = ask_response.json()["trial_ids"][0]
    running = client.post(
        f"/api/v1/studies/{state_study_id}/trials/running",
        json={"trial_ids": [trial_id]},
    )
    assert running.status_code == 200, running.text
    failed = client.post(
        f"/api/v1/studies/{state_study_id}/trials/failed",
        json={"trial_ids": [trial_id], "reason": "simulation failed"},
    )
    assert failed.status_code == 200, failed.text
    assert failed.json()["n_failed"] == 1
    trials = client.get(f"/api/v1/studies/{state_study_id}/trials").json()["trials"]
    assert trials[0]["state"] == "FAILED"
    assert trials[0]["metadata"]["failure_reason"] == "simulation failed"

    delete_response = client.delete(f"/api/v1/studies/{state_study_id}")
    assert delete_response.status_code == 200
    assert client.get(f"/api/v1/studies/{state_study_id}").status_code == 404
