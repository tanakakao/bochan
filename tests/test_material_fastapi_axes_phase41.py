from __future__ import annotations

from fastapi.testclient import TestClient

from bochan.serving.fastapi.app import create_app

client = TestClient(create_app())


def test_material_model_axes_catalog_lists_registered_families() -> None:
    response = client.get("/api/v1/materials/models/capabilities")

    assert response.status_code == 200
    payload = response.json()
    families = {item["family"] for item in payload["families"]}
    assert families == {"crabnet", "roost", "alignn", "chgnet", "m3gnet", "mace"}
    assert all(item["fidelity_route_implemented"] is False for item in payload["families"])


def test_material_model_axes_capabilities_for_family() -> None:
    response = client.get("/api/v1/materials/models/capabilities/roost")

    assert response.status_code == 200
    payload = response.json()
    assert payload["family"] == "roost"
    assert payload["domain"] == "composition"
    assert payload["implemented_routes"] == ["wide_output", "explicit_task"]


def test_validate_continuous_explicit_task_request() -> None:
    response = client.post(
        "/api/v1/materials/models/validate",
        json={
            "family": "RoOsT",
            "kind": "deep-kernel",
            "input_mode": "continuous",
            "output_mode": "scalar",
            "task_mode": "task-index",
            "task": {"task_feature": -1, "all_tasks": [0, 1, 2]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["family"] == "roost"
    assert payload["kind"] == "dkl"
    assert payload["task_mode"] == "explicit"
    assert payload["route"] == "explicit_task"
    assert payload["task"]["all_tasks"] == [0, 1, 2]


def test_validate_mixed_requires_cat_dims() -> None:
    response = client.post(
        "/api/v1/materials/models/validate",
        json={"family": "mace", "input_mode": "mixed"},
    )

    assert response.status_code == 422
    assert "cat_dims is required" in response.json()["detail"]


def test_validate_rejects_task_payload_without_explicit_task_mode() -> None:
    response = client.post(
        "/api/v1/materials/models/validate",
        json={
            "family": "mace",
            "task_mode": "none",
            "task": {"task_feature": -1, "all_tasks": [0, 1]},
        },
    )

    assert response.status_code == 422
    assert "task is only valid" in response.json()["detail"]


def test_task_fixed_features_resolves_negative_task_feature() -> None:
    response = client.post(
        "/api/v1/materials/models/task-fixed-features",
        json={
            "model": {
                "family": "crabnet",
                "task_mode": "explicit",
                "task": {"task_feature": -1, "all_tasks": [0, 2]},
            },
            "target_task": 2,
            "input_dim": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["fixed_features"] == {"4": 2.0}


def test_task_fixed_features_rejects_unknown_target_task() -> None:
    response = client.post(
        "/api/v1/materials/models/task-fixed-features",
        json={
            "model": {
                "family": "crabnet",
                "task_mode": "explicit",
                "task": {"task_feature": -1, "all_tasks": [0, 1]},
            },
            "target_task": 3,
            "input_dim": 4,
        },
    )

    assert response.status_code == 422
    assert "not included in all_tasks" in response.json()["detail"]


def test_reserved_fidelity_is_exposed_but_not_claimed_implemented() -> None:
    response = client.post(
        "/api/v1/materials/models/validate",
        json={"family": "chgnet", "fidelity_mode": "continuous"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "fidelity"
    assert payload["implemented"] is False
