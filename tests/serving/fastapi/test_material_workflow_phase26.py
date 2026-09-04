"""Phase 26 tests for MLIP workflow validation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bochan.serving.fastapi.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_material_workflow_validation_normalizes_aliases() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/validate",
        json={
            "backend": "alignn_ff",
            "quantity": "energy",
            "model_mode": "residual-gp",
            "workflow_mode": "bo",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["spec"] == {
        "backend": "alignn-ff",
        "quantity": "energy",
        "model_mode": "residual_gp",
        "workflow_mode": "relax_acquisition",
    }
    assert payload["requirements"] == [
        "structures",
        "train_X",
        "train_Y",
        "structure_graphs",
    ]


def test_material_workflow_validation_reports_force_topology_requirement() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/validate",
        json={
            "backend": "mace",
            "quantity": "force",
            "model_mode": "direct",
        },
    )

    assert response.status_code == 200
    assert response.json()["requirements"] == ["structures", "fixed_atom_count"]


def test_material_workflow_validation_rejects_unknown_backend() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/validate",
        json={
            "backend": "unknown",
            "quantity": "energy",
            "model_mode": "direct",
        },
    )

    assert response.status_code == 400
    assert "Unsupported material backend" in response.json()["detail"]


def test_material_workflow_validation_rejects_unknown_workflow_mode() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/validate",
        json={
            "backend": "m3gnet",
            "quantity": "stress",
            "model_mode": "residual_gp",
            "workflow_mode": "unknown",
        },
    )

    assert response.status_code == 400
    assert "Unsupported material workflow mode" in response.json()["detail"]


def test_material_workflow_validation_route_is_in_openapi() -> None:
    document = _client().get("/openapi.json").json()

    path = "/api/v1/materials/mlip/workflows/validate"
    assert path in document["paths"]
    assert "post" in document["paths"][path]
