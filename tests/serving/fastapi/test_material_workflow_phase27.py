"""Phase 27 tests for dependency-light MLIP relaxation workflow configuration."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bochan.serving.fastapi.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_configure_relaxation_workflow_fills_common_defaults() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/configure",
        json={
            "backend": "mace",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "rank",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["spec"] == {
        "backend": "mace",
        "quantity": "energy",
        "model_mode": "direct",
        "workflow_mode": "relax_rank",
    }
    assert payload["requirements"] == ["structures"]
    assert payload["relaxation"] == {
        "optimizer": "FIRE",
        "fmax": 0.05,
        "max_steps": 200,
        "relax_cell": False,
    }


def test_configure_relaxation_workflow_preserves_custom_settings_and_aliases() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/configure",
        json={
            "backend": "alignn_ff",
            "quantity": "force",
            "model_mode": "residual-gp",
            "workflow_mode": "bo",
            "relaxation": {
                "optimizer": "LBFGS",
                "fmax": 0.02,
                "max_steps": 500,
                "relax_cell": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["spec"] == {
        "backend": "alignn-ff",
        "quantity": "force",
        "model_mode": "residual_gp",
        "workflow_mode": "relax_acquisition",
    }
    assert payload["requirements"] == [
        "structures",
        "train_X",
        "train_Y",
        "structure_graphs",
        "fixed_atom_count",
    ]
    assert payload["relaxation"] == {
        "optimizer": "LBFGS",
        "fmax": 0.02,
        "max_steps": 500,
        "relax_cell": True,
    }


def test_configure_model_only_workflow_has_no_relaxation() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/configure",
        json={
            "backend": "chgnet",
            "quantity": "stress",
            "model_mode": "direct",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["spec"]["workflow_mode"] == "model_only"
    assert payload["relaxation"] is None


def test_configure_model_only_rejects_relaxation_settings() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/configure",
        json={
            "backend": "m3gnet",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "model_only",
            "relaxation": {"fmax": 0.03},
        },
    )

    assert response.status_code == 400
    assert "model_only workflows do not accept relaxation settings" in response.json()[
        "detail"
    ]


def test_configure_rejects_invalid_relaxation_values_via_schema() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/configure",
        json={
            "backend": "mace",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "rank",
            "relaxation": {"fmax": 0.0, "max_steps": 0},
        },
    )

    assert response.status_code == 422


def test_configure_rejects_unknown_optimizer_via_schema() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/configure",
        json={
            "backend": "mace",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "rank",
            "relaxation": {"optimizer": "Adam"},
        },
    )

    assert response.status_code == 422


def test_configure_route_is_in_openapi() -> None:
    document = _client().get("/openapi.json").json()

    assert "/api/v1/materials/mlip/workflows/configure" in document["paths"]
