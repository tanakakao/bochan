"""Phase 25 tests for MLIP capability discovery endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bochan.serving.fastapi.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_material_capability_catalog_endpoint() -> None:
    response = _client().get("/api/v1/materials/mlip/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["quantities"] == ["energy", "force", "stress"]
    assert payload["model_modes"] == ["direct", "residual_gp"]
    assert payload["workflow_modes"] == [
        "model_only",
        "relax_rank",
        "relax_acquisition",
    ]
    assert [item["backend"] for item in payload["backends"]] == [
        "mace",
        "chgnet",
        "m3gnet",
        "alignn-ff",
    ]


def test_material_capability_backend_endpoint_normalizes_alias() -> None:
    response = _client().get("/api/v1/materials/mlip/capabilities/alignn_ff")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "alignn-ff"
    assert payload["direct_quantities"] == ["energy", "force", "stress"]
    assert payload["residual_quantities"] == ["energy", "force", "stress"]
    assert payload["residual_requires_structure_graphs"] is True
    assert payload["force_fixed_topology"] is True
    assert payload["stress_components"] == 9


def test_material_capability_backend_endpoint_reports_mace_constraints() -> None:
    response = _client().get("/api/v1/materials/mlip/capabilities/mace")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "mace"
    assert payload["residual_requires_structure_graphs"] is False
    assert payload["supports_relaxation"] is True
    assert payload["supports_relax_rank"] is True
    assert payload["supports_relax_acquisition"] is True


def test_material_capability_backend_endpoint_returns_404_for_unknown_backend() -> None:
    response = _client().get("/api/v1/materials/mlip/capabilities/unknown")

    assert response.status_code == 404
    assert "Unsupported material backend" in response.json()["detail"]


def test_material_capability_routes_are_in_openapi() -> None:
    document = _client().get("/openapi.json").json()

    assert "/api/v1/materials/mlip/capabilities" in document["paths"]
    assert "/api/v1/materials/mlip/capabilities/{backend}" in document["paths"]
