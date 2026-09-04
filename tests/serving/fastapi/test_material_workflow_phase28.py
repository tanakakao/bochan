"""Phase 28 tests for runtime MLIP relaxation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from bochan.serving.fastapi.app import create_app
from bochan.serving.fastapi.routers import material_workflow


def _client() -> TestClient:
    return TestClient(create_app())


@dataclass
class _FakeRelaxationResult:
    structure: dict[str, Any]
    backend: str
    optimizer: str
    fmax: float
    max_steps: int
    relax_cell: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "backend": self.backend,
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            "relax_cell": self.relax_cell,
            "energy": -1.0,
            "initial_energy": -0.5,
            "forces": [],
            "stress": [[0.0, 0.0, 0.0]] * 3,
            "max_force": 0.0,
            "n_steps": 1,
            "converged": True,
            "model_name": "fake",
            "energy_change": -0.5,
        }


class _FakeRelaxer:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.calls: list[dict[str, Any]] = []

    def relax(self, structure: Any, **kwargs: Any) -> _FakeRelaxationResult:
        self.calls.append({"structure": structure, **kwargs})
        return _FakeRelaxationResult(
            structure=dict(structure),
            backend=self.backend,
            optimizer=str(kwargs["optimizer"]),
            fmax=float(kwargs["fmax"]),
            max_steps=int(kwargs["max_steps"]),
            relax_cell=bool(kwargs["relax_cell"]),
        )


def _structure(symbol: str) -> dict[str, Any]:
    return {
        "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        "coords": [[0.0, 0.0, 0.0]],
        "elements": [symbol],
        "cartesian": False,
    }


def test_execute_relaxation_uses_canonical_backend_and_defaults(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _factory(backend: str, **kwargs: Any) -> _FakeRelaxer:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeRelaxer(backend)

    monkeypatch.setattr(material_workflow, "create_structure_relaxer", _factory)
    response = _client().post(
        "/api/v1/materials/mlip/workflows/execute/relaxation",
        json={
            "backend": "alignn_ff",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "rank",
            "structures": [_structure("Si")],
            "backend_options": {"model_name": "fake-alignn"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["spec"]["backend"] == "alignn-ff"
    assert payload["spec"]["workflow_mode"] == "relax_rank"
    assert payload["relaxation"] == {
        "optimizer": "FIRE",
        "fmax": 0.05,
        "max_steps": 200,
        "relax_cell": False,
    }
    assert payload["results"][0]["backend"] == "alignn-ff"
    assert captured == {
        "backend": "alignn-ff",
        "kwargs": {"model_name": "fake-alignn"},
    }


def test_execute_relaxation_forwards_custom_settings(monkeypatch: Any) -> None:
    relaxer = _FakeRelaxer("mace")
    monkeypatch.setattr(
        material_workflow,
        "create_structure_relaxer",
        lambda backend, **kwargs: relaxer,
    )
    response = _client().post(
        "/api/v1/materials/mlip/workflows/execute/relaxation",
        json={
            "backend": "mace",
            "quantity": "force",
            "model_mode": "residual_gp",
            "workflow_mode": "bo",
            "structures": [_structure("Si"), _structure("Ge")],
            "relaxation": {
                "optimizer": "LBFGS",
                "fmax": 0.02,
                "max_steps": 500,
                "relax_cell": True,
            },
        },
    )

    assert response.status_code == 200
    assert len(relaxer.calls) == 2
    for call in relaxer.calls:
        assert call["optimizer"] == "LBFGS"
        assert call["fmax"] == 0.02
        assert call["max_steps"] == 500
        assert call["relax_cell"] is True


def test_execute_relaxation_rejects_model_only() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/execute/relaxation",
        json={
            "backend": "mace",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "model_only",
            "structures": [_structure("Si")],
        },
    )

    assert response.status_code == 422
    assert "do not have a relaxation execution stage" in response.json()["detail"]


def test_execute_relaxation_maps_missing_optional_dependency_to_503(monkeypatch: Any) -> None:
    def _missing(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("optional MLIP package is missing")

    monkeypatch.setattr(material_workflow, "create_structure_relaxer", _missing)
    response = _client().post(
        "/api/v1/materials/mlip/workflows/execute/relaxation",
        json={
            "backend": "mace",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "relax_rank",
            "structures": [_structure("Si")],
        },
    )

    assert response.status_code == 503
    assert "optional MLIP package is missing" in response.json()["detail"]


def test_execute_relaxation_requires_non_empty_structures() -> None:
    response = _client().post(
        "/api/v1/materials/mlip/workflows/execute/relaxation",
        json={
            "backend": "mace",
            "quantity": "energy",
            "model_mode": "direct",
            "workflow_mode": "relax_rank",
            "structures": [],
        },
    )

    assert response.status_code == 422


def test_execute_relaxation_route_is_in_openapi() -> None:
    document = _client().get("/openapi.json").json()
    assert "/api/v1/materials/mlip/workflows/execute/relaxation" in document["paths"]
