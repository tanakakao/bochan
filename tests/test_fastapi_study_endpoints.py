"""FastAPI Study endpoint tests for the canonical HTTP contract."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.dependencies import get_study_store
from bochan.serving.fastapi.stores import InMemoryStudyStore


def test_study_create_uses_canonical_config_contract() -> None:
    store = InMemoryStudyStore()
    app = create_app(title="study contract test")
    app.dependency_overrides[get_study_store] = lambda: store
    client = TestClient(app)
    response = client.post(
        "/api/v1/studies",
        json={
            "bounds": [[0.0], [1.0]],
            "fit_config": {"method": "auto", "beta": 0.5},
            "acquisition_config": {
                "name": "UCB",
                "objective_config": {"direction": "minimize"},
            },
            "optimize_config": {"q": 2},
        },
    )
    assert response.status_code == 200, response.text
    study = store.get(response.json()["study_id"])
    assert study.fit_config.beta == pytest.approx(0.5)
    assert study.acq_config.name == "UCB"
    assert study.acq_config.objective_config.direction == "minimize"
    assert study.opt_config.q == 2


def test_study_create_rejects_legacy_config_aliases() -> None:
    client = TestClient(create_app(title="study alias rejection test"))
    legacy_payloads = [
        {"fit_config": {"fit_method": "auto"}},
        {"acquisition_config": {"acq_name": "UCB"}},
        {"acquisition_config": {"name": "UCB", "objective_direction": "minimize"}},
        {"acq_config": {"name": "UCB"}},
        {"opt_config": {"q": 1}},
    ]
    for legacy in legacy_payloads:
        payload = {"bounds": [[0.0], [1.0]], **legacy}
        response = client.post("/api/v1/studies", json=payload)
        assert response.status_code == 422, response.text
