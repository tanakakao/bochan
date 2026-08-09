from __future__ import annotations

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("pandas")

from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.dependencies import get_tabular_optimizer_store
from bochan.serving.fastapi.stores import InMemoryTabularOptimizerStore
from bochan.tabular import ObservationTabularDataset


@pytest.fixture
def client_and_store():
    store = InMemoryTabularOptimizerStore()
    app = create_app(title="observation tabular test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: store
    return TestClient(app), store


def test_fastapi_partial_multitask_keeps_unobserved_targets(client_and_store) -> None:
    client, store = client_and_store
    response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": [
                {"x": 0.0, "y1": 1.0, "y2": None},
                {"x": 0.2, "y1": 2.0, "y2": 2.5},
                {"x": 0.4, "y1": None, "y2": 3.0},
                {"x": 0.6, "y1": 4.0, "y2": 4.0},
                {"x": 0.8, "y1": 5.0, "y2": 5.0},
                {"x": 1.0, "y1": 6.0, "y2": 6.0},
            ],
            "input_cols": ["x"],
            "target_cols": ["y1", "y2"],
            "target_missing_strategy": "keep",
            "model_config": {
                "task_type": "regression",
                "model_type": "multitask",
                "outcome_transform": False,
            },
            "fit_config": {"skip_fit": True},
        },
    )

    assert response.status_code == 200, response.text
    optimizer = store.get(response.json()["model_id"])
    assert isinstance(optimizer.dataset, ObservationTabularDataset)
    assert torch.isnan(optimizer.dataset.Y).sum().item() == 2
    assert torch.isnan(optimizer.bo.model.train_Y_wide).sum().item() == 2
    assert optimizer.bo.observations.report()["observed_per_output"] == [5, 5]


def test_fastapi_explicit_failure_and_pending_states_train_success_model(
    client_and_store,
) -> None:
    client, store = client_and_store
    response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": [
                {"x": 0.0, "strength": 1.0, "status": "success"},
                {"x": 0.25, "strength": None, "status": "failed"},
                {"x": 0.5, "strength": 2.0, "status": "success"},
                {"x": 0.75, "strength": None, "status": "pending"},
                {"x": 1.0, "strength": 3.0, "status": "success"},
            ],
            "input_cols": ["x"],
            "target_cols": ["strength"],
            "target_missing_strategy": "keep",
            "experiment_status_col": "status",
            "experiment_failure": {
                "fit_config": {"skip_fit": True},
                "min_success_probability": 0.7,
            },
            "model_config": {
                "task_type": "regression",
                "model_type": "base",
                "outcome_transform": False,
            },
            "fit_config": {"skip_fit": True},
        },
    )

    assert response.status_code == 200, response.text
    optimizer = store.get(response.json()["model_id"])
    observations = optimizer.bo.observations
    assert observations.failed_mask.tolist() == [False, True, False, False, False]
    assert observations.pending_mask.tolist() == [False, False, False, True, False]
    assert optimizer.bo.failure_bundle is not None
    torch.testing.assert_close(
        optimizer.bo.failure_bundle.train_Y.squeeze(-1),
        torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=optimizer.dataset.Y.dtype),
    )
    metadata = response.json()["metadata"]
    assert metadata["experiment_failure_model"]["enabled"] is True
    assert metadata["experiment_failure_model"]["n_failed"] == 1


def test_fastapi_failure_config_requires_explicit_status_column(client_and_store) -> None:
    client, _ = client_and_store
    response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
            "input_cols": ["x"],
            "target_cols": ["y"],
            "experiment_failure": {"fit_config": {"skip_fit": True}},
            "model_config": {"task_type": "regression", "model_type": "base"},
            "fit_config": {"skip_fit": True},
        },
    )

    assert response.status_code == 422
    assert "experiment_status_col" in response.text


def test_fastapi_status_column_cannot_be_an_input(client_and_store) -> None:
    client, _ = client_and_store
    response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": [
                {"x": 0.0, "y": 1.0, "status": "success"},
                {"x": 1.0, "y": None, "status": "failed"},
            ],
            "input_cols": ["x", "status"],
            "target_cols": ["y"],
            "experiment_status_col": "status",
            "model_config": {"task_type": "regression", "model_type": "base"},
            "fit_config": {"skip_fit": True},
        },
    )

    assert response.status_code == 422
    assert "must not be included in input_cols" in response.text
