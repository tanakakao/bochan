"""FastAPI coverage for TabularBayesianOptimizer endpoints."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")

from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.dependencies import get_tabular_optimizer_store
from bochan.serving.fastapi.stores import InMemoryTabularOptimizerStore


@pytest.fixture
def client_and_store():
    store = InMemoryTabularOptimizerStore()
    app = create_app(title="tabular test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: store
    return TestClient(app), store


def test_fit_tabular_model_encodes_string_categories(client_and_store) -> None:
    client, _ = client_and_store
    response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": [
                {"material": "A", "temperature": 100.0, "property": 0.1},
                {"material": "B", "temperature": 120.0, "property": 0.7},
                {"material": "A", "temperature": 140.0, "property": 0.4},
                {"material": "B", "temperature": 160.0, "property": 0.9},
            ],
            "input_cols": ["material", "temperature"],
            "target_cols": ["property"],
            "categorical_cols": ["material"],
            "model_config": {"task_type": "regression", "model_type": "base"},
            "fit_config": {"skip_fit": True},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["feature_names"] == ["material", "temperature"]
    assert body["target_names"] == ["property"]
    assert body["categorical_cols"] == ["material"]
    assert body["category_maps"]["material"] == {"A": 0, "B": 1}
    assert body["n_train"] == 4

    list_response = client.get("/api/v1/tabular/models")
    assert list_response.json()["model_ids"] == [body["model_id"]]

    delete_response = client.delete(f"/api/v1/tabular/models/{body['model_id']}")
    assert delete_response.status_code == 200
    assert client.get("/api/v1/tabular/models").json()["model_ids"] == []


class _FakeTabularOptimizer:
    def predict(self, data, **kwargs):
        assert list(data.columns) == ["material", "temperature"]
        return pd.DataFrame(
            {
                "material": data["material"],
                "temperature": data["temperature"],
                "property_mean": [0.5] * len(data),
            }
        )

    def candidate(self, **kwargs):
        return (
            pd.DataFrame(
                [
                    {
                        "material": "B",
                        "temperature": 150.0,
                    }
                ]
            ),
            1.25,
        )

    def ask(self, **kwargs):
        return self.candidate(**kwargs)


def test_tabular_predict_and_candidate_return_records(client_and_store) -> None:
    client, store = client_and_store
    model_id = store.add(_FakeTabularOptimizer())

    predict_response = client.post(
        f"/api/v1/tabular/models/{model_id}/predict",
        json={
            "data": [{"material": "A", "temperature": 125.0}],
            "include_input": True,
        },
    )
    assert predict_response.status_code == 200, predict_response.text
    assert predict_response.json()["records"] == [
        {"material": "A", "temperature": 125.0, "property_mean": 0.5}
    ]

    candidate_response = client.post(
        f"/api/v1/tabular/models/{model_id}/candidates",
        json={
            "acquisition_config": {"name": "ehvi"},
            "optimize_config": {"q": 1, "optimizer": "optimize_acqf"},
        },
    )
    assert candidate_response.status_code == 200, candidate_response.text
    assert candidate_response.json()["candidates"] == [
        {"material": "B", "temperature": 150.0}
    ]
    assert candidate_response.json()["acq_value"] == pytest.approx(1.25)


def test_fastapi_multiclass_pca_nparego_matches_tabular_api(client_and_store) -> None:
    client, store = client_and_store
    records = []
    for index in range(18):
        class_index = index % 3
        records.append(
            {
                "raw material 1": 0.1 + 0.02 * index,
                "raw material 2": 0.8 - 0.015 * index,
                "raw material 3": 0.1 + 0.005 * (index % 5),
                "temperature": 90.0 + 3.0 * index,
                "time": 5.0 + float(index % 6),
                "y_ord_str": ["a", "b", "c"][class_index],
                "y_ord_str2": class_index,
            }
        )

    fit_response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": records,
            "input_cols": [
                "raw material 1",
                "raw material 2",
                "raw material 3",
                "temperature",
                "time",
            ],
            "target_cols": ["y_ord_str", "y_ord_str2"],
            "model_config": {
                "task_type": "multiclass",
                "model_type": "pca",
                "input_transform_config": {
                    "perturbation": True,
                    "n_w": 4,
                    "std": 0.1,
                },
            },
            "fit_config": {"maxiter": 8},
        },
    )
    assert fit_response.status_code == 200, fit_response.text

    model_id = fit_response.json()["model_id"]
    optimizer = store.get(model_id)
    direct_candidates, _ = optimizer.candidate(
        acq_config={"name": "nparego"},
        opt_config={
            "q": 2,
            "optimizer": "optimize_acqf",
            "num_restarts": 2,
            "raw_samples": 4,
        },
    )
    assert len(direct_candidates) == 2

    candidate_response = client.post(
        f"/api/v1/tabular/models/{model_id}/candidates",
        json={
            "acquisition_config": {"name": "nparego"},
            "optimize_config": {
                "q": 2,
                "optimizer": "optimize_acqf",
                "num_restarts": 2,
                "raw_samples": 4,
            },
        },
    )

    assert candidate_response.status_code == 200, candidate_response.text
    body = candidate_response.json()
    assert len(body["candidates"]) == 2
    assert body["columns"] == [
        "raw material 1",
        "raw material 2",
        "raw material 3",
        "temperature",
        "time",
    ]
