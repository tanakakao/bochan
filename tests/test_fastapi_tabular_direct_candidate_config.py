"""FastAPI parity tests for direct TabularBayesianOptimizer candidate fields."""

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
    app = create_app(title="tabular direct candidate test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: store
    return TestClient(app), store


class _CapturingOptimizer:
    def __init__(self) -> None:
        self.candidate_kwargs = None

    def candidate(self, **kwargs):
        self.candidate_kwargs = kwargs
        return pd.DataFrame([{"x1": 0.25, "x2": 0.75}]), 0.5

    def ask(self, **kwargs):
        return self.candidate(**kwargs)


def test_tabular_candidate_forwards_explicit_direct_fields(client_and_store) -> None:
    client, store = client_and_store
    optimizer = _CapturingOptimizer()
    model_id = store.add(optimizer)

    response = client.post(
        f"/api/v1/tabular/models/{model_id}/candidates",
        json={
            "objective_mode": "scalar",
            "objective_output": "property",
            "objective_direction": "maximize",
            "acquisition_config": {"name": "ei"},
            "outcome_constraint_config": {
                "constraints": [
                    {
                        "output": "quality",
                        "target_class": "b",
                        "threshold": 0.75,
                        "sense": "ge",
                    }
                ]
            },
            "optimize_config": {
                "q": 1,
                "optimizer": "optimize_acqf",
                "num_restarts": 2,
                "raw_samples": 4,
            },
        },
    )

    assert response.status_code == 200, response.text
    assert optimizer.candidate_kwargs is not None
    assert optimizer.candidate_kwargs["objective_mode"] == "scalar"
    assert optimizer.candidate_kwargs["objective_output"] == "property"
    assert optimizer.candidate_kwargs["objective_direction"] == "maximize"
    constraint_config = optimizer.candidate_kwargs["outcome_constraint_config"]
    assert constraint_config["constraints"][0]["target_class"] == "b"


def test_fastapi_hybrid_ordinal_acquisitions_accept_direct_objective_and_constraint(
    client_and_store,
) -> None:
    client, _ = client_and_store
    records = []
    labels = ["a", "b", "c"]
    for index in range(18):
        records.append(
            {
                "x1": 0.05 + 0.04 * index,
                "x2": float(index % 5) / 4.0,
                "property": 0.1 + 0.04 * index,
                "y_ord_str": labels[index % 3],
            }
        )

    fit_response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": records,
            "input_cols": ["x1", "x2"],
            "target_cols": ["property", "y_ord_str"],
            "model_config": {
                "task_type": "hybrid",
                "model_type": "base",
                "input_transform_config": {
                    "perturbation": False,
                    "n_w": 4,
                    "std": 0.1,
                },
            },
            "multi_output_config": {
                "output_configs": [
                    {
                        "task_type": "regression",
                        "model_type": "base",
                        "name": "property",
                    },
                    {
                        "task_type": "ordinal",
                        "model_type": "base",
                        "name": "y_ord_str",
                        "ordered_categories": ["a", "b", "c"],
                    },
                ],
                "use_hybrid": True,
            },
            "fit_config": {"maxiter": 8},
        },
    )
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]

    acquisition_names = [
        "ei",
        "pi",
        "ucb",
        "bald",
        "entropy",
        "variance",
        "straddle",
        "icu",
    ]
    for acquisition_name in acquisition_names:
        candidate_response = client.post(
            f"/api/v1/tabular/models/{model_id}/candidates",
            json={
                "objective_mode": "scalar",
                "objective_output": "property",
                "objective_direction": "maximize",
                "acquisition_config": {"name": acquisition_name},
                "outcome_constraint_config": {
                    "constraints": [
                        {
                            "output": "y_ord_str",
                            "target_class": "b",
                            "threshold": 0.75,
                            "sense": "ge",
                        }
                    ]
                },
                "optimize_config": {
                    "q": 2,
                    "optimizer": "optimize_acqf",
                    "num_restarts": 2,
                    "raw_samples": 4,
                },
            },
        )

        assert candidate_response.status_code == 200, (
            acquisition_name,
            candidate_response.text,
        )
        body = candidate_response.json()
        assert len(body["candidates"]) == 2
        assert body["columns"] == ["x1", "x2"]
