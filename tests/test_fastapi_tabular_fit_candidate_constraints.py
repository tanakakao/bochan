"""FastAPI tests for candidate-time tabular optimization constraints."""

# ruff: noqa: E402

from __future__ import annotations

import math

import pytest

pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")

from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(title="tabular candidate constraints test"))


def _hybrid_records() -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    x1_values = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
    x2_values = [0.0, 0.1, 0.2, 0.4]
    x3_values = [0.0, 0.3, 0.6, 0.9]
    temperature_values = [50.0, 65.0, 80.0, 100.0]
    time_values = [10.0, 40.0, 80.0, 120.0]
    for index in range(24):
        x1 = x1_values[index % len(x1_values)]
        x2 = x2_values[(index * 2) % len(x2_values)]
        x3 = x3_values[(index * 3) % len(x3_values)]
        temperature = temperature_values[index % len(temperature_values)]
        time = time_values[(index * 3) % len(time_values)]
        property_value = 0.25 * x1 + 0.35 * x2 + 0.4 * x3 + 0.001 * temperature
        records.append(
            {
                "raw material 1": x1,
                "raw material 2": x2,
                "raw material 3": x3,
                "temperature": temperature,
                "time": time,
                "property": property_value,
                "y_cat_str": "b" if property_value > 0.35 else "a",
            }
        )
    return records


def _constraints() -> list[list[object]]:
    return [
        [
            ["raw material 1", "raw material 2", "raw material 3"],
            [1.0, 1.0, 1.0],
            "=",
            1.0,
        ],
        [
            ["temperature", "time"],
            [1.0, 1.0],
            ">=",
            100.0,
        ],
        [
            ["raw material 1", "raw material 2"],
            [1.0, 1.0],
            "<=",
            0.4,
        ],
    ]


def _repair_config() -> dict[str, object]:
    return {
        "steps": {
            "raw material 1": 0.01,
            "raw material 2": 0.01,
            "raw material 3": 0.01,
            "temperature": 1.0,
            "time": 2.0,
        },
        "comp_idx": [
            "raw material 1",
            "raw material 2",
            "raw material 3",
        ],
        "k": 2,
        "final_priority": "constraints",
        "max_iters": 24,
        "num_alternations": 3,
    }


def test_fastapi_combines_outcome_and_input_constraints_for_hybrid_ei(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": _hybrid_records(),
            "input_cols": [
                "raw material 1",
                "raw material 2",
                "raw material 3",
                "temperature",
                "time",
            ],
            "target_cols": ["property", "y_cat_str"],
            "model_config": {
                "task_type": "hybrid",
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
                        "task_type": "binary",
                        "model_type": "base",
                        "name": "y_cat_str",
                    },
                ],
                "use_hybrid": True,
            },
            "fit_config": {"skip_fit": True},
        },
    )
    assert response.status_code == 200, response.text
    model_id = response.json()["model_id"]

    candidate_response = client.post(
        f"/api/v1/tabular/models/{model_id}/candidates",
        json={
            "objective_mode": "scalar",
            "objective_output": "property",
            "objective_direction": "maximize",
            "acquisition_config": {"name": "ei"},
            "outcome_constraint_config": {
                "constraints": [
                    {
                        "output": "y_cat_str",
                        "target_class": "b",
                        "threshold": 0.75,
                        "sense": "ge",
                    }
                ]
            },
            "optimize_config": {
                "q": 2,
                "optimizer": "optimize_acqf",
                "num_restarts": 4,
                "raw_samples": 64,
                "optimizer_kwargs": {
                    "options": {
                        "maxiter": 12,
                        "batch_limit": 4,
                    }
                },
            },
            "constraints": _constraints(),
            "repair_config": _repair_config(),
        },
    )
    assert candidate_response.status_code == 200, candidate_response.text
    candidates = pd.DataFrame(candidate_response.json()["candidates"])
    assert len(candidates) == 2

    component_cols = ["raw material 1", "raw material 2", "raw material 3"]
    for _, candidate in candidates.iterrows():
        component_sum = float(candidate[component_cols].sum())
        first_two_sum = float(candidate[["raw material 1", "raw material 2"]].sum())
        process_sum = float(candidate[["temperature", "time"]].sum())
        zero_count = sum(
            abs(float(candidate[column])) <= 1e-8
            for column in component_cols
        )

        assert math.isclose(component_sum, 1.0, rel_tol=0.0, abs_tol=1e-6)
        assert first_two_sum <= 0.4 + 1e-6
        assert process_sum >= 100.0 - 1e-6
        assert zero_count >= 1
