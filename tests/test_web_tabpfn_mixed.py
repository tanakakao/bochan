from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("fastapi")


class _Criterion:
    def __init__(self) -> None:
        self.borders = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=-1)
        centers = torch.tensor([-0.5, 0.5, 1.5], dtype=logits.dtype, device=logits.device)
        mean = (probabilities * centers).sum(dim=-1)
        second = (probabilities * centers.square()).sum(dim=-1)
        return (second - mean.square()).clamp_min(1e-6)


class _MixedFakeTabPFNRegressor:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.fit_X: np.ndarray | None = None
        self.predict_X: np.ndarray | None = None
        self.criterion = _Criterion()

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X, *, output_type="mean"):
        X = np.asarray(X)
        self.predict_X = X.copy()
        mean = 1.0 - (X[:, 0] - 0.65) ** 2 + 0.05 * X[:, 1]
        if output_type != "full":
            return mean
        logits = torch.tensor(
            np.column_stack(
                [
                    0.2 + 0.1 * X[:, 0],
                    0.6 + 0.1 * X[:, 1],
                    0.2 + 0.05 * X[:, 0],
                ]
            ),
            dtype=torch.float32,
        )
        return {"mean": mean, "criterion": self.criterion, "logits": logits}


def test_web_mixed_tabpfn_preserves_categories_through_perturbed_search() -> None:
    from bochan.desktop.services import DatasetStore, build_dataset_record
    from bochan.serving.webapp.app import RegressionRunRequest
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    x = np.linspace(0.0, 1.0, 16)
    material = np.asarray(["A", "B"] * 8)
    target = 1.0 - (x - 0.65) ** 2 + 0.05 * (material == "B")
    record = build_dataset_record(
        data=pd.DataFrame({"x": x, "material": material, "target": target}),
        name="tabpfn-mixed-web.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    estimator = _MixedFakeTabPFNRegressor()

    request = RegressionRunRequest(
        dataset_id=record.dataset_id,
        feature_columns=["x", "material"],
        target_column="target",
        target_columns=["target"],
        directions={"target": "maximize"},
        model_type="tabpfn",
        model_kwargs={
            "estimator": estimator,
            "web_target_settings": [
                {
                    "target": "target",
                    "task_type": "regression",
                    "optimize": True,
                    "direction": "maximize",
                    "goal": "none",
                    "value": None,
                }
            ],
        },
        normalize=True,
        outcome_transform=True,
        input_perturbation=True,
        n_w=4,
        perturbation_std=0.05,
        search_space=[
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 1.0,
                "fixed": False,
            },
            {
                "name": "material",
                "type": "categorical",
                "categories": ["A", "B"],
                "fixed": False,
            },
        ],
        acquisition={
            "name": "EI",
            "acqf_kwargs": {
                "web_family": "bayesian_optimization",
                "web_risk_type": "cvar",
                "web_risk_alpha": 0.5,
            },
        },
        optimizer={"name": "ga", "q": 1, "sequential": True},
        cross_validation=False,
        feature_importance={"enabled": False},
    )

    torch.manual_seed(0)
    result = run_regression_web_workflow(request, store)

    assert result["model_type"] == "tabpfn"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["values"]["material"] in {"A", "B"}
    assert estimator.categorical_features_indices == [1]
    assert estimator.fit_X is not None
    assert estimator.fit_X.shape == (len(x), 2)
    assert set(np.unique(estimator.fit_X[:, 1])).issubset({0.0, 1.0})
    assert estimator.predict_X is not None
    assert set(np.unique(estimator.predict_X[:, 1])).issubset({0.0, 1.0})
    assert result["metadata"]["input_perturbation_risk_enabled"] is True
