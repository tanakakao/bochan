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


class _DiagnosticFakeTabPFNRegressor:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.criterion = _Criterion()
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X, *, output_type="mean"):
        X = np.asarray(X)
        self.predict_calls += 1
        mean = 0.4 + 0.7 * X[:, 0] - 0.2 * X[:, 1]
        if output_type != "full":
            return mean
        logits = torch.tensor(
            np.column_stack(
                [
                    0.3 + 0.1 * X[:, 0],
                    0.5 + 0.1 * X[:, 1],
                    0.2 + 0.05 * (X[:, 0] + X[:, 1]),
                ]
            ),
            dtype=torch.float32,
        )
        return {"mean": mean, "criterion": self.criterion, "logits": logits}


def test_web_tabpfn_cross_validation_and_permutation_importance_run() -> None:
    from bochan.desktop.services import DatasetStore, build_dataset_record
    from bochan.serving.webapp.app import RegressionRunRequest
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    x1 = np.linspace(0.0, 1.0, 18)
    x2 = np.linspace(1.0, 0.0, 18)
    target = 0.4 + 0.7 * x1 - 0.2 * x2
    record = build_dataset_record(
        data=pd.DataFrame({"x1": x1, "x2": x2, "target": target}),
        name="tabpfn-diagnostics-web.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    estimator = _DiagnosticFakeTabPFNRegressor()

    request = RegressionRunRequest(
        dataset_id=record.dataset_id,
        feature_columns=["x1", "x2"],
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
        outcome_transform=False,
        search_space=[
            {"name": "x1", "type": "numeric", "lower": 0.0, "upper": 1.0, "fixed": False},
            {"name": "x2", "type": "numeric", "lower": 0.0, "upper": 1.0, "fixed": False},
        ],
        acquisition={
            "name": "EI",
            "acqf_kwargs": {"web_family": "bayesian_optimization"},
        },
        optimizer={"name": "ga", "q": 1, "sequential": True},
        cross_validation=True,
        cv_config={
            "splitter": "kfold",
            "n_splits": 3,
            "shuffle": False,
        },
        feature_importance={
            "enabled": True,
            "source": "training",
            "config": {
                "n_repeats": 2,
                "diagnostic_methods": [],
                "compute_noise_importance": False,
                "error_policy": "raise",
            },
            "visualization": {
                "top_k": 5,
                "include_noise": False,
            },
        },
    )

    torch.manual_seed(0)
    result = run_regression_web_workflow(request, store)

    assert result["model_type"] == "tabpfn"
    assert len(result["candidates"]) == 1
    assert result["metadata"]["cross_validation"] is not None
    assert result["feature_importance"] is not None
    assert len(result["feature_importance_summary"]) >= 2
    assert result["feature_importance_warnings"] == [
        "Feature importance was evaluated on training data and may be optimistic."
    ]
    assert result["metadata"]["timings_ms"]["feature_importance"] >= 0.0
    assert estimator.fit_calls >= 1
    assert estimator.predict_calls > 0
