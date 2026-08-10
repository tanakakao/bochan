from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("fastapi")
pytest.importorskip("plotly")


class _FakeBarCriterion:
    def __init__(self) -> None:
        self.borders = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=-1)
        centers = torch.tensor(
            [-0.5, 0.5, 1.5],
            dtype=logits.dtype,
            device=logits.device,
        )
        mean = (probabilities * centers).sum(dim=-1)
        second = (probabilities * centers.square()).sum(dim=-1)
        return (second - mean.square()).clamp_min(1e-6)


class _FakeTabPFNRegressor:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.criterion = _FakeBarCriterion()

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X, *, output_type="mean"):
        X = np.asarray(X)
        mean = 1.0 - (X[:, 0] - 0.65) ** 2
        if output_type != "full":
            return mean
        logits = torch.tensor(
            np.column_stack(
                [
                    0.2 + 0.2 * X[:, 0],
                    0.6 + 0.1 * X[:, 0],
                    0.2 - 0.1 * X[:, 0],
                ]
            ),
            dtype=torch.float32,
        )
        return {
            "mean": mean,
            "criterion": self.criterion,
            "logits": logits,
        }


def _request(dataset_id: str, estimator: object):
    from bochan.serving.webapp.app import RegressionRunRequest

    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
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
        fit_maxiter=1,
        normalize=True,
        outcome_transform=True,
        search_space=[
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 1.0,
                "fixed": False,
            }
        ],
        acquisition={
            "name": "EI",
            "beta": 2.0,
            "acqf_kwargs": {"web_family": "bayesian_optimization"},
        },
        optimizer={
            "name": "ga",
            "q": 1,
            "num_restarts": 1,
            "raw_samples": 8,
            "sequential": True,
        },
        feature_importance={"enabled": False},
    )


def test_tabpfn_run_retains_yy_and_1d_visualization_session() -> None:
    from bochan.desktop.services import DatasetStore, build_dataset_record
    from bochan.serving.webapp.logging import reset_request_id, set_request_id
    from bochan.serving.webapp.visualization_sessions import build_visualization
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    x = np.linspace(0.0, 1.0, 12)
    record = build_dataset_record(
        data=pd.DataFrame({"x": x, "target": 1.0 - (x - 0.65) ** 2}),
        name="tabpfn-visualization.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    run_id = "tabpfn-visualization-session"
    token = set_request_id(run_id)
    try:
        result = run_regression_web_workflow(
            _request(record.dataset_id, _FakeTabPFNRegressor()),
            store,
        )
    finally:
        reset_request_id(token)

    assert result["visualization_run_id"] == run_id
    yyplot = build_visualization(
        run_id,
        {"kind": "yyplot", "target": "target"},
    )
    one_dimensional = build_visualization(
        run_id,
        {
            "kind": "1d",
            "target": "target",
            "features": ["x"],
            "fixed_values": {},
            "show_type": "pred",
            "n": 20,
        },
    )

    assert yyplot["figure"]["data"]
    assert one_dimensional["figure"]["data"]
