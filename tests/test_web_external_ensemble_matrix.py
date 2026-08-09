from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("fastapi")


class _FakeProbabilityMember:
    def __init__(self, num_classes: int, offset: float = 0.0) -> None:
        self.num_classes = int(num_classes)
        self.offset = float(offset)
        self.classes_ = np.arange(self.num_classes)
        self.fit_X: np.ndarray | None = None
        self.predict_calls = 0

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.arange(self.num_classes)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        self.predict_calls += 1
        if self.num_classes == 2:
            p1 = np.clip(0.1 + 0.75 * X[:, 0] + self.offset, 0.01, 0.99)
            return np.column_stack([1.0 - p1, p1])
        raw = np.column_stack(
            [
                1.2 - X[:, 0] + self.offset,
                0.4 + 0.8 * X[:, 0],
                0.3 + 0.4 * (1.0 - np.abs(X[:, 0] - 0.5)) - self.offset,
            ]
        )
        raw = np.clip(raw, 1e-4, None)
        return raw / raw.sum(axis=1, keepdims=True)


class _FakeProbabilityForest:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.classes_ = np.arange(self.num_classes)
        self.estimators_ = [
            _FakeProbabilityMember(num_classes, -0.03),
            _FakeProbabilityMember(num_classes, 0.03),
        ]
        self.fit_X: np.ndarray | None = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.arange(self.num_classes)
        return self


def _store(num_classes: int) -> tuple[object, str, int]:
    from bochan.desktop.services import DatasetStore, build_dataset_record

    x = np.linspace(0.0, 1.0, 15)
    if num_classes == 2:
        target = (x >= 0.5).astype(int)
    else:
        target = np.digitize(x, [0.34, 0.67])
    record = build_dataset_record(
        data=pd.DataFrame({"x": x, "target": target}),
        name="ensemble-classification.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id, len(x)


def _model_kwargs(model_type: str, num_classes: int) -> tuple[dict[str, object], list[_FakeProbabilityMember]]:
    if model_type == "random_forest":
        forest = _FakeProbabilityForest(num_classes)
        return {"estimator": forest}, list(forest.estimators_)

    members = [
        _FakeProbabilityMember(num_classes, -0.03),
        _FakeProbabilityMember(num_classes, 0.03),
    ]
    return {
        "estimators": members,
        "bootstrap": False,
    }, members


def _request(
    dataset_id: str,
    *,
    model_type: str,
    num_classes: int,
    model_kwargs: dict[str, object],
    input_perturbation: bool,
):
    from bochan.serving.webapp.app import RegressionRunRequest

    target_setting: dict[str, object] = {
        "target": "target",
        "task_type": "classification",
        "optimize": True,
        "direction": "maximize",
        "goal": "none",
        "value": None,
    }
    if num_classes == 2:
        target_setting["target_class"] = 1
    else:
        target_setting["target_classes"] = [2]

    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column="target",
        target_columns=["target"],
        directions={"target": "maximize"},
        model_type=model_type,
        model_kwargs={
            **model_kwargs,
            "web_target_settings": [target_setting],
        },
        fit_maxiter=6,
        normalize=True,
        outcome_transform=False,
        input_perturbation=input_perturbation,
        n_w=3,
        perturbation_std=0.05,
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
            "name": "predictive_entropy",
            "beta": 2.0,
            "acqf_kwargs": {
                "web_family": "active_learning",
                "web_risk_type": "none",
                "web_risk_alpha": 0.2,
            },
        },
        optimizer={
            "name": "ga",
            "q": 1,
            "num_restarts": 1,
            "raw_samples": 8,
            "sequential": True,
        },
        cross_validation=False,
        feature_importance={"enabled": False},
    )


@pytest.mark.parametrize(
    "model_type",
    ["random_forest", "lightgbm_ensemble", "ngboost_ensemble"],
)
@pytest.mark.parametrize("num_classes", [2, 3])
def test_web_external_ensemble_classification_matrix_runs(
    model_type: str,
    num_classes: int,
) -> None:
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    torch.manual_seed(0)
    store, dataset_id, n_rows = _store(num_classes)
    model_kwargs, members = _model_kwargs(model_type, num_classes)
    result = run_regression_web_workflow(
        _request(
            dataset_id,
            model_type=model_type,
            num_classes=num_classes,
            model_kwargs=model_kwargs,
            input_perturbation=False,
        ),
        store,
    )

    assert result["model_type"] == model_type
    assert len(result["candidates"]) == 1
    assert result["metadata"]["optimizer"] == "evo"
    assert result["metadata"]["search_method"] == "ga"
    for member in members:
        assert member.predict_calls > 0
        if member.fit_X is not None:
            assert member.fit_X.shape[0] == n_rows


@pytest.mark.parametrize(
    "model_type",
    ["random_forest", "lightgbm_ensemble", "ngboost_ensemble"],
)
def test_web_external_binary_ensemble_input_perturbation_runs(model_type: str) -> None:
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    torch.manual_seed(0)
    store, dataset_id, _ = _store(2)
    model_kwargs, members = _model_kwargs(model_type, 2)
    result = run_regression_web_workflow(
        _request(
            dataset_id,
            model_type=model_type,
            num_classes=2,
            model_kwargs=model_kwargs,
            input_perturbation=True,
        ),
        store,
    )

    assert len(result["candidates"]) == 1
    assert result["metadata"]["input_perturbation_risk_enabled"] is True
    for member in members:
        assert member.predict_calls > 0
