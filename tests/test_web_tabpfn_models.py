from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]


class _FakeBarCriterion:
    def __init__(self) -> None:
        self.borders = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=-1)
        centers = torch.tensor([-0.5, 0.5, 1.5], dtype=logits.dtype, device=logits.device)
        mean = (probabilities * centers).sum(dim=-1)
        second = (probabilities * centers.square()).sum(dim=-1)
        return (second - mean.square()).clamp_min(1e-6)


class _FakeTabPFNRegressor:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.fit_X: np.ndarray | None = None
        self.predict_calls = 0
        self.criterion = _FakeBarCriterion()

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X, *, output_type="mean"):
        X = np.asarray(X)
        self.predict_calls += 1
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
        return {"mean": mean, "criterion": self.criterion, "logits": logits}


class _FakeTabPFNClassifier:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.categorical_features_indices = None
        self.classes_ = np.arange(self.num_classes)
        self.fit_X: np.ndarray | None = None
        self.predict_calls = 0

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.arange(self.num_classes)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        self.predict_calls += 1
        if self.num_classes == 2:
            p1 = np.clip(0.1 + 0.8 * X[:, 0], 0.01, 0.99)
            return np.column_stack([1.0 - p1, p1])
        raw = np.column_stack(
            [
                1.2 - X[:, 0],
                0.4 + 0.8 * X[:, 0],
                0.2 + 0.5 * (1.0 - np.abs(X[:, 0] - 0.5)),
            ]
        )
        return raw / raw.sum(axis=1, keepdims=True)


def _store(data: pd.DataFrame) -> tuple[object, str]:
    from bochan.desktop.services import DatasetStore, build_dataset_record

    record = build_dataset_record(
        data=data,
        name="tabpfn-web.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id


def _request(
    dataset_id: str,
    *,
    task_type: str,
    estimator: object,
    q: int = 1,
    input_perturbation: bool = False,
):
    from bochan.serving.webapp.app import RegressionRunRequest

    acquisition_name = "EI" if task_type == "regression" else "predictive_entropy"
    family = "bayesian_optimization" if task_type == "regression" else "active_learning"
    risk_type = "cvar" if input_perturbation else "none"
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
                    "task_type": task_type,
                    "optimize": True,
                    "direction": "maximize",
                    "goal": "none",
                    "value": None,
                    "target_classes": [2] if task_type == "classification" else [],
                }
            ],
        },
        fit_maxiter=7,
        normalize=True,
        outcome_transform=task_type == "regression",
        input_perturbation=input_perturbation,
        n_w=4,
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
            "name": acquisition_name,
            "beta": 2.0,
            "acqf_kwargs": {
                "web_family": family,
                "web_risk_type": risk_type,
                "web_risk_alpha": 0.5,
            },
        },
        optimizer={
            "name": "ga",
            "q": q,
            "num_restarts": 1,
            "raw_samples": 8,
            "sequential": True,
        },
        cross_validation=False,
        feature_importance={"enabled": False},
    )


def test_web_exposes_tabpfn_as_foundation_derivative_free_model() -> None:
    source = (ROOT / "web" / "src" / "modelOptions.ts").read_text(encoding="utf-8")

    assert '"foundation"' in source
    assert 'label: "基盤モデル"' in source
    assert '{ value: "tabpfn", label: "TabPFN", family: "foundation" }' in source
    assert 'modelType === "tabpfn"' in source
    assert "requiresDerivativeFreeSearch" in source


def test_web_capabilities_and_registry_expose_supported_tabpfn_tasks() -> None:
    from bochan.api.model_registry import DEFAULT_MODEL_REGISTRY
    from bochan.serving.webapp.app import WEB_CAPABILITIES

    registry = DEFAULT_MODEL_REGISTRY.raw()
    assert "tabpfn" in WEB_CAPABILITIES["model_types"]
    for input_type in ("normal", "mixed"):
        for task_type in ("regression", "binary", "multiclass"):
            assert "tabpfn" in registry[input_type][task_type]
        assert "tabpfn" not in registry[input_type]["ordinal"]


def test_web_tabpfn_runtime_defaults_are_small_and_explicit_kwargs_win() -> None:
    from bochan.serving.webapp.model_runtime import apply_web_model_runtime_defaults

    defaults = apply_web_model_runtime_defaults({}, model_type="tabpfn", fit_maxiter=128)
    explicit = apply_web_model_runtime_defaults(
        {
            "n_estimators": 7,
            "show_progress_bar": True,
            "n_preprocessing_jobs": 3,
        },
        model_type="tabpfn",
        fit_maxiter=1,
    )

    assert defaults == {
        "n_estimators": 4,
        "show_progress_bar": False,
        "n_preprocessing_jobs": 1,
    }
    assert explicit["n_estimators"] == 7
    assert explicit["show_progress_bar"] is True
    assert explicit["n_preprocessing_jobs"] == 3


def _risk_request(*, input_perturbation: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="tabpfn",
        input_perturbation=input_perturbation,
        n_w=16,
        acquisition={
            "acqf_kwargs": {
                "web_family": "bayesian_optimization",
                "web_risk_type": "none",
                "web_risk_alpha": 0.2,
            }
        },
    )


def test_web_tabpfn_uses_bounded_ga_budgets() -> None:
    from bochan.serving.webapp.risk_settings import web_risk_run
    from bochan.serving.webapp.search_settings import resolve_search_method

    with web_risk_run(_risk_request(input_perturbation=False)):
        _, ordinary, _ = resolve_search_method("ga", multi_objective=False)
    with web_risk_run(_risk_request(input_perturbation=True)):
        _, perturbed, _ = resolve_search_method("ga", multi_objective=False)

    assert ordinary["options"] == {"pop_size": 16, "num_generations": 20}
    assert perturbed["options"] == {"pop_size": 12, "num_generations": 16}


def test_web_tabpfn_q3_ga_uses_joint_execution_copy() -> None:
    from bochan.serving.webapp.candidate_runtime import (
        apply_web_candidate_runtime_defaults,
        uses_batched_external_joint_batch,
    )

    request = _request(
        "unused",
        task_type="regression",
        estimator=_FakeTabPFNRegressor(),
        q=3,
    )
    resolved = apply_web_candidate_runtime_defaults(request)

    assert uses_batched_external_joint_batch(request) is True
    assert request.optimizer.sequential is True
    assert resolved.optimizer.q == 3
    assert resolved.optimizer.sequential is False


def test_web_tabpfn_regression_q3_cvar_input_perturbation_runs_end_to_end() -> None:
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    torch.manual_seed(0)
    x = np.linspace(0.0, 1.0, 14)
    store, dataset_id = _store(
        pd.DataFrame({"x": x, "target": 1.0 - (x - 0.65) ** 2})
    )
    estimator = _FakeTabPFNRegressor()
    result = run_regression_web_workflow(
        _request(
            dataset_id,
            task_type="regression",
            estimator=estimator,
            q=3,
            input_perturbation=True,
        ),
        store,
    )

    uniqueness = result["metadata"]["candidate_uniqueness"]
    assert result["model_type"] == "tabpfn"
    assert len(result["candidates"]) == 3
    assert uniqueness["requested_q"] == 3
    assert uniqueness["sequential"] is False
    assert uniqueness["unique_count"] == 3
    assert result["batch_acq_value"] is not None
    assert result["metadata"]["input_perturbation_risk_type"] == "cvar"
    assert result["metadata"]["input_perturbation_risk_enabled"] is True
    assert estimator.fit_X is not None
    assert estimator.fit_X.shape[0] == len(x)
    assert estimator.predict_calls > 0


@pytest.mark.parametrize("num_classes", [2, 3])
def test_web_tabpfn_classification_active_learning_runs_end_to_end(num_classes: int) -> None:
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    torch.manual_seed(0)
    x = np.linspace(0.0, 1.0, 15)
    if num_classes == 2:
        target = (x >= 0.5).astype(int)
    else:
        target = np.digitize(x, [0.34, 0.67])
    store, dataset_id = _store(pd.DataFrame({"x": x, "target": target}))
    estimator = _FakeTabPFNClassifier(num_classes)
    request = _request(
        dataset_id,
        task_type="classification",
        estimator=estimator,
    )
    if num_classes == 2:
        settings = [
            {
                "target": "target",
                "task_type": "classification",
                "optimize": True,
                "direction": "maximize",
                "goal": "none",
                "value": None,
                "target_class": 1,
            }
        ]
    else:
        settings = [
            {
                "target": "target",
                "task_type": "classification",
                "optimize": True,
                "direction": "maximize",
                "goal": "none",
                "value": None,
                "target_classes": [2],
            }
        ]
    request = request.model_copy(
        update={
            "model_kwargs": {
                "estimator": estimator,
                "web_target_settings": settings,
            }
        }
    )

    result = run_regression_web_workflow(request, store)

    assert result["model_type"] == "tabpfn"
    assert len(result["candidates"]) == 1
    assert result["metadata"]["optimizer"] == "evo"
    assert result["metadata"]["search_method"] == "ga"
    assert estimator.fit_X is not None
    assert estimator.predict_calls > 0


def test_tabpfn_ordinal_configuration_fails_with_core_registry_error() -> None:
    from bochan.api import ModelConfig
    from bochan.api.factory import resolve_model_cls

    with pytest.raises(ValueError, match="Unknown model setting"):
        resolve_model_cls(
            ModelConfig(
                task_type="ordinal",
                model_type="tabpfn",
            )
        )
