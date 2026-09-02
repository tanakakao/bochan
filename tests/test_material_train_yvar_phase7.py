from types import SimpleNamespace

import pytest
import torch

from bochan.api.configs import FitConfig, ModelConfig
from bochan.api.evaluation.cross_validation import (
    CrossValidationConfig,
    cross_validate_observations,
)
from bochan.api.observation import ExperimentFailureConfig, ObservationData


class _FakeFailureModel:
    def posterior(self, X):
        probability = torch.sigmoid(X[:, :1])
        return SimpleNamespace(
            mean=probability,
            variance=probability * (1.0 - probability),
        )


class _FakeOptimizer:
    fit_observations = []
    fit_failure_configs = []

    def __init__(
        self,
        *,
        model_config=None,
        fit_config=None,
        bounds=None,
        model_registry=None,
        acquisition_registry=None,
    ):
        self.model_config = model_config or ModelConfig(
            task_type="regression", model_type="base"
        )
        self.fit_config = fit_config or FitConfig()
        self.bounds = bounds
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry
        self.model = object()
        self.failure_model = None

    def fit(
        self,
        *,
        observation_data,
        model_config=None,
        fit_config=None,
        failure_config=None,
    ):
        type(self).fit_observations.append(observation_data)
        type(self).fit_failure_configs.append(failure_config)
        self.model = object()
        self.failure_model = _FakeFailureModel() if failure_config is not None else None
        return self

    def predict(self, X, *, return_result=False):
        mean = X[:, :1]
        variance = torch.full_like(mean, 0.04)
        result = SimpleNamespace(
            mean=mean,
            variance=variance,
            variance_kind="posterior",
        )
        return result if return_result else mean


def _optimizer():
    return _FakeOptimizer(
        model_config=ModelConfig(task_type="regression", model_type="base"),
        fit_config=FitConfig(),
    )


def _observations_with_failure_and_noise():
    X = torch.linspace(-2.0, 2.0, 9, dtype=torch.double).unsqueeze(-1)
    failed = torch.tensor(
        [False, True, False, True, False, True, False, True, False]
    )
    pending = torch.tensor(
        [False, False, False, False, False, False, False, False, True]
    )
    Y = torch.tensor(
        [[-2.0], [float("nan")], [-1.0], [float("nan")], [0.0],
         [float("nan")], [1.0], [float("nan")], [float("nan")]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.01], [float("nan")], [0.02], [float("nan")], [0.03],
         [float("nan")], [0.04], [float("nan")], [float("nan")]],
        dtype=torch.double,
    )
    return ObservationData(
        X=X,
        Y=Y,
        Yvar=Yvar,
        failed_mask=failed,
        pending_mask=pending,
    )


def test_phase7_failure_cv_uses_completed_rows_and_preserves_objective_masks():
    _FakeOptimizer.fit_observations.clear()
    _FakeOptimizer.fit_failure_configs.clear()
    observations = _observations_with_failure_and_noise()

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(n_splits=2, random_state=7),
        failure_config=ExperimentFailureConfig(),
    )

    assert result.failure_model is not None
    assert result.metadata["failure_model_evaluated"] is True
    assert result.metadata["failure_probability_target"] == "success"
    assert result.metadata["n_completed_rows"] == 8
    assert result.metadata["n_failure_validation_rows"] == 8
    assert result.metadata["n_excluded_pending_rows"] == 1
    assert result.failure_model.oof_predictions.indices.tolist() == list(range(8))
    assert result.output.oof_predictions.indices.tolist() == [0, 2, 4, 6]
    assert set(result.failure_model.oof_metrics) >= {
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
    }
    assert all(config is not None for config in _FakeOptimizer.fit_failure_configs)
    assert all(
        fold_observations.Yvar is not None
        for fold_observations in _FakeOptimizer.fit_observations
    )


def test_phase7_without_failure_config_preserves_phase5_protocol():
    observations = _observations_with_failure_and_noise()
    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(n_splits=2, random_state=3),
    )

    assert result.failure_model is None
    assert result.metadata["failure_model_evaluated"] is False
    assert "failure_protocol" not in result.metadata
    assert result.metadata["n_objective_rows"] == 4
    assert result.output.oof_predictions.indices.tolist() == [0, 2, 4, 6]


def test_phase7_one_class_validation_sets_roc_auc_nan_and_warns():
    X = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    failed = torch.tensor([False, False, True, True, False, True])
    Y = torch.tensor(
        [[0.0], [1.0], [float("nan")], [float("nan")], [4.0], [float("nan")]],
        dtype=torch.double,
    )
    observations = ObservationData(X=X, Y=Y, failed_mask=failed)

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(
            splitter="kfold",
            n_splits=3,
            shuffle=False,
        ),
        failure_config=ExperimentFailureConfig(),
    )

    assert result.failure_model is not None
    auc_values = result.failure_model.test_metric_summary["roc_auc"].values
    assert torch.isnan(auc_values).any()
    assert any("ROC-AUC is NaN" in warning for warning in result.warnings)


def test_phase7_rejects_fold_without_both_failure_classes():
    X = torch.arange(4, dtype=torch.double).unsqueeze(-1)
    failed = torch.tensor([False, False, False, True])
    Y = torch.tensor([[0.0], [1.0], [2.0], [float("nan")]], dtype=torch.double)
    observations = ObservationData(X=X, Y=Y, failed_mask=failed)

    with pytest.raises(ValueError, match="both successful and failed"):
        cross_validate_observations(
            _optimizer(),
            observations,
            cv_config=CrossValidationConfig(
                splitter="kfold",
                n_splits=2,
                shuffle=False,
            ),
            failure_config=ExperimentFailureConfig(),
        )
