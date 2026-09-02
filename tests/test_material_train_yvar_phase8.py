from types import SimpleNamespace

import torch

from bochan.api.configs import FitConfig, ModelConfig
from bochan.api.evaluation.cross_validation import (
    CrossValidationConfig,
    cross_validate_observations,
)
from bochan.api.observation import ExperimentFailureConfig, ObservationData
from bochan.inspection import FeatureImportanceConfig


class _FakeFailureModel:
    def posterior(self, X):
        probability = torch.sigmoid(X[:, :1])
        return SimpleNamespace(
            mean=probability,
            variance=probability * (1.0 - probability),
        )


class _FakeOptimizer:
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
        self.model = SimpleNamespace()
        self.bundle = SimpleNamespace(cat_dims=[])
        self.failure_model = None

    def fit(
        self,
        *,
        observation_data,
        model_config=None,
        fit_config=None,
        failure_config=None,
    ):
        self.observations = observation_data
        self.model = SimpleNamespace()
        self.bundle = SimpleNamespace(cat_dims=[])
        self.failure_model = _FakeFailureModel() if failure_config is not None else None
        return self

    def predict(self, X, *, return_result=False):
        mean = X[:, :2]
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


def _fi_config():
    return FeatureImportanceConfig(
        n_repeats=2,
        random_state=11,
        diagnostic_methods=[],
        compute_noise_importance=False,
    )


def test_phase8_objective_feature_importance_uses_output_specific_observed_rows():
    X = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
            [4.0, 14.0],
            [5.0, 15.0],
            [6.0, 16.0],
            [7.0, 17.0],
        ],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, float("nan")],
            [2.0, 12.0],
            [3.0, float("nan")],
            [4.0, 14.0],
            [5.0, float("nan")],
            [6.0, 16.0],
            [7.0, float("nan")],
        ],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [
            [0.01, 0.02],
            [0.01, float("nan")],
            [0.01, 0.02],
            [0.01, float("nan")],
            [0.01, 0.02],
            [0.01, float("nan")],
            [0.01, 0.02],
            [0.01, float("nan")],
        ],
        dtype=torch.double,
    )
    observations = ObservationData(X=X, Y=Y, Yvar=Yvar)

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(
            splitter="kfold",
            n_splits=2,
            shuffle=False,
            feature_names=["x0", "x1"],
            feature_importance_config=_fi_config(),
        ),
    )

    assert result.feature_importance is not None
    assert set(result.feature_importance.outputs) == {"output_0", "output_1"}
    assert result.feature_importance.metadata["n_folds_by_output"] == {
        "output_0": 2,
        "output_1": 2,
    }
    assert result.metadata["feature_importance_evaluated"] is True
    assert result.metadata["known_observation_variance"] is True
    assert result.failure_feature_importance is None

    output_1_rows = {0, 2, 4, 6}
    for fold in result.outputs["output_1"].folds:
        assert fold.feature_importance is not None
        assert set(fold.feature_importance.outputs) == {"output_1"}
        assert set(fold.test_indices.tolist()).issubset(output_1_rows)
        assert fold.feature_importance.metadata["n_validation_rows"] == len(
            fold.test_indices
        )


def test_phase8_failure_feature_importance_uses_completed_rows_only():
    X = torch.arange(18, dtype=torch.double).reshape(9, 2) / 10.0
    failed = torch.tensor(
        [False, True, False, True, False, True, False, True, False]
    )
    pending = torch.tensor(
        [False, False, False, False, False, False, False, False, True]
    )
    Y = torch.tensor(
        [[0.0], [float("nan")], [0.2], [float("nan")], [0.4],
         [float("nan")], [0.6], [float("nan")], [float("nan")]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.01], [float("nan")], [0.02], [float("nan")], [0.03],
         [float("nan")], [0.04], [float("nan")], [float("nan")]],
        dtype=torch.double,
    )
    observations = ObservationData(
        X=X,
        Y=Y,
        Yvar=Yvar,
        failed_mask=failed,
        pending_mask=pending,
    )

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(
            n_splits=2,
            random_state=7,
            feature_names=["x0", "x1"],
            feature_importance_config=_fi_config(),
        ),
        failure_config=ExperimentFailureConfig(),
    )

    assert result.failure_model is not None
    assert result.failure_feature_importance is not None
    assert set(result.failure_feature_importance.outputs) == {"experiment_success"}
    assert result.metadata["failure_feature_importance_evaluated"] is True
    assert result.metadata["n_excluded_pending_rows"] == 1

    for fold in result.failure_model.folds:
        assert fold.feature_importance is not None
        assert 8 not in fold.test_indices.tolist()
        assert fold.feature_importance.metadata["validation_protocol"] == "completed_rows"
        assert fold.feature_importance.metadata["n_validation_rows"] == len(
            fold.test_indices
        )


def test_phase8_without_feature_importance_config_preserves_phase7_contract():
    X = torch.arange(12, dtype=torch.double).reshape(6, 2)
    Y = X[:, :1].clone()
    observations = ObservationData(X=X, Y=Y)

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(n_splits=2, random_state=3),
    )

    assert result.feature_importance is None
    assert result.failure_feature_importance is None
    assert result.metadata["feature_importance_evaluated"] is False
    assert all(fold.feature_importance is None for fold in result.output.folds)
