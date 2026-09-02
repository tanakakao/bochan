from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import CrossValidationConfig, FitConfig, ModelConfig, ObservationData, PredictionResult
from bochan.api.evaluation.cross_validation import cross_validate_observations


class _FakeObservationCVOptimizer:
    captured_observations: list[ObservationData] = []

    def __init__(
        self,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        *,
        bounds=None,
        model_registry=None,
        acquisition_registry=None,
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.bounds = bounds
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry
        self.model = SimpleNamespace()

    def fit(
        self,
        *,
        observation_data: ObservationData,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> _FakeObservationCVOptimizer:
        self.model_config = model_config or self.model_config
        self.fit_config = fit_config or self.fit_config
        self.observation_data = observation_data
        type(self).captured_observations.append(observation_data)
        return self

    def predict(self, X, *, return_result: bool = False):
        x = torch.as_tensor(X).reshape(-1, 1)
        mean = torch.cat((x + 1.0, 2.0 * x + 1.0), dim=-1)
        result = PredictionResult(
            posterior=None,
            mean=mean,
            variance=torch.full_like(mean, 0.04),
            task_type="multi_objective",
            variance_kind="predictive",
        )
        return result if return_result else mean


def _partial_known_noise_observations() -> ObservationData:
    X = torch.arange(8, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [
            [1.0, float("nan")],
            [2.0, 3.0],
            [float("nan"), 5.0],
            [4.0, 7.0],
            [5.0, float("nan")],
            [float("nan"), 11.0],
            [7.0, 13.0],
            [float("nan"), float("nan")],
        ],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [
            [0.10, float("nan")],
            [0.11, 0.21],
            [float("nan"), 0.22],
            [0.13, 0.23],
            [0.14, float("nan")],
            [float("nan"), 0.25],
            [0.16, 0.26],
            [float("nan"), float("nan")],
        ],
        dtype=torch.double,
    )
    return ObservationData.from_status(
        X,
        Y,
        status=["success"] * 6 + ["failed", "pending"],
        Yvar=Yvar,
    )


def test_observation_cv_scores_only_successful_observed_cells_and_keeps_yvar() -> None:
    _FakeObservationCVOptimizer.captured_observations = []
    optimizer = _FakeObservationCVOptimizer(
        ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )
    observations = _partial_known_noise_observations()

    result = cross_validate_observations(
        optimizer,
        observations,
        cv_config=CrossValidationConfig(n_splits=3, shuffle=False),
    )

    assert result.metadata["observation_aware"] is True
    assert result.metadata["failure_model_evaluated"] is False
    assert result.metadata["known_observation_variance"] is True
    assert result.metadata["n_objective_rows"] == 6
    assert result.metadata["n_excluded_failed_rows"] == 1
    assert result.metadata["n_excluded_pending_rows"] == 1
    assert result.outputs["output_0"].oof_predictions.indices.tolist() == [0, 1, 3, 4]
    assert result.outputs["output_1"].oof_predictions.indices.tolist() == [1, 2, 3, 5]
    assert result.outputs["output_0"].oof_metrics["rmse"] == pytest.approx(0.0)
    assert result.outputs["output_1"].oof_metrics["rmse"] == pytest.approx(0.0)

    assert len(_FakeObservationCVOptimizer.captured_observations) == 3
    for fold_observations in _FakeObservationCVOptimizer.captured_observations:
        assert not bool(fold_observations.failed_mask.any())
        assert not bool(fold_observations.pending_mask.any())
        assert fold_observations.Yvar is not None
        valid = fold_observations.observed_mask
        assert bool(torch.isfinite(fold_observations.Yvar[valid]).all())
        assert bool((fold_observations.Yvar[valid] > 0.0).all())


def test_observation_cv_rejects_fold_without_training_values_for_an_output() -> None:
    X = torch.arange(4, dtype=torch.double).unsqueeze(-1)
    observations = ObservationData(
        X=X,
        Y=torch.tensor(
            [[1.0, 1.0], [2.0, 3.0], [3.0, float("nan")], [4.0, float("nan")]],
            dtype=torch.double,
        ),
    )
    optimizer = _FakeObservationCVOptimizer(
        ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )

    with pytest.raises(ValueError, match="no observed values.*output_1"):
        cross_validate_observations(
            optimizer,
            observations,
            cv_config=CrossValidationConfig(n_splits=2, shuffle=False),
        )


def test_observation_cv_rejects_feature_importance_until_output_protocol_exists() -> None:
    optimizer = _FakeObservationCVOptimizer(
        ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )

    with pytest.raises(ValueError, match="feature importance"):
        cross_validate_observations(
            optimizer,
            _partial_known_noise_observations(),
            cv_config=CrossValidationConfig(feature_importance_config=object()),
        )
