"""Tests for the regular API cross-validation surface."""

import math

import pytest
import torch

from bochan.api import BayesianOptimizer, CrossValidationConfig, FitConfig, ModelConfig


def test_regression_cross_validation_oof_and_no_side_effects() -> None:
    """OOF predictions preserve order and the calling optimizer stays untouched."""
    train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(3 * train_X)
    optimizer = BayesianOptimizer(ModelConfig(outcome_transform=False), FitConfig(skip_fit=True))

    result = optimizer.cross_validate(
        train_X,
        train_Y,
        cv_config=CrossValidationConfig(n_splits=2, random_state=7),
    )

    assert result.oof_predictions.indices.tolist() == list(range(8))
    assert torch.equal(result.oof_predictions.prediction_count, torch.ones(8, dtype=torch.long))
    assert torch.isnan(result.oof_predictions.fold_prediction_std).all()
    assert result.oof_predictions.predictive_std is not None
    assert set(result.output.oof_metrics) == {"rmse", "mae", "mape", "r2"}
    assert result.models is None
    assert optimizer.model is optimizer.bundle is optimizer.mll is None
    assert optimizer.train_X is optimizer.train_Y is None
    assert optimizer.history == []


def test_leave_one_out_and_mape_zero_warnings() -> None:
    """Undefined per-fold R2 and safe zero-target MAPE are reported explicitly."""
    train_X = torch.arange(4, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.double)
    optimizer = BayesianOptimizer(ModelConfig(outcome_transform=False), FitConfig(skip_fit=True))

    result = optimizer.cross_validate(
        train_X,
        train_Y,
        cv_config=CrossValidationConfig(splitter="loo"),
    )

    assert torch.isnan(result.test_metric_summary["r2"].values).all()
    assert math.isfinite(result.output.oof_metrics["r2"])
    assert math.isnan(result.output.oof_metrics["mape"])
    assert any("R2" in warning for warning in result.warnings)
    assert any("MAPE" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 1},
        {"classification_threshold": 1.1},
        {"classification_average": "invalid"},
        {"mape_zero_policy": "invalid"},
    ],
)
def test_cross_validation_config_validation(kwargs: dict[str, object]) -> None:
    """Invalid settings fail before model construction."""
    with pytest.raises(ValueError):
        CrossValidationConfig(**kwargs)


def test_cross_validation_feature_importance_is_optional_and_aggregated() -> None:
    """Validation-fold importance is opt-in and retains fold-level results."""
    from bochan.inspection import FeatureImportanceConfig

    train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = 2 * train_X
    optimizer = BayesianOptimizer(ModelConfig(outcome_transform=False), FitConfig(skip_fit=True))
    disabled = optimizer.cross_validate(train_X, train_Y, cv_config=CrossValidationConfig(n_splits=2))
    assert disabled.feature_importance is None
    enabled = optimizer.cross_validate(
        train_X,
        train_Y,
        cv_config=CrossValidationConfig(
            n_splits=2,
            feature_names=["x"],
            feature_importance_config=FeatureImportanceConfig(n_repeats=2, diagnostic_methods=[]),
        ),
    )
    assert enabled.feature_importance.outputs["output_0"].predictive_methods["permutation"]
    assert all(fold.feature_importance is not None for fold in enabled.output.folds)
    summary = enabled.feature_importance.outputs["output_0"].predictive_methods["permutation"].entries["x"]
    assert summary.valid_fold_count == 2
    assert len(summary.within_fold_repeat_std) == 2
