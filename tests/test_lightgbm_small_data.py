from __future__ import annotations

import pytest
import torch

from bochan.models.external.lightgbm import (
    _default_lightgbm_min_child_samples,
    _resolve_lightgbm_estimator_kwargs,
)
from bochan.models.regression.external import (
    LightGBMEnsembleModel,
    LightGBMRegressorModel,
)


pytest.importorskip("lightgbm")


def _small_regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
    Y = (torch.sin(5.0 * X) + 0.3 * X).clone()
    return X, Y


def test_small_data_default_scales_with_observation_count() -> None:
    assert _default_lightgbm_min_child_samples(5) == 1
    assert _default_lightgbm_min_child_samples(24) == 3
    assert _default_lightgbm_min_child_samples(100) == 10
    assert _default_lightgbm_min_child_samples(200) == 20
    assert _default_lightgbm_min_child_samples(1000) == 20


def test_explicit_lightgbm_leaf_size_is_never_overridden() -> None:
    assert _resolve_lightgbm_estimator_kwargs(
        {"min_child_samples": 7},
        n_samples=24,
    )["min_child_samples"] == 7

    resolved_alias = _resolve_lightgbm_estimator_kwargs(
        {"min_data_in_leaf": 6},
        n_samples=24,
    )
    assert resolved_alias["min_data_in_leaf"] == 6
    assert "min_child_samples" not in resolved_alias


def test_small_data_regressor_does_not_collapse_to_constant_prediction() -> None:
    train_X, train_Y = _small_regression_data()
    model = LightGBMRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        n_estimators=20,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()

    assert model.estimator.get_params()["min_child_samples"] == 3
    posterior = model.posterior(
        torch.tensor([[0.15], [0.85]], dtype=torch.double)
    )
    prediction = posterior.mean.squeeze(-1)

    assert torch.isfinite(prediction).all()
    assert not torch.isclose(prediction[0], prediction[1], atol=1e-8, rtol=1e-8)


def test_small_data_ensemble_does_not_collapse_to_constant_prediction() -> None:
    train_X, train_Y = _small_regression_data()
    model = LightGBMEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=5,
        n_estimators=20,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()

    assert all(
        estimator.get_params()["min_child_samples"] == 3
        for estimator in model.estimators
    )
    posterior = model.posterior(
        torch.tensor([[0.15], [0.85]], dtype=torch.double)
    )
    prediction = posterior.mean.squeeze(-1)

    assert torch.isfinite(prediction).all()
    assert not torch.isclose(prediction[0], prediction[1], atol=1e-8, rtol=1e-8)
    assert torch.any(posterior.variance > 0.0)
