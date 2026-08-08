from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.ensemble import EnsembleModel
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.regression.boosting import (
    NGBoostMixedEnsembleModel,
    NGBoostMixedRegressorModel,
)


@dataclass
class _FakeDistribution:
    loc: np.ndarray
    scale: np.ndarray

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"loc": self.loc, "scale": self.scale}


class _FakeNGBoost:
    def __init__(self, *, bias: float | None = None, scale: float = 0.5) -> None:
        self.bias = bias
        self.scale = float(scale)
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        if self.bias is None:
            self.bias = float(np.mean(self.fit_y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + float(self.bias)

    def pred_dist(self, X):
        loc = self.predict(X)
        return _FakeDistribution(
            loc=loc,
            scale=np.full_like(loc, self.scale, dtype=float),
        )


def _mixed_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 2.0],
            [0.25, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.0], [1.0], [2.0], [1.0]], dtype=torch.double)
    return train_X, train_Y


def test_mixed_regressor_one_hot_encodes_only_for_estimator() -> None:
    train_X, train_Y = _mixed_training_data()
    estimator = _FakeNGBoost(bias=1.0)
    model = NGBoostMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert isinstance(model, Model)
    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert estimator.fit_X is not None
    np.testing.assert_allclose(
        estimator.fit_X,
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.5, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.25, 0.0, 1.0, 0.0],
            ]
        ),
    )

    posterior = model.posterior(
        torch.tensor([[0.75, 2.0], [0.1, 0.0]], dtype=torch.double)
    )
    assert isinstance(posterior, GPyTorchPosterior)
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[1.75], [1.1]], dtype=torch.double),
    )


def test_mixed_regressor_rejects_unseen_category() -> None:
    train_X, train_Y = _mixed_training_data()
    model = NGBoostMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=_FakeNGBoost(bias=1.0),
    ).fit()

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.4, 3.0]], dtype=torch.double))


def test_mixed_regressor_works_with_standard_qlogei_sampler() -> None:
    train_X, train_Y = _mixed_training_data()
    model = NGBoostMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=_FakeNGBoost(bias=1.0, scale=0.4),
    ).fit()
    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())

    value = acqf(torch.tensor([[[0.8, 1.0]]], dtype=torch.double))

    assert torch.isfinite(value).all()


def test_mixed_ensemble_uses_encoded_features_and_member_means() -> None:
    train_X, train_Y = _mixed_training_data()
    estimators = [
        _FakeNGBoost(bias=0.0),
        _FakeNGBoost(bias=1.0),
        _FakeNGBoost(bias=2.0),
    ]
    model = NGBoostMixedEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimators=estimators,
        bootstrap=False,
    ).fit()

    assert isinstance(model, EnsembleModel)
    for estimator in estimators:
        assert estimator.fit_X is not None
        assert estimator.fit_X.shape == (4, 4)

    posterior = model.posterior(
        torch.tensor([[0.25, 0.0], [0.75, 2.0]], dtype=torch.double)
    )
    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[1.25], [1.75]], dtype=torch.double),
    )
    torch.testing.assert_close(
        posterior.variance,
        torch.full((2, 1), 2.0 / 3.0, dtype=torch.double),
    )


def test_mixed_models_are_selected_by_cat_dims_from_default_registry() -> None:
    single_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="ngboost",
            cat_dims=[1],
            outcome_transform=False,
        )
    )
    ensemble_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="ngboost_ensemble",
            cat_dims=[1],
            outcome_transform=False,
        )
    )

    assert single_cls is NGBoostMixedRegressorModel
    assert ensemble_cls is NGBoostMixedEnsembleModel


def test_high_level_fit_path_builds_mixed_ngboost_from_cat_dims() -> None:
    train_X, train_Y = _mixed_training_data()
    estimator = _FakeNGBoost(bias=1.0)
    config = ModelConfig(
        task_type="regression",
        model_type="ngboost",
        cat_dims=[1],
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )

    bundle = build_model(train_X, train_Y, config)
    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, NGBoostMixedRegressorModel)
    assert fitted.input_type == "mixed"
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
    assert estimator.fit_X.shape[-1] == 4
