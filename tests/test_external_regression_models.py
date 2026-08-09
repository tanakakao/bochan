from __future__ import annotations

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
from bochan.models.regression.external import (
    NGBoostEnsembleModel,
    NGBoostMixedRegressorModel,
    NGBoostRegressorModel,
    RandomForestMixedRegressorModel,
    RandomForestRegressorModel,
)


class _FakeDistribution:
    def __init__(self, loc, scale) -> None:
        self.params = {"loc": np.asarray(loc), "scale": np.asarray(scale)}


class _FakeNGBoost:
    def __init__(self, *, bias: float = 0.0, scale: float = 0.4) -> None:
        self.bias = float(bias)
        self.scale = float(scale)
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.bias

    def pred_dist(self, X):
        loc = self.predict(X)
        return _FakeDistribution(loc, np.full_like(loc, self.scale, dtype=float))


class _FakeTree:
    def __init__(self, bias: float) -> None:
        self.bias = float(bias)

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.bias


class _FakeForest:
    def __init__(self, biases=(0.0, 1.0, 2.0)) -> None:
        self.estimators_ = [_FakeTree(value) for value in biases]
        self.fit_X = None
        self.fit_y = None
        self.fit_kwargs = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        return self


def _data():
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    return X, Y


def _mixed_data():
    X = torch.tensor(
        [[0.0, 0.0], [0.25, 1.0], [0.5, 2.0], [0.75, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]], dtype=torch.double)
    return X, Y


def test_ngboost_regression_botorch_contract_and_qlogei() -> None:
    train_X, train_Y = _data()
    model = NGBoostRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeNGBoost(),
    ).fit()

    assert isinstance(model, Model)
    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])

    acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
    assert torch.isfinite(acq(torch.tensor([[[0.8]]], dtype=torch.double))).all()


def test_ngboost_ensemble_uses_member_predictions() -> None:
    train_X, train_Y = _data()
    model = NGBoostEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[_FakeNGBoost(bias=0.0), _FakeNGBoost(bias=1.0), _FakeNGBoost(bias=2.0)],
        bootstrap=False,
    ).fit()

    assert isinstance(model, EnsembleModel)
    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[1.25], [1.75]], dtype=torch.double),
    )


def test_random_forest_regression_uses_tree_members_and_qlogei() -> None:
    train_X, train_Y = _data()
    model = RandomForestRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForest(),
    ).fit()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[1.25], [1.75]], dtype=torch.double),
    )
    acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
    assert torch.isfinite(acq(torch.tensor([[[0.8]]], dtype=torch.double))).all()


@pytest.mark.parametrize(
    ("model_cls", "estimator"),
    [
        (NGBoostMixedRegressorModel, _FakeNGBoost()),
        (RandomForestMixedRegressorModel, _FakeForest()),
    ],
)
def test_mixed_external_regression_encodes_categories_at_estimator_boundary(
    model_cls,
    estimator,
) -> None:
    train_X, train_Y = _mixed_data()
    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert estimator.fit_X.shape == (5, 4)
    posterior = model.posterior(torch.tensor([[0.4, 1.0]], dtype=torch.double))
    assert torch.isfinite(posterior.mean).all()

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.4, 3.0]], dtype=torch.double))


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("ngboost", NGBoostRegressorModel),
        ("ngboost_ensemble", NGBoostEnsembleModel),
        ("random_forest", RandomForestRegressorModel),
    ],
)
def test_external_regression_registry(model_type, expected_cls) -> None:
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type=model_type,
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


def test_high_level_external_regression_fit() -> None:
    train_X, train_Y = _data()
    estimator = _FakeForest()
    config = ModelConfig(
        task_type="regression",
        model_type="random_forest",
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    bundle = build_model(train_X, train_Y, config)
    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, RandomForestRegressorModel)
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
