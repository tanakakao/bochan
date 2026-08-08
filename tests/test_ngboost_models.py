from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.ensemble import EnsembleModel
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.regression.boosting import NGBoostEnsembleModel, NGBoostRegressorModel


class _FakeDistribution:
    def __init__(self, loc, scale) -> None:
        self._loc = np.asarray(loc)
        self._scale = np.asarray(scale)

    @property
    def params(self):
        return {"loc": self._loc, "scale": self._scale}


class _FakeNGBoost:
    def __init__(self, *, bias: float = 0.0, scale: float = 0.5) -> None:
        self.bias = float(bias)
        self.scale = float(scale)
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.bias

    def pred_dist(self, X):
        loc = self.predict(X)
        scale = np.full_like(loc, self.scale, dtype=float)
        return _FakeDistribution(loc=loc, scale=scale)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    return train_X, train_Y


def test_ngboost_regressor_is_botorch_model_and_returns_gpytorch_posterior() -> None:
    train_X, train_Y = _training_data()
    estimator = _FakeNGBoost(scale=0.4)
    model = NGBoostRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
    ).fit()

    assert isinstance(model, Model)
    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))

    assert isinstance(posterior, GPyTorchPosterior)
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[0.25], [0.75]], dtype=torch.double),
    )
    torch.testing.assert_close(
        posterior.variance,
        torch.full((2, 1), 0.16, dtype=torch.double),
    )


def test_ngboost_regressor_posterior_rsample_preserves_botorch_shape() -> None:
    train_X, train_Y = _training_data()
    model = NGBoostRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeNGBoost(),
    ).fit()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    samples = posterior.rsample(torch.Size([5]))

    assert samples.shape == torch.Size([5, 2, 1])


def test_ngboost_regressor_works_with_standard_qlogei_sampler() -> None:
    train_X, train_Y = _training_data()
    model = NGBoostRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeNGBoost(),
    ).fit()
    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())

    value = acqf(torch.tensor([[[0.8]]], dtype=torch.double))

    assert torch.isfinite(value).all()


def test_ngboost_ensemble_is_botorch_ensemble_and_uses_member_means() -> None:
    train_X, train_Y = _training_data()
    estimators = [_FakeNGBoost(bias=0.0), _FakeNGBoost(bias=1.0), _FakeNGBoost(bias=2.0)]
    model = NGBoostEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=estimators,
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
    # BoTorch EnsemblePosterior follows torch.var's unbiased sample-variance
    # convention for equally weighted finite ensembles.
    torch.testing.assert_close(
        posterior.variance,
        torch.full((2, 1), 1.0, dtype=torch.double),
    )


def test_ngboost_ensemble_works_with_standard_qlogei_sampler() -> None:
    train_X, train_Y = _training_data()
    model = NGBoostEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[_FakeNGBoost(bias=0.5), _FakeNGBoost(bias=1.0), _FakeNGBoost(bias=1.5)],
        bootstrap=False,
    ).fit()
    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())

    value = acqf(torch.tensor([[[0.8]]], dtype=torch.double))

    assert torch.isfinite(value).all()


def test_ngboost_is_available_from_default_registry() -> None:
    model_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="ngboost",
            outcome_transform=False,
        )
    )
    ensemble_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="ngboost_ensemble",
            outcome_transform=False,
        )
    )

    assert model_cls is NGBoostRegressorModel
    assert ensemble_cls is NGBoostEnsembleModel


def test_high_level_fit_path_can_fit_ngboost_bound_method() -> None:
    train_X, train_Y = _training_data()
    estimator = _FakeNGBoost()
    config = ModelConfig(
        task_type="regression",
        model_type="ngboost",
        model_cls=NGBoostRegressorModel,
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    bundle = build_model(train_X, train_Y, config)

    fitted = fit_model(bundle, FitConfig())

    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
    assert estimator.fit_y is not None


def test_ngboost_ensemble_bootstrap_is_reproducible() -> None:
    train_X = torch.arange(10, dtype=torch.double).unsqueeze(-1)
    train_Y = train_X.clone()
    estimators_a = [_FakeNGBoost(), _FakeNGBoost()]
    estimators_b = [_FakeNGBoost(), _FakeNGBoost()]

    NGBoostEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=estimators_a,
        bootstrap=True,
        random_state=7,
    ).fit()
    NGBoostEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=estimators_b,
        bootstrap=True,
        random_state=7,
    ).fit()

    for estimator_a, estimator_b in zip(estimators_a, estimators_b, strict=True):
        np.testing.assert_array_equal(estimator_a.fit_X, estimator_b.fit_X)
        np.testing.assert_array_equal(estimator_a.fit_y, estimator_b.fit_y)
