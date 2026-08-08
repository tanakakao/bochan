from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.ensemble import EnsembleModel
from botorch.posteriors.ensemble import EnsemblePosterior

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.regression.boosting import RandomForestRegressorModel


class _FakeTree:
    def __init__(self, bias: float) -> None:
        self.bias = float(bias)

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.bias


class _FakeForest:
    def __init__(self, biases=(0.0, 1.0, 2.0)) -> None:
        self.estimators_ = [_FakeTree(bias) for bias in biases]
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        return self


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    return train_X, train_Y


def test_random_forest_is_botorch_ensemble_and_uses_tree_predictions() -> None:
    train_X, train_Y = _training_data()
    estimator = _FakeForest()
    model = RandomForestRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
    ).fit()

    assert isinstance(model, EnsembleModel)
    posterior = model.posterior(
        torch.tensor([[0.25], [0.75]], dtype=torch.double)
    )

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


def test_random_forest_works_with_standard_qlogei_sampler() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForest(),
    ).fit()
    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())

    value = acqf(torch.tensor([[[0.8]]], dtype=torch.double))

    assert torch.isfinite(value).all()


def test_random_forest_is_available_from_default_registry() -> None:
    model_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="random_forest",
            outcome_transform=False,
        )
    )

    assert model_cls is RandomForestRegressorModel


def test_high_level_fit_path_can_fit_random_forest_bound_method() -> None:
    train_X, train_Y = _training_data()
    estimator = _FakeForest()
    config = ModelConfig(
        task_type="regression",
        model_type="random_forest",
        model_cls=RandomForestRegressorModel,
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    bundle = build_model(train_X, train_Y, config)

    fitted = fit_model(bundle, FitConfig())

    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
    assert estimator.fit_y is not None


def test_random_forest_passes_sample_weight() -> None:
    train_X, train_Y = _training_data()
    estimator = _FakeForest()
    weights = np.array([1.0, 2.0, 3.0])

    RandomForestRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
    ).fit(sample_weight=weights)

    np.testing.assert_array_equal(estimator.fit_kwargs["sample_weight"], weights)
