from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.ensemble import EnsembleModel
from botorch.posteriors.ensemble import EnsemblePosterior

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.regression.boosting import RandomForestMixedRegressorModel


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

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self


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


def test_mixed_random_forest_one_hot_encodes_only_for_estimator() -> None:
    train_X, train_Y = _mixed_training_data()
    estimator = _FakeForest()
    model = RandomForestMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert isinstance(model, EnsembleModel)
    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert model.categorical_encoder.encoded_dim == 4
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
        torch.tensor([[0.25, 0.0], [0.75, 2.0]], dtype=torch.double)
    )
    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])


def test_mixed_random_forest_rejects_unseen_category() -> None:
    train_X, train_Y = _mixed_training_data()
    model = RandomForestMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=_FakeForest(),
    ).fit()

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.4, 3.0]], dtype=torch.double))


def test_mixed_random_forest_works_with_standard_qlogei_sampler() -> None:
    train_X, train_Y = _mixed_training_data()
    model = RandomForestMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=_FakeForest(),
    ).fit()
    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())

    value = acqf(torch.tensor([[[0.8, 1.0]]], dtype=torch.double))

    assert torch.isfinite(value).all()


def test_mixed_random_forest_is_selected_by_cat_dims_from_default_registry() -> None:
    model_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="random_forest",
            cat_dims=[1],
            outcome_transform=False,
        )
    )

    assert model_cls is RandomForestMixedRegressorModel


def test_high_level_fit_path_builds_mixed_random_forest_from_cat_dims() -> None:
    train_X, train_Y = _mixed_training_data()
    estimator = _FakeForest()
    config = ModelConfig(
        task_type="regression",
        model_type="random_forest",
        cat_dims=[1],
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )

    bundle = build_model(train_X, train_Y, config)
    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, RandomForestMixedRegressorModel)
    assert fitted.input_type == "mixed"
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
    assert estimator.fit_X.shape[-1] == 4
