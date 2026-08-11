from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.acquisition.binary.active_learning.single_output import (
    qBinaryBALD,
    qBinaryProbabilityVariance,
)
from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.binary.external import (
    NGBoostBinaryClassificationModel,
    NGBoostBinaryEnsembleModel,
    NGBoostMixedBinaryClassificationModel,
    RandomForestBinaryClassificationModel,
    RandomForestMixedBinaryClassificationModel,
)
from bochan.models.classification.common.posterior import ClassificationEnsemblePosterior


class _FakeProbabilityMember:
    def __init__(self, *, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.classes_ = np.array([0, 1], dtype=int)
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = np.clip(0.2 + 0.55 * X[:, 0] + self.offset, 0.01, 0.99)
        return np.column_stack([1.0 - p1, p1])


class _FakeForest:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1], dtype=int)
        self.estimators_ = [
            _FakeProbabilityMember(offset=-0.08),
            _FakeProbabilityMember(offset=0.0),
            _FakeProbabilityMember(offset=0.08),
        ]
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self


def _data():
    X = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    return X, Y


def _mixed_data():
    X = torch.tensor(
        [[0.0, 0.0], [0.2, 1.0], [0.4, 2.0], [0.6, 0.0], [0.8, 1.0], [1.0, 2.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    return X, Y


def test_random_forest_binary_probability_and_epistemic_variance() -> None:
    train_X, train_Y = _data()
    model = RandomForestBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForest(),
    ).fit()

    X = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    posterior = model.posterior(X)
    assert isinstance(posterior, ClassificationEnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    assert torch.all((posterior.mean >= 0.0) & (posterior.mean <= 1.0))
    assert torch.any(posterior.epistemic_variance > 0.0)
    torch.testing.assert_close(posterior.variance, posterior.mean * (1.0 - posterior.mean))


def test_ngboost_binary_single_and_ensemble_probability_members() -> None:
    train_X, train_Y = _data()
    single = NGBoostBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeProbabilityMember(),
    ).fit()
    single_post = single.posterior(torch.tensor([[0.4]], dtype=torch.double))
    assert single_post.values.shape == torch.Size([1, 1, 1])
    torch.testing.assert_close(
        single_post.epistemic_variance,
        torch.zeros_like(single_post.epistemic_variance),
    )

    ensemble = NGBoostBinaryEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeProbabilityMember(offset=-0.08),
            _FakeProbabilityMember(offset=0.08),
        ],
        bootstrap=False,
    ).fit()
    assert torch.any(
        ensemble.posterior(torch.tensor([[0.4]], dtype=torch.double)).epistemic_variance > 0.0
    )


def test_binary_external_models_work_with_existing_active_learning_acquisitions() -> None:
    train_X, train_Y = _data()
    model = RandomForestBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForest(),
    ).fit()
    X = torch.tensor([[[0.35]]], dtype=torch.double)

    bald = qBinaryBALD(
        model=model,
        num_samples=16,
        exclude_observed_duplicates=False,
    )(X)
    variance = qBinaryProbabilityVariance(
        model=model,
        exclude_observed_duplicates=False,
    )(X)
    assert torch.isfinite(bald).all()
    assert torch.isfinite(variance).all()


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("random_forest", RandomForestBinaryClassificationModel),
        ("ngboost", NGBoostBinaryClassificationModel),
        ("ngboost_ensemble", NGBoostBinaryEnsembleModel),
    ],
)
def test_binary_registry_resolves_task_local_models(model_type, expected_cls) -> None:
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="binary",
            model_type=model_type,
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


def test_binary_mixed_registry_and_encoding() -> None:
    train_X, train_Y = _mixed_data()
    estimator = _FakeForest()
    model = RandomForestMixedBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert estimator.fit_X.shape == (6, 4)
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="binary",
            model_type="ngboost",
            cat_dims=[1],
            outcome_transform=False,
        )
    )
    assert resolved is NGBoostMixedBinaryClassificationModel

    with pytest.raises(ValueError, match="not observed during training"):
        model.class_probs(torch.tensor([[0.5, 3.0]], dtype=torch.double))


def test_high_level_binary_external_fit() -> None:
    train_X, train_Y = _data()
    estimator = _FakeForest()
    config = ModelConfig(
        task_type="binary",
        model_type="random_forest",
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    fitted = fit_model(build_model(train_X, train_Y, config), FitConfig())

    assert isinstance(fitted.model, RandomForestBinaryClassificationModel)
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
