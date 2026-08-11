from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.acquisition.multiclass.active_learning.single_output import (
    qMulticlassBALD,
    qMulticlassProbabilityVariance,
)
from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.common.posterior import ClassificationEnsemblePosterior
from bochan.models.classification.multiclass.external import (
    NGBoostMixedMulticlassClassificationModel,
    NGBoostMulticlassClassificationModel,
    NGBoostMulticlassEnsembleModel,
    RandomForestMixedMulticlassClassificationModel,
    RandomForestMulticlassClassificationModel,
)


class _FakeMulticlassMember:
    def __init__(self, *, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.classes_ = np.array([0, 1, 2], dtype=int)
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        x = X[:, 0]
        logits = np.column_stack(
            [
                1.2 - x + self.offset,
                0.8 - np.abs(x - 0.5),
                0.2 + x - self.offset,
            ]
        )
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class _FakeMulticlassForest:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2], dtype=int)
        self.estimators_ = [
            _FakeMulticlassMember(offset=-0.12),
            _FakeMulticlassMember(offset=0.0),
            _FakeMulticlassMember(offset=0.12),
        ]
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self


def _data():
    X = torch.tensor(
        [[0.0], [0.15], [0.3], [0.5], [0.7], [0.85], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def _mixed_data():
    X = torch.tensor(
        [[0.0, 0.0], [0.15, 1.0], [0.3, 2.0], [0.5, 0.0], [0.7, 1.0], [0.85, 2.0], [1.0, 0.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def test_random_forest_multiclass_probability_ensemble() -> None:
    train_X, train_Y = _data()
    model = RandomForestMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        estimator=_FakeMulticlassForest(),
    ).fit()

    X = torch.tensor([[0.35], [0.75]], dtype=torch.double)
    posterior = model.posterior(X)
    assert isinstance(posterior, ClassificationEnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 3])
    torch.testing.assert_close(
        posterior.mean.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    assert torch.any(posterior.epistemic_variance > 0.0)


def test_ngboost_multiclass_single_and_ensemble() -> None:
    train_X, train_Y = _data()
    single = NGBoostMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        estimator=_FakeMulticlassMember(),
    ).fit()
    single_post = single.posterior(torch.tensor([[0.4]], dtype=torch.double))
    assert single_post.values.shape == torch.Size([1, 1, 3])
    torch.testing.assert_close(
        single_post.epistemic_variance,
        torch.zeros_like(single_post.epistemic_variance),
    )

    ensemble = NGBoostMulticlassEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        estimators=[
            _FakeMulticlassMember(offset=-0.1),
            _FakeMulticlassMember(offset=0.1),
        ],
        bootstrap=False,
    ).fit()
    assert torch.any(
        ensemble.posterior(torch.tensor([[0.4]], dtype=torch.double)).epistemic_variance > 0.0
    )


def test_multiclass_external_models_work_with_existing_active_learning() -> None:
    train_X, train_Y = _data()
    model = RandomForestMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        estimator=_FakeMulticlassForest(),
    ).fit()
    X = torch.tensor([[[0.42]]], dtype=torch.double)

    bald = qMulticlassBALD(
        model=model,
        num_samples=16,
        exclude_observed_duplicates=False,
    )(X)
    variance = qMulticlassProbabilityVariance(
        model=model,
        exclude_observed_duplicates=False,
    )(X)
    assert torch.isfinite(bald).all()
    assert torch.isfinite(variance).all()


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("random_forest", RandomForestMulticlassClassificationModel),
        ("ngboost", NGBoostMulticlassClassificationModel),
        ("ngboost_ensemble", NGBoostMulticlassEnsembleModel),
    ],
)
def test_multiclass_registry_resolves_task_local_models(model_type, expected_cls) -> None:
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="multiclass",
            model_type=model_type,
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


def test_multiclass_mixed_registry_and_encoding() -> None:
    train_X, train_Y = _mixed_data()
    estimator = _FakeMulticlassForest()
    model = RandomForestMixedMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        num_classes=3,
        estimator=estimator,
    ).fit()

    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert estimator.fit_X.shape == (7, 4)
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="multiclass",
            model_type="ngboost",
            cat_dims=[1],
            outcome_transform=False,
        )
    )
    assert resolved is NGBoostMixedMulticlassClassificationModel

    with pytest.raises(ValueError, match="not observed during training"):
        model.class_probs(torch.tensor([[0.5, 3.0]], dtype=torch.double))


def test_high_level_multiclass_external_fit() -> None:
    train_X, train_Y = _data()
    estimator = _FakeMulticlassForest()
    config = ModelConfig(
        task_type="multiclass",
        model_type="random_forest",
        outcome_transform=False,
        model_kwargs={"num_classes": 3, "estimator": estimator},
    )
    fitted = fit_model(build_model(train_X, train_Y, config), FitConfig())

    assert isinstance(fitted.model, RandomForestMulticlassClassificationModel)
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
