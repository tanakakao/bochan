from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.acquisition.binary.active_learning.single_output import (
    qBinaryBALD,
    qBinaryProbabilityVariance,
)
from bochan.acquisition.binary.epistemic import binary_probability_moments
from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.external import (
    NGBoostBinaryClassificationModel,
    NGBoostBinaryEnsembleModel,
    NGBoostMixedBinaryClassificationModel,
    NGBoostMixedBinaryEnsembleModel,
    RandomForestBinaryClassificationModel,
    RandomForestMixedBinaryClassificationModel,
)
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior


class _FakeProbabilityMember:
    def __init__(self, *, num_classes: int = 2, offset: float = 0.0) -> None:
        self.num_classes = int(num_classes)
        self.offset = float(offset)
        self.classes_ = np.arange(self.num_classes)

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.num_classes != 2:
            raise RuntimeError("This fake member is configured for binary tests.")
        p1 = np.clip(0.2 + 0.5 * X[:, 0] + self.offset, 0.02, 0.98)
        return np.stack([1.0 - p1, p1], axis=-1)


class _FakeForestClassifier:
    def __init__(self, offsets=(-0.1, 0.0, 0.1)) -> None:
        self.estimators_ = [_FakeProbabilityMember(offset=value) for value in offsets]
        self.classes_ = np.array([0, 1])
        self.fit_X = None
        self.fit_y = None
        self.fit_kwargs = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        self.classes_ = np.unique(self.fit_y)
        return self


class _FakeNGBoostClassifier(_FakeProbabilityMember):
    def __init__(self, *, offset: float = 0.0) -> None:
        super().__init__(num_classes=2, offset=offset)
        self.fit_X = None
        self.fit_y = None
        self.fit_kwargs = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        self.classes_ = np.unique(self.fit_y)
        return self


def _binary_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor([[0.0], [0.25], [0.75], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.double)
    return train_X, train_Y


def _mixed_binary_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 2.0],
            [0.6, 0.0],
            [0.8, 1.0],
            [1.0, 2.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.double)
    return train_X, train_Y


def _assert_binary_posterior_decomposition(model) -> None:
    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    posterior = model.posterior(X)

    assert isinstance(posterior, ClassificationEnsemblePosterior)
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert posterior.epistemic_variance.shape == torch.Size([2, 1])
    assert posterior.aleatoric_variance.shape == torch.Size([2, 1])
    torch.testing.assert_close(
        posterior.variance,
        posterior.mean * (1.0 - posterior.mean),
    )
    torch.testing.assert_close(
        posterior.aleatoric_variance + posterior.epistemic_variance,
        posterior.total_label_variance,
        atol=1e-12,
        rtol=1e-12,
    )
    probs = model.class_probs(X)
    assert probs.shape == torch.Size([2, 2])
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, dtype=torch.double))


def test_random_forest_binary_uses_tree_probability_disagreement() -> None:
    train_X, train_Y = _binary_training_data()
    model = RandomForestBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForestClassifier(),
    ).fit()

    _assert_binary_posterior_decomposition(model)
    posterior = model.posterior(torch.tensor([[0.2], [0.8]], dtype=torch.double))
    assert torch.all(posterior.epistemic_variance > 0)
    assert posterior.values.shape == torch.Size([3, 2, 1])


def test_ngboost_binary_single_has_zero_member_disagreement() -> None:
    train_X, train_Y = _binary_training_data()
    model = NGBoostBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeNGBoostClassifier(offset=0.0),
    ).fit()

    _assert_binary_posterior_decomposition(model)
    posterior = model.posterior(torch.tensor([[0.2], [0.8]], dtype=torch.double))
    torch.testing.assert_close(
        posterior.epistemic_variance,
        torch.zeros_like(posterior.epistemic_variance),
    )


def test_ngboost_binary_ensemble_exposes_epistemic_probability_samples() -> None:
    train_X, train_Y = _binary_training_data()
    model = NGBoostBinaryEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeNGBoostClassifier(offset=-0.1),
            _FakeNGBoostClassifier(offset=0.0),
            _FakeNGBoostClassifier(offset=0.1),
        ],
        bootstrap=False,
    ).fit()

    posterior = model.epistemic_probability_posterior(
        torch.tensor([[0.2], [0.8]], dtype=torch.double)
    )
    assert posterior.values.shape == torch.Size([3, 2, 1])
    assert torch.all(posterior.epistemic_variance > 0)


def test_external_binary_models_work_with_existing_epistemic_utilities_and_bald() -> None:
    train_X, train_Y = _binary_training_data()
    model = RandomForestBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForestClassifier(offsets=(-0.15, 0.0, 0.15)),
    ).fit()
    X = torch.tensor([[[0.35]], [[0.65]]], dtype=torch.double)

    mean, epistemic, aleatoric, total = binary_probability_moments(
        model,
        X.squeeze(-2),
        num_samples=128,
    )
    assert torch.isfinite(mean).all()
    assert torch.isfinite(epistemic).all()
    assert torch.isfinite(aleatoric).all()
    assert torch.isfinite(total).all()
    assert torch.any(epistemic > 0)

    bald = qBinaryBALD(model=model, num_samples=32)
    probability_variance = qBinaryProbabilityVariance(model=model, num_samples=64)
    bald_value = bald(X)
    variance_value = probability_variance(X)

    assert torch.isfinite(bald_value).all()
    assert torch.isfinite(variance_value).all()


def test_mixed_binary_models_share_one_hot_encoder_and_reject_unseen_categories() -> None:
    train_X, train_Y = _mixed_binary_training_data()
    rf_estimator = _FakeForestClassifier()
    ngb_estimators = [
        _FakeNGBoostClassifier(offset=-0.05),
        _FakeNGBoostClassifier(offset=0.05),
    ]

    rf = RandomForestMixedBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=rf_estimator,
    ).fit()
    ngb = NGBoostMixedBinaryEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimators=ngb_estimators,
        bootstrap=False,
    ).fit()

    assert rf_estimator.fit_X.shape[1] == 4
    assert ngb_estimators[0].fit_X.shape[1] == 4
    assert rf.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert ngb.categorical_values == {1: (0.0, 1.0, 2.0)}

    X = torch.tensor([[0.5, 1.0]], dtype=torch.double)
    assert torch.isfinite(rf.posterior(X).mean).all()
    assert torch.isfinite(ngb.posterior(X).mean).all()

    unseen = torch.tensor([[0.5, 3.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="not observed during training"):
        rf.posterior(unseen)
    with pytest.raises(ValueError, match="not observed during training"):
        ngb.posterior(unseen)


def test_binary_external_models_are_available_from_normal_and_mixed_registry() -> None:
    assert resolve_model_cls(ModelConfig(task_type="binary", model_type="random_forest")) is RandomForestBinaryClassificationModel
    assert resolve_model_cls(ModelConfig(task_type="binary", model_type="ngboost")) is NGBoostBinaryClassificationModel
    assert resolve_model_cls(ModelConfig(task_type="binary", model_type="ngboost_ensemble")) is NGBoostBinaryEnsembleModel
    assert (
        resolve_model_cls(ModelConfig(task_type="binary", model_type="random_forest", cat_dims=[1]))
        is RandomForestMixedBinaryClassificationModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="binary", model_type="ngboost", cat_dims=[1]))
        is NGBoostMixedBinaryClassificationModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="binary", model_type="ngboost_ensemble", cat_dims=[1]))
        is NGBoostMixedBinaryEnsembleModel
    )


def test_high_level_binary_build_and_fit_routes_external_estimators() -> None:
    train_X, train_Y = _binary_training_data()
    estimator = _FakeForestClassifier()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="binary",
            model_type="random_forest",
            model_kwargs={"estimator": estimator},
        ),
    )

    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, RandomForestBinaryClassificationModel)
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
    assert estimator.fit_y is not None
