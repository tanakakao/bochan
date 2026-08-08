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
from bochan.models.classification.external import (
    NGBoostMixedMulticlassClassificationModel,
    NGBoostMixedMulticlassEnsembleModel,
    NGBoostMulticlassClassificationModel,
    NGBoostMulticlassEnsembleModel,
    RandomForestMixedMulticlassClassificationModel,
    RandomForestMulticlassClassificationModel,
)
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior


class _FakeMulticlassMember:
    def __init__(self, *, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        X = np.asarray(X)
        x = X[:, 0]
        scores = np.stack(
            [
                0.6 + 0.25 * (1.0 - x) - self.offset,
                0.5 + 0.20 * x + self.offset,
                0.35 + 0.10 * np.sin(np.pi * x),
            ],
            axis=-1,
        )
        scores = np.clip(scores, 1e-4, None)
        return scores / scores.sum(axis=-1, keepdims=True)


class _FakeForestMulticlass:
    def __init__(self, offsets=(-0.08, 0.0, 0.08)) -> None:
        self.estimators_ = [_FakeMulticlassMember(offset=value) for value in offsets]
        self.classes_ = np.array([0, 1, 2])
        self.fit_X = None
        self.fit_y = None
        self.fit_kwargs = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        self.classes_ = np.unique(self.fit_y)
        return self


class _FakeNGBoostMulticlass(_FakeMulticlassMember):
    def __init__(self, *, offset: float = 0.0) -> None:
        super().__init__(offset=offset)
        self.fit_X = None
        self.fit_y = None
        self.fit_kwargs = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        self.classes_ = np.unique(self.fit_y)
        return self


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.0], [0.1], [0.35], [0.5], [0.7], [0.9], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [1], [1], [2], [2], [2]], dtype=torch.double)
    return train_X, train_Y


def _mixed_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.15, 1.0],
            [0.3, 2.0],
            [0.45, 0.0],
            [0.6, 1.0],
            [0.75, 2.0],
            [0.9, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [1], [1], [2], [2], [2], [1]], dtype=torch.double)
    return train_X, train_Y


def _assert_multiclass_probability_posterior(model) -> ClassificationEnsemblePosterior:
    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    posterior = model.posterior(X)

    assert isinstance(posterior, ClassificationEnsemblePosterior)
    assert posterior.mean.shape == torch.Size([2, 3])
    assert posterior.variance.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        posterior.mean.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    torch.testing.assert_close(
        posterior.variance,
        posterior.mean * (1.0 - posterior.mean),
    )
    samples = posterior.rsample(torch.Size([16]))
    assert samples.shape == torch.Size([16, 2, 3])
    torch.testing.assert_close(
        samples.sum(dim=-1),
        torch.ones(16, 2, dtype=torch.double),
    )
    return posterior


def test_random_forest_multiclass_uses_tree_probability_samples() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForestMulticlass(),
    ).fit()

    posterior = _assert_multiclass_probability_posterior(model)

    assert posterior.values.shape == torch.Size([3, 2, 3])
    assert torch.any(posterior.epistemic_variance > 0)
    assert model.predict_class(torch.tensor([[0.2], [0.8]], dtype=torch.double)).shape == torch.Size([2])


def test_ngboost_multiclass_single_and_ensemble_probability_semantics() -> None:
    train_X, train_Y = _training_data()
    single = NGBoostMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeNGBoostMulticlass(offset=0.0),
    ).fit()
    ensemble = NGBoostMulticlassEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeNGBoostMulticlass(offset=-0.08),
            _FakeNGBoostMulticlass(offset=0.0),
            _FakeNGBoostMulticlass(offset=0.08),
        ],
        bootstrap=False,
    ).fit()

    single_posterior = _assert_multiclass_probability_posterior(single)
    ensemble_posterior = _assert_multiclass_probability_posterior(ensemble)

    torch.testing.assert_close(
        single_posterior.epistemic_variance,
        torch.zeros_like(single_posterior.epistemic_variance),
    )
    assert torch.any(ensemble_posterior.epistemic_variance > 0)


def test_multiclass_external_model_has_gaussian_log_probability_acquisition_bridge() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeForestMulticlass(),
    ).fit()
    X = torch.tensor([[[0.35]], [[0.65]]], dtype=torch.double)

    latent = model.latent_posterior(X)
    latent_samples = latent.rsample(torch.Size([8]))
    assert latent.mean.shape == torch.Size([2, 1, 3])
    assert latent_samples.shape == torch.Size([8, 2, 1, 3])
    assert torch.isfinite(latent_samples).all()

    bald = qMulticlassBALD(model=model, num_samples=16)
    probability_variance = qMulticlassProbabilityVariance(model=model, num_samples=16)
    bald_value = bald(X)
    variance_value = probability_variance(X)

    assert torch.isfinite(bald_value).all()
    assert torch.isfinite(variance_value).all()


def test_mixed_multiclass_models_share_one_hot_encoder() -> None:
    train_X, train_Y = _mixed_training_data()
    rf_estimator = _FakeForestMulticlass()
    ngb_estimators = [
        _FakeNGBoostMulticlass(offset=-0.05),
        _FakeNGBoostMulticlass(offset=0.05),
    ]

    rf = RandomForestMixedMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=rf_estimator,
    ).fit()
    ngb = NGBoostMixedMulticlassEnsembleModel(
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
    torch.testing.assert_close(rf.posterior(X).mean.sum(dim=-1), torch.ones(1, dtype=torch.double))
    torch.testing.assert_close(ngb.posterior(X).mean.sum(dim=-1), torch.ones(1, dtype=torch.double))

    unseen = torch.tensor([[0.5, 3.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="not observed during training"):
        rf.posterior(unseen)
    with pytest.raises(ValueError, match="not observed during training"):
        ngb.posterior(unseen)


def test_multiclass_external_models_are_available_from_normal_and_mixed_registry() -> None:
    assert (
        resolve_model_cls(ModelConfig(task_type="multiclass", model_type="random_forest"))
        is RandomForestMulticlassClassificationModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="multiclass", model_type="ngboost"))
        is NGBoostMulticlassClassificationModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="multiclass", model_type="ngboost_ensemble"))
        is NGBoostMulticlassEnsembleModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="multiclass", model_type="random_forest", cat_dims=[1]))
        is RandomForestMixedMulticlassClassificationModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="multiclass", model_type="ngboost", cat_dims=[1]))
        is NGBoostMixedMulticlassClassificationModel
    )
    assert (
        resolve_model_cls(ModelConfig(task_type="multiclass", model_type="ngboost_ensemble", cat_dims=[1]))
        is NGBoostMixedMulticlassEnsembleModel
    )


def test_high_level_multiclass_build_and_fit_routes_external_estimator() -> None:
    train_X, train_Y = _training_data()
    estimator = _FakeNGBoostMulticlass()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="multiclass",
            model_type="ngboost",
            model_kwargs={"estimator": estimator},
        ),
    )

    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, NGBoostMulticlassClassificationModel)
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
    assert estimator.fit_y is not None
