from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior

from bochan.acquisition.objective.ordinal import OrdinalExpectedUtilityMCObjective
from bochan.acquisition.ordinal.active_learning.single_output import (
    qOrdinalBALD,
    qOrdinalPredictiveEntropy,
)
from bochan.acquisition.ordinal.bayesian_optimization.single_output import (
    qOrdinalExpectedUtility,
    qOrdinalProbabilityOfFeasibility,
)
from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.ordinal.external import (
    NGBoostMixedOrdinalModel,
    NGBoostOrdinalEnsembleModel,
    NGBoostOrdinalModel,
    RandomForestMixedOrdinalModel,
    RandomForestOrdinalModel,
)
from bochan.models.ordinal.external.base import (
    _class_probs_from_cumulative,
    _cumulative_from_class_probs,
)
from bochan.posteriors.ordinal_ensemble import OrdinalEnsemblePosterior


class _FakeBinaryTree:
    def __init__(self, *, intercept: float, slope: float = 0.25) -> None:
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = np.clip(self.intercept + self.slope * X[:, 0], 0.01, 0.99)
        return np.column_stack([1.0 - p1, p1])


class _FakeForest:
    def __init__(self, intercepts) -> None:
        self.estimators_ = [
            _FakeBinaryTree(intercept=float(value))
            for value in intercepts
        ]
        self.classes_ = np.array([0, 1], dtype=int)
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self


class _FakeNGBoost:
    def __init__(self, *, intercept: float, slope: float = 0.2) -> None:
        self.intercept = float(intercept)
        self.slope = float(slope)
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
        p1 = np.clip(self.intercept + self.slope * X[:, 0], 0.01, 0.99)
        return np.column_stack([1.0 - p1, p1])


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [1], [1], [2], [2]], dtype=torch.long)
    return train_X, train_Y


def _mixed_training_data() -> tuple[torch.Tensor, torch.Tensor]:
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
    train_Y = torch.tensor([[0], [0], [1], [1], [2], [2]], dtype=torch.long)
    return train_X, train_Y


def _rf_estimators():
    return [
        _FakeForest([0.30, 0.45, 0.55]),
        _FakeForest([0.55, 0.35, 0.65]),
    ]


def _ng_estimators():
    return [
        _FakeNGBoost(intercept=0.35),
        _FakeNGBoost(intercept=0.60),
    ]


def test_cumulative_probability_projection_is_monotone_and_normalized() -> None:
    raw = np.array(
        [
            [0.30, 0.70],
            [0.90, 0.20],
        ]
    )
    probs = _class_probs_from_cumulative(raw)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2))
    assert np.all(probs >= 0.0)

    cumulative = _cumulative_from_class_probs(torch.tensor(probs))
    assert torch.all(cumulative[..., :-1] >= cumulative[..., 1:])


def test_random_forest_ordinal_exposes_exact_probability_ensemble_and_latent_bridge() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=_rf_estimators(),
    ).fit()

    X = torch.tensor([[0.35], [0.75]], dtype=torch.double)
    probability_posterior = model.ordinal_probability_posterior(X)
    assert isinstance(probability_posterior, OrdinalEnsemblePosterior)
    assert probability_posterior.values.shape == torch.Size([3, 2, 3])
    torch.testing.assert_close(
        probability_posterior.mean.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    assert torch.all(probability_posterior.epistemic_variance >= 0.0)

    posterior = model.posterior(X)
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_ngboost_single_and_ensemble_ordinal_probability_members() -> None:
    train_X, train_Y = _training_data()
    single = NGBoostOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=_ng_estimators(),
    ).fit()
    single_prob = single.ordinal_probability_posterior(
        torch.tensor([[0.35]], dtype=torch.double)
    )
    assert single_prob.values.shape == torch.Size([1, 1, 3])
    torch.testing.assert_close(
        single_prob.epistemic_variance,
        torch.zeros_like(single_prob.epistemic_variance),
    )

    ensemble = NGBoostOrdinalEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            [
                _FakeNGBoost(intercept=0.25),
                _FakeNGBoost(intercept=0.55),
            ],
            [
                _FakeNGBoost(intercept=0.45),
                _FakeNGBoost(intercept=0.50),
            ],
            [
                _FakeNGBoost(intercept=0.60),
                _FakeNGBoost(intercept=0.30),
            ],
        ],
        bootstrap=False,
    ).fit()
    ensemble_prob = ensemble.ordinal_probability_posterior(
        torch.tensor([[0.35]], dtype=torch.double)
    )
    assert ensemble_prob.values.shape == torch.Size([3, 1, 3])
    assert torch.any(ensemble_prob.epistemic_variance > 0.0)


def test_random_forest_ordinal_works_with_existing_active_learning_acquisitions() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=_rf_estimators(),
    ).fit()
    X = torch.tensor([[[0.35]]], dtype=torch.double)

    entropy = qOrdinalPredictiveEntropy(
        model=model,
        exclude_observed_duplicates=False,
    )(X)
    bald = qOrdinalBALD(
        model=model,
        num_samples=8,
        exclude_observed_duplicates=False,
    )(X)
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(bald).all()


def test_random_forest_ordinal_works_with_existing_utility_bo_and_exact_pof() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=_rf_estimators(),
    ).fit()
    utilities = torch.tensor([0.0, 1.0, 3.0], dtype=torch.double)
    objective = OrdinalExpectedUtilityMCObjective(
        ordinal_likelihood=model.ordinal_likelihood,
        utility_values=utilities,
    )
    X = torch.tensor([[[0.35]]], dtype=torch.double)

    expected_utility = qOrdinalExpectedUtility(
        model=model,
        objective=objective,
    )(X)
    feasibility = qOrdinalProbabilityOfFeasibility(
        model=model,
        mode="class_ge",
        min_class=2,
    )(X)
    assert torch.isfinite(expected_utility).all()
    assert torch.isfinite(feasibility).all()
    assert torch.all((feasibility >= 0.0) & (feasibility <= 1.0))


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("random_forest", RandomForestOrdinalModel),
        ("ngboost", NGBoostOrdinalModel),
        ("ngboost_ensemble", NGBoostOrdinalEnsembleModel),
    ],
)
def test_default_registry_resolves_normal_external_ordinal_models(
    model_type,
    expected_cls,
) -> None:
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="ordinal",
            model_type=model_type,
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


def test_default_registry_resolves_mixed_external_ordinal_models() -> None:
    rf_cls = resolve_model_cls(
        ModelConfig(
            task_type="ordinal",
            model_type="random_forest",
            cat_dims=[1],
            outcome_transform=False,
        )
    )
    ng_cls = resolve_model_cls(
        ModelConfig(
            task_type="ordinal",
            model_type="ngboost",
            cat_dims=[1],
            outcome_transform=False,
        )
    )
    assert rf_cls is RandomForestMixedOrdinalModel
    assert ng_cls is NGBoostMixedOrdinalModel


def test_high_level_fit_path_uses_external_ordinal_fit() -> None:
    train_X, train_Y = _training_data()
    estimators = _rf_estimators()
    config = ModelConfig(
        task_type="ordinal",
        model_type="random_forest",
        outcome_transform=False,
        model_kwargs={"estimators": estimators},
    )
    bundle = build_model(train_X, train_Y, config)
    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, RandomForestOrdinalModel)
    assert fitted.mll is None
    assert fitted.model.is_fitted
    assert estimators[0].fit_X is not None
    np.testing.assert_array_equal(
        estimators[0].fit_y,
        np.array([0, 0, 1, 1, 1, 1]),
    )
    np.testing.assert_array_equal(
        estimators[1].fit_y,
        np.array([0, 0, 0, 0, 1, 1]),
    )


@pytest.mark.parametrize(
    "model_cls",
    [RandomForestMixedOrdinalModel, NGBoostMixedOrdinalModel],
)
def test_mixed_ordinal_one_hot_encodes_only_at_estimator_boundary(model_cls) -> None:
    train_X, train_Y = _mixed_training_data()
    estimators = (
        _rf_estimators()
        if model_cls is RandomForestMixedOrdinalModel
        else _ng_estimators()
    )

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimators=estimators,
    ).fit()

    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    assert estimators[0].fit_X.shape == (6, 4)

    posterior = model.ordinal_probability_posterior(
        torch.tensor([[0.35, 1.0]], dtype=torch.double)
    )
    assert posterior.mean.shape == torch.Size([1, 3])


def test_mixed_ordinal_rejects_unseen_category() -> None:
    train_X, train_Y = _mixed_training_data()
    model = RandomForestMixedOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimators=_rf_estimators(),
    ).fit()

    with pytest.raises(ValueError, match="not observed during training"):
        model.class_probs(torch.tensor([[0.35, 3.0]], dtype=torch.double))


def test_exact_expected_utility_uses_probability_posterior() -> None:
    train_X, train_Y = _training_data()
    model = RandomForestOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=_rf_estimators(),
    ).fit()
    X = torch.tensor([[0.35]], dtype=torch.double)
    utilities = torch.tensor([0.0, 2.0, 5.0], dtype=torch.double)

    expected = (model.class_probs(X) * utilities).sum(dim=-1)
    torch.testing.assert_close(model.expected_utility(X, utilities), expected)
