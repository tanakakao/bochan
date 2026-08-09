from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.models.ordinal.external import RandomForestOrdinalModel
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

    def fit(self, X, y, **kwargs):
        del X, y, kwargs
        return self


def _rf_model() -> RandomForestOrdinalModel:
    train_X = torch.tensor(
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [1], [1], [2], [2]], dtype=torch.long)
    estimators = [
        _FakeForest([0.30, 0.45, 0.55]),
        _FakeForest([0.55, 0.35, 0.65]),
    ]
    return RandomForestOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=estimators,
    ).fit()


def test_random_forest_ordinal_expected_utility_matches_probability_mean() -> None:
    model = _rf_model()
    X = torch.tensor([[0.35], [0.75]], dtype=torch.double)
    utilities = torch.tensor([0.0, 1.0, 3.0], dtype=torch.double)

    posterior = model.ordinal_probability_posterior(X)
    expected = (posterior.mean * utilities).sum(dim=-1)

    actual = model.expected_utility(X, utilities)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == torch.Size([2])
    assert torch.isfinite(actual).all()


def test_ordinal_probability_posterior_exposes_complete_utility_contract() -> None:
    values = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.20, 0.50, 0.30]],
            [[0.50, 0.30, 0.20], [0.10, 0.40, 0.50]],
            [[0.60, 0.25, 0.15], [0.30, 0.30, 0.40]],
        ],
        dtype=torch.double,
    )
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)
    utilities = torch.tensor([0.0, 1.0, 3.0], dtype=torch.double)
    posterior = OrdinalEnsemblePosterior(values=values, weights=weights)

    member_utility = (values * utilities).sum(dim=-1)
    expected_member_utility = member_utility
    expected_utility = (posterior.mean * utilities).sum(dim=-1)
    expected_utility_variance = (
        weights[:, None]
        * (member_utility - expected_utility.unsqueeze(0)).square()
    ).sum(dim=0)

    torch.testing.assert_close(posterior.class_probs(), posterior.mean)
    torch.testing.assert_close(posterior.expected_utility(utilities), expected_utility)
    torch.testing.assert_close(
        posterior.member_expected_utility(utilities),
        expected_member_utility,
    )
    torch.testing.assert_close(
        posterior.utility_epistemic_variance(utilities),
        expected_utility_variance,
    )
    torch.testing.assert_close(
        posterior.predict_class(),
        posterior.mean.argmax(dim=-1),
    )
    assert torch.all(posterior.epistemic_variance >= 0.0)


def test_ordinal_probability_utility_rejects_wrong_number_of_classes() -> None:
    posterior = OrdinalEnsemblePosterior(
        values=torch.tensor([[[0.6, 0.3, 0.1]]], dtype=torch.double)
    )

    with pytest.raises(ValueError, match="utilities must have length 3"):
        posterior.expected_utility(torch.tensor([0.0, 1.0], dtype=torch.double))

    with pytest.raises(ValueError, match="utilities must have length 3"):
        posterior.member_expected_utility(torch.tensor([0.0, 1.0], dtype=torch.double))
