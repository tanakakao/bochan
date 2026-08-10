from __future__ import annotations

import numpy as np
import torch

from bochan.models.classification.binary.foundation import (
    TabPFNBinaryClassificationModel,
    TabPFNMixedBinaryClassificationModel,
)
from bochan.models.regression.foundation import (
    TabPFNMixedRegressorModel,
    TabPFNRegressorModel,
)
from bochan.models.transforms.input import build_input_transform


class _FakeBarCriterion:
    def __init__(self) -> None:
        self.borders = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        centers = torch.tensor([-0.5, 0.5, 1.5], device=logits.device, dtype=logits.dtype)
        mean = (probs * centers).sum(dim=-1)
        second = (probs * centers.square()).sum(dim=-1)
        return (second - mean.square()).clamp_min(1e-6)


class _FakeTabPFNRegressor:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.predict_X: np.ndarray | None = None
        self.criterion = _FakeBarCriterion()

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X, *, output_type="mean"):
        X = np.asarray(X)
        self.predict_X = X.copy()
        if output_type != "full":
            return X[:, 0]
        logits = torch.tensor(
            np.column_stack(
                [
                    0.2 + X[:, 0],
                    0.7 - 0.2 * X[:, 0],
                    0.1 + 0.1 * X[:, 0],
                ]
            ),
            dtype=torch.float32,
        )
        return {
            "mean": X[:, 0],
            "criterion": self.criterion,
            "logits": logits,
        }


class _FakeTabPFNClassifier:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.predict_X: np.ndarray | None = None
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        self.predict_X = X.copy()
        p1 = np.clip(0.15 + 0.7 * X[:, 0], 0.01, 0.99)
        return np.column_stack([1.0 - p1, p1])


def _transform(
    train_X: torch.Tensor,
    *,
    n_w: int,
    categorical_idx: list[int] | None = None,
):
    bounds = torch.stack(
        [train_X.min(dim=0).values, train_X.max(dim=0).values],
        dim=0,
    )
    return build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=True,
        categorical_idx=categorical_idx,
        n_w=n_w,
        std=0.05,
        normalize=True,
    )


def test_tabpfn_regression_fit_is_nominal_and_posterior_is_perturbation_expanded() -> None:
    torch.manual_seed(0)
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = (train_X + 0.1).clone()
    n_w = 4
    estimator = _FakeTabPFNRegressor()
    model = TabPFNRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
        input_transform=_transform(train_X, n_w=n_w),
    ).fit()

    assert estimator.fit_X is not None
    assert estimator.fit_X.shape == (len(train_X), 1)

    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    posterior = model.posterior(X)

    assert posterior.mean.shape == torch.Size([len(X) * n_w, 1])
    assert posterior.variance.shape == posterior.mean.shape
    assert estimator.predict_X is not None
    assert estimator.predict_X.shape[0] == len(X) * n_w
    assert torch.isfinite(posterior.mean).all()
    assert torch.all(posterior.variance > 0)


def test_tabpfn_binary_probability_posterior_is_perturbation_expanded() -> None:
    torch.manual_seed(0)
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    n_w = 3
    estimator = _FakeTabPFNClassifier()
    model = TabPFNBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
        input_transform=_transform(train_X, n_w=n_w),
    ).fit()

    assert estimator.fit_X is not None
    assert estimator.fit_X.shape[0] == len(train_X)

    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(X)

    assert posterior.mean.shape == torch.Size([len(X) * n_w, 1])
    assert estimator.predict_X is not None
    assert estimator.predict_X.shape[0] == len(X) * n_w
    assert torch.all((posterior.mean >= 0) & (posterior.mean <= 1))


def test_tabpfn_mixed_perturbation_preserves_native_categorical_values() -> None:
    torch.manual_seed(0)
    train_X = torch.tensor(
        [
            [0.0, 10.0],
            [0.2, 20.0],
            [0.4, 30.0],
            [0.6, 10.0],
            [0.8, 20.0],
            [1.0, 30.0],
        ],
        dtype=torch.double,
    )
    train_Y_reg = train_X[:, :1] + 0.1
    train_Y_cls = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    transform = _transform(train_X, n_w=4, categorical_idx=[1])

    reg_estimator = _FakeTabPFNRegressor()
    reg = TabPFNMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y_reg,
        cat_dims=[1],
        estimator=reg_estimator,
        input_transform=transform,
    ).fit()
    reg.posterior(torch.tensor([[0.5, 20.0]], dtype=torch.double))

    cls_estimator = _FakeTabPFNClassifier()
    cls = TabPFNMixedBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y_cls,
        cat_dims=[1],
        estimator=cls_estimator,
        input_transform=transform,
    ).fit()
    cls.posterior(torch.tensor([[0.5, 20.0]], dtype=torch.double))

    assert reg_estimator.predict_X is not None
    assert cls_estimator.predict_X is not None
    np.testing.assert_array_equal(reg_estimator.predict_X[:, 1], np.ones(4))
    np.testing.assert_array_equal(cls_estimator.predict_X[:, 1], np.ones(4))
    assert reg_estimator.categorical_features_indices == [1]
    assert cls_estimator.categorical_features_indices == [1]
