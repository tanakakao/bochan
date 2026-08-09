from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.binary.foundation import (
    TabPFNBinaryClassificationModel,
    TabPFNMixedBinaryClassificationModel,
)
from bochan.models.classification.multiclass.foundation import (
    TabPFNMixedMulticlassClassificationModel,
    TabPFNMulticlassClassificationModel,
)
from bochan.models.regression.foundation import (
    TabPFNMixedRegressorModel,
    TabPFNRegressorModel,
)


class _FakeBarCriterion:
    def __init__(self) -> None:
        self.borders = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        self.calls = 0

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        probs = logits.softmax(dim=-1)
        centers = torch.tensor([-0.5, 0.5, 1.5], device=logits.device, dtype=logits.dtype)
        mean = (probs * centers).sum(dim=-1)
        second = (probs * centers.square()).sum(dim=-1)
        return (second - mean.square()).clamp_min(1e-6)


class _FakeTabPFNRegressor:
    def __init__(self) -> None:
        self.categorical_features_indices = None
        self.fit_X = None
        self.fit_y = None
        self.criterion = _FakeBarCriterion()
        self.predict_output_types: list[str] = []

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X, *, output_type="mean"):
        X = np.asarray(X)
        self.predict_output_types.append(str(output_type))
        if output_type != "full":
            return X[:, 0] + 0.25
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
            "mean": X[:, 0] + 0.25,
            "median": X[:, 0] + 0.2,
            "mode": X[:, 0] + 0.1,
            "quantiles": None,
            "criterion": self.criterion,
            "logits": logits,
        }


class _FakeTabPFNClassifier:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.categorical_features_indices = None
        self.fit_X = None
        self.fit_y = None
        self.classes_ = None

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.unique(self.fit_y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.num_classes == 2:
            p1 = np.clip(0.2 + 0.6 * X[:, 0], 0.01, 0.99)
            return np.column_stack([1.0 - p1, p1])

        raw = np.column_stack(
            [
                0.5 + 0.1 * X[:, 0],
                0.3 + 0.2 * X[:, 0],
                0.2 + 0.3 * X[:, 0],
            ]
        )
        return raw / raw.sum(axis=-1, keepdims=True)


def _regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.1], [0.3], [0.6], [0.8], [1.1]], dtype=torch.double)
    return X, Y


def _mixed_regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
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
    Y = torch.tensor([[0.1], [0.2], [0.4], [0.7], [0.9], [1.2]], dtype=torch.double)
    return X, Y


def _binary_data(*, mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if mixed:
        X, _ = _mixed_regression_data()
    else:
        X = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    return X, Y


def _multiclass_data(*, mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if mixed:
        X, _ = _mixed_regression_data()
    else:
        X = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0], [1], [2], [0], [1], [2]], dtype=torch.long)
    return X, Y


def test_tabpfn_regression_uses_full_bar_distribution_moments() -> None:
    train_X, train_Y = _regression_data()
    estimator = _FakeTabPFNRegressor()
    model = TabPFNRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
    ).fit()

    X = torch.tensor([[[0.35]], [[0.65]]], dtype=torch.double)
    posterior = model.posterior(X)

    assert model.is_fitted
    assert estimator.predict_output_types[-1] == "full"
    assert estimator.criterion.calls == 1
    assert posterior.mean.shape == torch.Size([2, 1, 1])
    assert posterior.variance.shape == torch.Size([2, 1, 1])
    torch.testing.assert_close(
        posterior.mean.squeeze(-1).squeeze(-1),
        torch.tensor([0.60, 0.90], dtype=torch.double),
    )
    assert torch.all(posterior.variance > 0.0)
    assert torch.isfinite(posterior.variance).all()

    full = model.tabpfn_distribution(X)
    assert full["criterion"] is estimator.criterion
    assert "logits" in full


def test_tabpfn_mixed_regression_preserves_width_and_native_categories() -> None:
    train_X, train_Y = _mixed_regression_data()
    estimator = _FakeTabPFNRegressor()
    model = TabPFNMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert estimator.fit_X.shape == (6, 2)
    np.testing.assert_array_equal(estimator.fit_X[:, 1], np.array([0, 1, 2, 0, 1, 2]))
    assert estimator.categorical_features_indices == [1]
    assert model.cat_dims == [1]
    assert model.categorical_values == {1: (10.0, 20.0, 30.0)}

    posterior = model.posterior(torch.tensor([[0.5, 20.0]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([1, 1])

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.5, 40.0]], dtype=torch.double))


def test_tabpfn_binary_probability_posterior_has_one_public_member() -> None:
    train_X, train_Y = _binary_data()
    estimator = _FakeTabPFNClassifier(num_classes=2)
    model = TabPFNBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
    ).fit()

    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)
    posterior = model.posterior(X)
    probs = model.class_probs(X)

    assert posterior.values.shape == torch.Size([1, 1, 2, 1])
    assert probs.shape == torch.Size([1, 2, 2])
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, 2, dtype=torch.double))
    torch.testing.assert_close(
        posterior.epistemic_variance,
        torch.zeros_like(posterior.epistemic_variance),
    )
    assert model.predict_class(X).shape == torch.Size([1, 2])


def test_tabpfn_mixed_binary_uses_native_categorical_indices() -> None:
    train_X, train_Y = _binary_data(mixed=True)
    estimator = _FakeTabPFNClassifier(num_classes=2)
    model = TabPFNMixedBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert estimator.fit_X.shape == (6, 2)
    np.testing.assert_array_equal(estimator.fit_X[:, 1], np.array([0, 1, 2, 0, 1, 2]))
    assert estimator.categorical_features_indices == [1]
    probs = model.class_probs(torch.tensor([[0.4, 30.0]], dtype=torch.double))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, dtype=torch.double))


def test_tabpfn_multiclass_probability_posterior_has_one_public_member() -> None:
    train_X, train_Y = _multiclass_data()
    estimator = _FakeTabPFNClassifier(num_classes=3)
    model = TabPFNMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
    ).fit()

    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)
    posterior = model.posterior(X)

    assert posterior.values.shape == torch.Size([1, 1, 2, 3])
    torch.testing.assert_close(
        posterior.mean.sum(dim=-1),
        torch.ones(1, 2, dtype=torch.double),
    )
    torch.testing.assert_close(
        posterior.epistemic_variance,
        torch.zeros_like(posterior.epistemic_variance),
    )


def test_tabpfn_mixed_multiclass_uses_native_categorical_indices() -> None:
    train_X, train_Y = _multiclass_data(mixed=True)
    estimator = _FakeTabPFNClassifier(num_classes=3)
    model = TabPFNMixedMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert estimator.fit_X.shape == (6, 2)
    assert estimator.categorical_features_indices == [1]
    probs = model.class_probs(torch.tensor([[0.4, 10.0]], dtype=torch.double))
    assert probs.shape == torch.Size([1, 3])
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, dtype=torch.double))


@pytest.mark.parametrize(
    ("input_type", "task_type", "expected_cls"),
    [
        ("normal", "regression", TabPFNRegressorModel),
        ("mixed", "regression", TabPFNMixedRegressorModel),
        ("normal", "binary", TabPFNBinaryClassificationModel),
        ("mixed", "binary", TabPFNMixedBinaryClassificationModel),
        ("normal", "multiclass", TabPFNMulticlassClassificationModel),
        ("mixed", "multiclass", TabPFNMixedMulticlassClassificationModel),
    ],
)
def test_tabpfn_registry_routes(input_type, task_type, expected_cls) -> None:
    cat_dims = [1] if input_type == "mixed" else []
    resolved = resolve_model_cls(
        ModelConfig(
            task_type=task_type,
            model_type="tabpfn",
            input_type=input_type,
            cat_dims=cat_dims,
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


def test_high_level_fit_path_uses_tabpfn_regression_fit() -> None:
    train_X, train_Y = _regression_data()
    estimator = _FakeTabPFNRegressor()
    config = ModelConfig(
        task_type="regression",
        model_type="tabpfn",
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    bundle = build_model(train_X, train_Y, config)
    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, TabPFNRegressorModel)
    assert fitted.mll is None
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None


def test_high_level_fit_path_uses_tabpfn_binary_external_protocol() -> None:
    train_X, train_Y = _binary_data()
    estimator = _FakeTabPFNClassifier(num_classes=2)
    config = ModelConfig(
        task_type="binary",
        model_type="tabpfn",
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    bundle = build_model(train_X, train_Y, config)
    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, TabPFNBinaryClassificationModel)
    assert fitted.mll is None
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
