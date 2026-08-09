from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from bochan.acquisition.binary.active_learning.single_output import qBinaryBALD
from bochan.acquisition.multiclass.active_learning.single_output import qMulticlassBALD
from bochan.acquisition.ordinal.active_learning.single_output import qOrdinalPredictiveEntropy
from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.binary.external import (
    LightGBMBinaryClassificationModel,
    LightGBMBinaryEnsembleModel,
    LightGBMMixedBinaryClassificationModel,
    LightGBMMixedBinaryEnsembleModel,
)
from bochan.models.classification.multiclass.external import (
    LightGBMMixedMulticlassClassificationModel,
    LightGBMMixedMulticlassEnsembleModel,
    LightGBMMulticlassClassificationModel,
    LightGBMMulticlassEnsembleModel,
)
from bochan.models.ordinal.external import (
    LightGBMMixedOrdinalEnsembleModel,
    LightGBMMixedOrdinalModel,
    LightGBMOrdinalEnsembleModel,
    LightGBMOrdinalModel,
)
from bochan.models.regression.external import (
    LightGBMEnsembleModel,
    LightGBMMixedEnsembleModel,
    LightGBMMixedRegressorModel,
    LightGBMRegressorModel,
)
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior
from bochan.posteriors.ordinal_ensemble import OrdinalEnsemblePosterior


class _FakeLGBMRegressor:
    def __init__(self, *, bias: float = 0.0) -> None:
        self.bias = float(bias)
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.bias


class _FakeLGBMClassifier:
    def __init__(self, *, num_classes: int = 2, offset: float = 0.0) -> None:
        self.num_classes = int(num_classes)
        self.offset = float(offset)
        self.classes_ = np.arange(self.num_classes)
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        self.classes_ = np.arange(self.num_classes)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        x = X[:, 0]
        if self.num_classes == 2:
            p1 = np.clip(0.15 + 0.65 * x + self.offset, 0.02, 0.98)
            return np.column_stack([1.0 - p1, p1])

        scores = np.column_stack(
            [
                1.2 - x + self.offset,
                np.full_like(x, 0.7),
                0.3 + x - self.offset,
            ]
        )
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)


def _regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)
    Y = (2.0 * X + 0.2).clone()
    return X, Y


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    return X, Y


def _multiclass_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.1], [0.2], [0.4], [0.5], [0.6], [0.8], [0.9], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def _mixed_regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 10.0], [0.2, 20.0], [0.4, 30.0], [0.6, 10.0], [0.8, 20.0], [1.0, 30.0]],
        dtype=torch.double,
    )
    Y = (X[:, :1] + 0.01 * X[:, 1:2]).clone()
    return X, Y


def _mixed_classification_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [
            [0.0, 10.0],
            [0.15, 20.0],
            [0.3, 30.0],
            [0.45, 10.0],
            [0.6, 20.0],
            [0.75, 30.0],
            [0.9, 10.0],
            [1.0, 20.0],
            [0.85, 30.0],
        ],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def test_single_lightgbm_regressor_exposes_deterministic_gpytorch_posterior() -> None:
    train_X, train_Y = _regression_data()
    estimator = _FakeLGBMRegressor(bias=0.5)
    model = LightGBMRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=estimator,
        min_variance=1e-6,
    ).fit()

    posterior = model.posterior(torch.tensor([[0.3], [0.7]], dtype=torch.double))

    assert isinstance(posterior, GPyTorchPosterior)
    torch.testing.assert_close(
        posterior.mean,
        torch.tensor([[0.8], [1.2]], dtype=torch.double),
    )
    torch.testing.assert_close(
        posterior.variance,
        torch.full((2, 1), 1e-6, dtype=torch.double),
    )


def test_lightgbm_regression_ensemble_exposes_member_disagreement() -> None:
    train_X, train_Y = _regression_data()
    model = LightGBMEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeLGBMRegressor(bias=0.0),
            _FakeLGBMRegressor(bias=0.5),
            _FakeLGBMRegressor(bias=1.0),
        ],
        bootstrap=False,
    ).fit()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))

    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    assert torch.all(posterior.variance > 0.0)

    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())
    value = acqf(torch.tensor([[[0.85]]], dtype=torch.double))
    assert torch.isfinite(value).all()


def test_mixed_lightgbm_uses_native_categorical_width_and_compact_codes() -> None:
    train_X, train_Y = _mixed_regression_data()
    estimator = _FakeLGBMRegressor()
    model = LightGBMMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
    ).fit()

    assert estimator.fit_X is not None
    assert estimator.fit_X.shape == (6, 2)
    np.testing.assert_array_equal(estimator.fit_X[:, 1], np.array([0, 1, 2, 0, 1, 2]))
    assert estimator.fit_kwargs["categorical_feature"] == [1]
    assert model.categorical_values == {1: (10.0, 20.0, 30.0)}

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.5, 40.0]], dtype=torch.double))


def test_binary_lightgbm_single_and_ensemble_probability_uncertainty() -> None:
    train_X, train_Y = _binary_data()
    single = LightGBMBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        estimator=_FakeLGBMClassifier(num_classes=2),
    ).fit()
    single_posterior = single.posterior(torch.tensor([[0.3], [0.7]], dtype=torch.double))
    assert isinstance(single_posterior, ClassificationEnsemblePosterior)
    torch.testing.assert_close(
        single_posterior.epistemic_variance,
        torch.zeros_like(single_posterior.epistemic_variance),
    )

    ensemble = LightGBMBinaryEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeLGBMClassifier(num_classes=2, offset=-0.08),
            _FakeLGBMClassifier(num_classes=2, offset=0.0),
            _FakeLGBMClassifier(num_classes=2, offset=0.08),
        ],
        bootstrap=False,
    ).fit()
    posterior = ensemble.posterior(torch.tensor([[0.3], [0.7]], dtype=torch.double))
    assert torch.any(posterior.epistemic_variance > 0.0)

    value = qBinaryBALD(
        model=ensemble,
        num_samples=8,
        exclude_observed_duplicates=False,
    )(torch.tensor([[[0.35]]], dtype=torch.double))
    assert torch.isfinite(value).all()


def test_multiclass_lightgbm_ensemble_works_with_bald() -> None:
    train_X, train_Y = _multiclass_data()
    model = LightGBMMulticlassEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeLGBMClassifier(num_classes=3, offset=-0.1),
            _FakeLGBMClassifier(num_classes=3, offset=0.0),
            _FakeLGBMClassifier(num_classes=3, offset=0.1),
        ],
        bootstrap=False,
    ).fit()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        posterior.mean.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    assert torch.any(posterior.epistemic_variance > 0.0)

    value = qMulticlassBALD(
        model=model,
        num_samples=8,
        exclude_observed_duplicates=False,
    )(torch.tensor([[[0.33]]], dtype=torch.double))
    assert torch.isfinite(value).all()


def test_ordinal_lightgbm_single_and_ensemble_preserve_probability_contract() -> None:
    train_X, train_Y = _multiclass_data()
    single = LightGBMOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            _FakeLGBMClassifier(num_classes=2, offset=0.05),
            _FakeLGBMClassifier(num_classes=2, offset=-0.05),
        ],
    ).fit()
    single_probability = single.ordinal_probability_posterior(
        torch.tensor([[0.35]], dtype=torch.double)
    )
    assert isinstance(single_probability, OrdinalEnsemblePosterior)
    assert single_probability.values.shape == torch.Size([1, 1, 3])
    torch.testing.assert_close(
        single_probability.mean.sum(dim=-1),
        torch.ones(1, dtype=torch.double),
    )

    ensemble = LightGBMOrdinalEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        estimators=[
            [
                _FakeLGBMClassifier(num_classes=2, offset=-0.08),
                _FakeLGBMClassifier(num_classes=2, offset=-0.12),
            ],
            [
                _FakeLGBMClassifier(num_classes=2, offset=0.0),
                _FakeLGBMClassifier(num_classes=2, offset=-0.04),
            ],
            [
                _FakeLGBMClassifier(num_classes=2, offset=0.08),
                _FakeLGBMClassifier(num_classes=2, offset=0.04),
            ],
        ],
        bootstrap=False,
    ).fit()
    probability = ensemble.ordinal_probability_posterior(
        torch.tensor([[0.35]], dtype=torch.double)
    )
    assert probability.values.shape == torch.Size([3, 1, 3])
    assert torch.any(probability.epistemic_variance > 0.0)

    entropy = qOrdinalPredictiveEntropy(
        model=ensemble,
        exclude_observed_duplicates=False,
    )(torch.tensor([[[0.35]]], dtype=torch.double))
    assert torch.isfinite(entropy).all()


def test_mixed_lightgbm_classification_and_ordinal_use_native_categorical_features() -> None:
    train_X, train_Y = _mixed_classification_data()

    classifier = _FakeLGBMClassifier(num_classes=3)
    multiclass = LightGBMMixedMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=classifier,
    ).fit()
    assert multiclass.is_fitted
    assert classifier.fit_X is not None
    assert classifier.fit_X.shape == (9, 2)
    assert classifier.fit_kwargs["categorical_feature"] == [1]

    ordinal_estimators = [
        _FakeLGBMClassifier(num_classes=2),
        _FakeLGBMClassifier(num_classes=2, offset=-0.1),
    ]
    ordinal = LightGBMMixedOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimators=ordinal_estimators,
    ).fit()
    assert ordinal.is_fitted
    assert ordinal_estimators[0].fit_X is not None
    assert ordinal_estimators[0].fit_X.shape == (9, 2)
    assert ordinal_estimators[0].fit_kwargs["categorical_feature"] == [1]


@pytest.mark.parametrize(
    ("task_type", "model_type", "mixed", "expected_cls"),
    [
        ("regression", "lightgbm", False, LightGBMRegressorModel),
        ("regression", "lightgbm_ensemble", False, LightGBMEnsembleModel),
        ("regression", "lightgbm", True, LightGBMMixedRegressorModel),
        ("regression", "lightgbm_ensemble", True, LightGBMMixedEnsembleModel),
        ("binary", "lightgbm", False, LightGBMBinaryClassificationModel),
        ("binary", "lightgbm_ensemble", False, LightGBMBinaryEnsembleModel),
        ("binary", "lightgbm", True, LightGBMMixedBinaryClassificationModel),
        ("binary", "lightgbm_ensemble", True, LightGBMMixedBinaryEnsembleModel),
        ("multiclass", "lightgbm", False, LightGBMMulticlassClassificationModel),
        ("multiclass", "lightgbm_ensemble", False, LightGBMMulticlassEnsembleModel),
        ("multiclass", "lightgbm", True, LightGBMMixedMulticlassClassificationModel),
        ("multiclass", "lightgbm_ensemble", True, LightGBMMixedMulticlassEnsembleModel),
        ("ordinal", "lightgbm", False, LightGBMOrdinalModel),
        ("ordinal", "lightgbm_ensemble", False, LightGBMOrdinalEnsembleModel),
        ("ordinal", "lightgbm", True, LightGBMMixedOrdinalModel),
        ("ordinal", "lightgbm_ensemble", True, LightGBMMixedOrdinalEnsembleModel),
    ],
)
def test_lightgbm_registry_routes_all_task_and_input_variants(
    task_type,
    model_type,
    mixed,
    expected_cls,
) -> None:
    config = ModelConfig(
        task_type=task_type,
        model_type=model_type,
        cat_dims=[1] if mixed else [],
        outcome_transform=False,
    )
    assert resolve_model_cls(config) is expected_cls


def test_high_level_fit_path_fits_lightgbm_regression() -> None:
    train_X, train_Y = _regression_data()
    estimator = _FakeLGBMRegressor()
    config = ModelConfig(
        task_type="regression",
        model_type="lightgbm",
        outcome_transform=False,
        model_kwargs={"estimator": estimator},
    )
    bundle = build_model(train_X, train_Y, config)

    fitted = fit_model(bundle, FitConfig())

    assert isinstance(fitted.model, LightGBMRegressorModel)
    assert fitted.model.is_fitted
    assert estimator.fit_X is not None
