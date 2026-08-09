"""Shared LightGBM classification implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import numpy as np
from botorch.models.ensemble import EnsembleModel
from torch import Tensor
from torch.nn import Module

from bochan.models.external.common import (
    _check_one_to_one_input_transform,
    _require_classification_targets,
)
from bochan.models.external.lightgbm import _resolve_lightgbm_callbacks

from .probability import (
    _ExternalProbabilityClassifierMixin,
    _align_probability_columns,
    _classification_bootstrap_indices,
)


def _new_lightgbm_classifier(kwargs: Mapping[str, Any]) -> Any:
    """Create ``LGBMClassifier`` lazily so LightGBM remains optional."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LightGBM classification requires the optional dependency. "
            "Install bochan with `pip install 'bochan[lightgbm]'` or install `lightgbm`."
        ) from exc
    return LGBMClassifier(**dict(kwargs))


def _lightgbm_classifier_fit_kwargs(
    model: Any,
    *,
    X_val: np.ndarray | None,
    Y_val: np.ndarray | None,
    val_sample_weight: Any | None,
    eval_metric: str | Callable[..., Any] | Sequence[Any] | None,
    callbacks: Sequence[Callable[..., Any]] | None,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if X_val is not None:
        kwargs["eval_set"] = [(X_val, Y_val)]
    if val_sample_weight is not None:
        if X_val is None:
            raise ValueError("val_sample_weight requires X_val and Y_val.")
        weights = np.asarray(val_sample_weight)
        if weights.shape[0] != X_val.shape[0]:
            raise ValueError(
                "val_sample_weight must contain one value per validation observation."
            )
        kwargs["eval_sample_weight"] = [weights]
    if eval_metric is not None:
        kwargs["eval_metric"] = eval_metric
    resolved_callbacks = _resolve_lightgbm_callbacks(
        callbacks,
        early_stopping_rounds=early_stopping_rounds,
        has_validation=X_val is not None,
    )
    if resolved_callbacks is not None:
        kwargs["callbacks"] = resolved_callbacks
    cat_dims = getattr(model, "cat_dims", None)
    if cat_dims:
        kwargs["categorical_feature"] = list(cat_dims)
    return kwargs


class _LightGBMClassificationModel(_ExternalProbabilityClassifierMixin, EnsembleModel):
    """Single LightGBM classifier exposed as one probability-ensemble member."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        binary: bool,
        num_classes: int | None = None,
        input_transform: Module | None = None,
        estimator: Any | None = None,
        **lightgbm_kwargs: Any,
    ) -> None:
        super().__init__(weights=None)
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="LightGBM classifier",
            num_classes=requested_classes,
        )
        _check_one_to_one_input_transform(
            input_transform,
            model_name="LightGBM classifier",
        )

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.binary = bool(binary)
        self.num_classes = int(inferred_classes)
        self.estimator = (
            estimator
            if estimator is not None
            else _new_lightgbm_classifier(lightgbm_kwargs)
        )
        self._configure_probability_acquisition_bridge()
        self._is_fitted = False

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        X_val: Any | None = None,
        Y_val: Any | None = None,
        sample_weight: Any | None = None,
        val_sample_weight: Any | None = None,
        eval_metric: str | Callable[..., Any] | Sequence[Any] | None = None,
        callbacks: Sequence[Callable[..., Any]] | None = None,
        early_stopping_rounds: int | None = None,
        **ignore: Any,
    ) -> Self:
        """Fit the wrapped LightGBM classifier."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this LightGBM classification model.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        fit_kwargs = _lightgbm_classifier_fit_kwargs(
            self,
            X_val=val_X,
            Y_val=val_Y,
            val_sample_weight=val_sample_weight,
            eval_metric=eval_metric,
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
        )
        if sample_weight is not None:
            weights = np.asarray(sample_weight)
            if weights.shape[0] != train_X.shape[0]:
                raise ValueError("sample_weight must contain one value per training observation.")
            fit_kwargs["sample_weight"] = weights
        self.estimator.fit(train_X, train_Y, **fit_kwargs)

        classes = np.asarray(getattr(self.estimator, "classes_", []), dtype=int)
        expected = np.arange(self.num_classes)
        if not np.array_equal(classes, expected):
            raise RuntimeError(
                f"Fitted LightGBM classifier exposes classes {classes.tolist()}, "
                f"expected {expected.tolist()}."
            )
        self._is_fitted = True
        self.eval()
        return self

    def _member_probability_arrays(self, estimator_X: np.ndarray) -> list[np.ndarray]:
        return [
            _align_probability_columns(
                self.estimator.predict_proba(estimator_X),
                member_classes=getattr(self.estimator, "classes_", None),
                num_classes=self.num_classes,
                model_name=type(self).__name__,
            )
        ]


class _LightGBMClassificationEnsembleModel(
    _ExternalProbabilityClassifierMixin,
    EnsembleModel,
):
    """Bootstrap ensemble of full LightGBM classifiers."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        binary: bool,
        num_classes: int | None = None,
        ensemble_size: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: Module | None = None,
        estimators: Sequence[Any] | None = None,
        estimator_factory: Callable[[], Any] | None = None,
        weights: Tensor | None = None,
        **lightgbm_kwargs: Any,
    ) -> None:
        super().__init__(weights=weights)
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="LightGBM classifier ensemble",
            num_classes=requested_classes,
        )
        _check_one_to_one_input_transform(
            input_transform,
            model_name="LightGBM classifier ensemble",
        )

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.binary = bool(binary)
        self.num_classes = int(inferred_classes)
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self._lightgbm_kwargs = dict(lightgbm_kwargs)

        if estimators is not None:
            self.estimators = list(estimators)
            if not self.estimators:
                raise ValueError("estimators must contain at least one classifier.")
            if ensemble_size is not None and int(ensemble_size) != len(self.estimators):
                raise ValueError("ensemble_size must match len(estimators) when both are provided.")
            self.ensemble_size = len(self.estimators)
        else:
            self.ensemble_size = 20 if ensemble_size is None else int(ensemble_size)
            if self.ensemble_size <= 0:
                raise ValueError("ensemble_size must be positive.")
            self.estimators = self._build_estimators(estimator_factory)

        if weights is not None and weights.numel() != self.ensemble_size:
            raise ValueError("weights must contain one value per ensemble member.")
        self._configure_probability_acquisition_bridge()
        self._is_fitted = False

    def _build_estimators(self, estimator_factory: Callable[[], Any] | None) -> list[Any]:
        if estimator_factory is not None:
            return [estimator_factory() for _ in range(self.ensemble_size)]

        seed_rng = np.random.default_rng(self.random_state)
        estimators = []
        for _ in range(self.ensemble_size):
            kwargs = dict(self._lightgbm_kwargs)
            if "random_state" not in kwargs:
                kwargs["random_state"] = int(seed_rng.integers(0, np.iinfo(np.int32).max))
            estimators.append(_new_lightgbm_classifier(kwargs))
        return estimators

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        X_val: Any | None = None,
        Y_val: Any | None = None,
        sample_weight: Any | None = None,
        val_sample_weight: Any | None = None,
        eval_metric: str | Callable[..., Any] | Sequence[Any] | None = None,
        callbacks: Sequence[Callable[..., Any]] | None = None,
        early_stopping_rounds: int | None = None,
        **ignore: Any,
    ) -> Self:
        """Fit LightGBM classifier members on class-preserving bootstrap samples."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this LightGBM classifier ensemble.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        shared_kwargs = _lightgbm_classifier_fit_kwargs(
            self,
            X_val=val_X,
            Y_val=val_Y,
            val_sample_weight=val_sample_weight,
            eval_metric=eval_metric,
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
        )
        base_weights = None if sample_weight is None else np.asarray(sample_weight)
        if base_weights is not None and base_weights.shape[0] != train_X.shape[0]:
            raise ValueError("sample_weight must contain one value per training observation.")

        rng = np.random.default_rng(self.random_state)
        for estimator in self.estimators:
            indices = _classification_bootstrap_indices(
                train_Y,
                rng=rng,
                bootstrap=self.bootstrap,
                num_classes=self.num_classes,
            )
            member_kwargs = dict(shared_kwargs)
            if base_weights is not None:
                member_kwargs["sample_weight"] = base_weights[indices]
            estimator.fit(train_X[indices], train_Y[indices], **member_kwargs)

        self._is_fitted = True
        self.eval()
        return self

    def _member_probability_arrays(self, estimator_X: np.ndarray) -> list[np.ndarray]:
        return [
            _align_probability_columns(
                estimator.predict_proba(estimator_X),
                member_classes=getattr(estimator, "classes_", None),
                num_classes=self.num_classes,
                model_name=type(self).__name__,
            )
            for estimator in self.estimators
        ]


__all__ = [
    "_LightGBMClassificationEnsembleModel",
    "_LightGBMClassificationModel",
]
