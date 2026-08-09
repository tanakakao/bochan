"""Shared NGBoost classification implementation."""

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

from .probability import (
    _ExternalProbabilityClassifierMixin,
    _align_probability_columns,
    _classification_bootstrap_indices,
)


def _new_ngboost_classifier(kwargs: Mapping[str, Any], *, num_classes: int) -> Any:
    """Create ``NGBClassifier`` lazily and configure categorical output size."""
    try:
        from ngboost import NGBClassifier
        from ngboost.distns import k_categorical
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "NGBoost classification requires the optional dependency. "
            "Install bochan with `pip install 'bochan[ngboost]'` or install `ngboost`."
        ) from exc

    classifier_kwargs = dict(kwargs)
    if int(num_classes) > 2 and "Dist" not in classifier_kwargs:
        classifier_kwargs["Dist"] = k_categorical(int(num_classes))
    return NGBClassifier(**classifier_kwargs)


def _ngboost_fit_kwargs(
    *,
    X_val: np.ndarray | None,
    Y_val: np.ndarray | None,
    sample_weight: Any | None,
    val_sample_weight: Any | None,
    train_loss_monitor: Callable[..., Any] | None,
    val_loss_monitor: Callable[..., Any] | None,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if X_val is not None:
        kwargs["X_val"] = X_val
        kwargs["Y_val"] = Y_val
    if sample_weight is not None:
        kwargs["sample_weight"] = np.asarray(sample_weight)
    if val_sample_weight is not None:
        kwargs["val_sample_weight"] = np.asarray(val_sample_weight)
    if train_loss_monitor is not None:
        kwargs["train_loss_monitor"] = train_loss_monitor
    if val_loss_monitor is not None:
        kwargs["val_loss_monitor"] = val_loss_monitor
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return kwargs


class _NGBoostClassificationModel(_ExternalProbabilityClassifierMixin, EnsembleModel):
    """Single NGBClassifier exposed as a one-member probability ensemble."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        binary: bool,
        num_classes: int | None = None,
        input_transform: Module | None = None,
        estimator: Any | None = None,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(weights=None)
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="NGBoost classifier",
            num_classes=requested_classes,
        )
        _check_one_to_one_input_transform(
            input_transform,
            model_name="NGBoost classifier",
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
            else _new_ngboost_classifier(ngboost_kwargs, num_classes=self.num_classes)
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
        train_loss_monitor: Callable[..., Any] | None = None,
        val_loss_monitor: Callable[..., Any] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Self:
        """Fit the wrapped NGBClassifier on stored data."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this NGBoost classification model.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        fit_kwargs = _ngboost_fit_kwargs(
            X_val=val_X,
            Y_val=val_Y,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.estimator.fit(train_X, train_Y, **fit_kwargs)
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


class _NGBoostClassificationEnsembleModel(_ExternalProbabilityClassifierMixin, EnsembleModel):
    """Bootstrap ensemble of NGBClassifier models for epistemic probabilities."""

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
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(weights=weights)
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="NGBoost classifier ensemble",
            num_classes=requested_classes,
        )
        _check_one_to_one_input_transform(
            input_transform,
            model_name="NGBoost classifier ensemble",
        )

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.binary = bool(binary)
        self.num_classes = int(inferred_classes)
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self._ngboost_kwargs = dict(ngboost_kwargs)

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
            kwargs = dict(self._ngboost_kwargs)
            if "random_state" not in kwargs and self.random_state is not None:
                kwargs["random_state"] = int(seed_rng.integers(0, np.iinfo(np.int32).max))
            estimators.append(_new_ngboost_classifier(kwargs, num_classes=self.num_classes))
        return estimators

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        X_val: Any | None = None,
        Y_val: Any | None = None,
        sample_weight: Any | None = None,
        val_sample_weight: Any | None = None,
        train_loss_monitor: Callable[..., Any] | None = None,
        val_loss_monitor: Callable[..., Any] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Self:
        """Fit NGBoost members on class-preserving bootstrap samples."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this NGBoost classifier ensemble.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        shared_kwargs = _ngboost_fit_kwargs(
            X_val=val_X,
            Y_val=val_Y,
            sample_weight=None,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )

        base_sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        if base_sample_weight is not None and base_sample_weight.shape[0] != train_X.shape[0]:
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
            if base_sample_weight is not None:
                member_kwargs["sample_weight"] = base_sample_weight[indices]
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
    "_NGBoostClassificationEnsembleModel",
    "_NGBoostClassificationModel",
]
