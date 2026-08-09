"""Cumulative LightGBM ordinal models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import numpy as np
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module

from bochan.models.classification.common.lightgbm import (
    _lightgbm_classifier_fit_kwargs,
)
from bochan.models.classification.common.probability import (
    _align_probability_columns,
    _classification_bootstrap_indices,
)
from bochan.models.external.lightgbm import _resolve_lightgbm_estimator_kwargs
from bochan.models.external.native_categorical import _NativeCategoricalMixin

from .base import (
    _ExternalCumulativeOrdinalMixin,
    _class_probs_from_cumulative,
    _initialize_external_ordinal_model,
    _threshold_targets,
)


def _new_lightgbm_binary_classifier(kwargs: Mapping[str, Any]) -> Any:
    """Create a binary ``LGBMClassifier`` lazily."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LightGBM ordinal models require the optional dependency. "
            "Install bochan with `pip install 'bochan[lightgbm]'` or install `lightgbm`."
        ) from exc
    return LGBMClassifier(**dict(kwargs))


def _positive_probability(estimator: Any, X: np.ndarray, *, model_name: str) -> np.ndarray:
    probabilities = _align_probability_columns(
        estimator.predict_proba(X),
        member_classes=getattr(estimator, "classes_", None),
        num_classes=2,
        model_name=model_name,
    )
    return probabilities[:, 1]


class LightGBMOrdinalModel(_ExternalCumulativeOrdinalMixin, Model):
    """Ordinal model composed of K-1 cumulative LightGBM classifiers."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        input_transform: Module | None = None,
        estimators: Sequence[Any] | None = None,
        estimator_factory: Callable[[int], Any] | None = None,
        latent_jitter: float = 1e-6,
        **lightgbm_kwargs: Any,
    ) -> None:
        super().__init__()
        _, inferred_classes = _initialize_external_ordinal_model(
            self,
            train_X=train_X,
            train_Y=train_Y,
            model_name="LightGBM ordinal model",
            num_classes=num_classes,
            input_transform=input_transform,
            latent_jitter=latent_jitter,
            weights=None,
        )
        self._lightgbm_kwargs = _resolve_lightgbm_estimator_kwargs(
            lightgbm_kwargs,
            n_samples=int(train_X.shape[-2]),
        )
        self.estimators = self._build_estimators(
            num_thresholds=inferred_classes - 1,
            estimators=estimators,
            estimator_factory=estimator_factory,
        )

    def _build_estimators(
        self,
        *,
        num_thresholds: int,
        estimators: Sequence[Any] | None,
        estimator_factory: Callable[[int], Any] | None,
    ) -> list[Any]:
        if estimators is not None:
            result = list(estimators)
            if len(result) != num_thresholds:
                raise ValueError(
                    f"estimators must contain {num_thresholds} threshold classifiers, "
                    f"got {len(result)}."
                )
            return result
        if estimator_factory is not None:
            return [estimator_factory(threshold) for threshold in range(num_thresholds)]

        return [
            _new_lightgbm_binary_classifier(dict(self._lightgbm_kwargs))
            for _ in range(num_thresholds)
        ]

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
        """Fit one LightGBM binary classifier per cumulative threshold."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this LightGBM ordinal model.")

        train_X, labels = self._prepare_training_arrays()
        targets = _threshold_targets(labels, self.num_classes)
        val_X, val_labels = self._prepare_validation_arrays(X_val, Y_val)
        val_targets = (
            None if val_labels is None else _threshold_targets(val_labels, self.num_classes)
        )

        train_weights = None if sample_weight is None else np.asarray(sample_weight)
        if train_weights is not None and train_weights.shape[0] != train_X.shape[0]:
            raise ValueError("sample_weight must contain one value per training observation.")

        for threshold, estimator in enumerate(self.estimators):
            fit_kwargs = _lightgbm_classifier_fit_kwargs(
                self,
                X_val=val_X,
                Y_val=None if val_targets is None else val_targets[:, threshold],
                val_sample_weight=val_sample_weight,
                eval_metric=eval_metric,
                callbacks=callbacks,
                early_stopping_rounds=early_stopping_rounds,
            )
            if train_weights is not None:
                fit_kwargs["sample_weight"] = train_weights
            estimator.fit(train_X, targets[:, threshold], **fit_kwargs)

        self._is_fitted = True
        self.eval()
        return self

    def _member_class_probability_arrays(
        self,
        estimator_X: np.ndarray,
    ) -> list[np.ndarray]:
        cumulative = np.stack(
            [
                _positive_probability(
                    estimator,
                    estimator_X,
                    model_name=type(self).__name__,
                )
                for estimator in self.estimators
            ],
            axis=-1,
        )
        return [_class_probs_from_cumulative(cumulative)]


class LightGBMOrdinalEnsembleModel(_ExternalCumulativeOrdinalMixin, Model):
    """Bootstrap ensemble of cumulative LightGBM ordinal models."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        ensemble_size: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: Module | None = None,
        estimators: Sequence[Sequence[Any]] | None = None,
        estimator_factory: Callable[[int, int], Any] | None = None,
        weights: Tensor | None = None,
        latent_jitter: float = 1e-6,
        **lightgbm_kwargs: Any,
    ) -> None:
        super().__init__()
        _, inferred_classes = _initialize_external_ordinal_model(
            self,
            train_X=train_X,
            train_Y=train_Y,
            model_name="LightGBM ordinal ensemble",
            num_classes=num_classes,
            input_transform=input_transform,
            latent_jitter=latent_jitter,
            weights=weights,
        )
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self._lightgbm_kwargs = _resolve_lightgbm_estimator_kwargs(
            lightgbm_kwargs,
            n_samples=int(train_X.shape[-2]),
        )
        self.estimators = self._build_estimators(
            num_thresholds=inferred_classes - 1,
            ensemble_size=ensemble_size,
            estimators=estimators,
            estimator_factory=estimator_factory,
        )
        self.ensemble_size = len(self.estimators)
        configured = self._configured_member_weights
        if configured is not None and configured.numel() != self.ensemble_size:
            raise ValueError(
                f"weights must contain one value per LightGBM ordinal member; "
                f"expected {self.ensemble_size}, got {configured.numel()}."
            )

    def _build_estimators(
        self,
        *,
        num_thresholds: int,
        ensemble_size: int | None,
        estimators: Sequence[Sequence[Any]] | None,
        estimator_factory: Callable[[int, int], Any] | None,
    ) -> list[list[Any]]:
        if estimators is not None:
            groups = [list(group) for group in estimators]
            if not groups:
                raise ValueError("estimators must contain at least one ordinal member.")
            for group in groups:
                if len(group) != num_thresholds:
                    raise ValueError(
                        f"Each estimator group must contain {num_thresholds} thresholds."
                    )
            if ensemble_size is not None and int(ensemble_size) != len(groups):
                raise ValueError("ensemble_size must match len(estimators) when both are provided.")
            return groups

        size = 20 if ensemble_size is None else int(ensemble_size)
        if size <= 0:
            raise ValueError("ensemble_size must be positive.")
        seed_rng = np.random.default_rng(self.random_state)
        groups: list[list[Any]] = []
        for member in range(size):
            group: list[Any] = []
            for threshold in range(num_thresholds):
                if estimator_factory is not None:
                    estimator = estimator_factory(member, threshold)
                else:
                    kwargs = dict(self._lightgbm_kwargs)
                    if "random_state" not in kwargs:
                        kwargs["random_state"] = int(
                            seed_rng.integers(0, np.iinfo(np.int32).max)
                        )
                    estimator = _new_lightgbm_binary_classifier(kwargs)
                group.append(estimator)
            groups.append(group)
        return groups

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
        """Fit coherent cumulative LightGBM members on shared bootstrap samples."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this LightGBM ordinal ensemble.")

        train_X, labels = self._prepare_training_arrays()
        val_X, val_labels = self._prepare_validation_arrays(X_val, Y_val)
        val_targets = (
            None if val_labels is None else _threshold_targets(val_labels, self.num_classes)
        )
        train_weights = None if sample_weight is None else np.asarray(sample_weight)
        if train_weights is not None and train_weights.shape[0] != train_X.shape[0]:
            raise ValueError("sample_weight must contain one value per training observation.")

        rng = np.random.default_rng(self.random_state)
        for group in self.estimators:
            indices = _classification_bootstrap_indices(
                labels,
                rng=rng,
                bootstrap=self.bootstrap,
                num_classes=self.num_classes,
            )
            member_labels = labels[indices]
            member_targets = _threshold_targets(member_labels, self.num_classes)
            member_weights = None if train_weights is None else train_weights[indices]
            for threshold, estimator in enumerate(group):
                fit_kwargs = _lightgbm_classifier_fit_kwargs(
                    self,
                    X_val=val_X,
                    Y_val=None if val_targets is None else val_targets[:, threshold],
                    val_sample_weight=val_sample_weight,
                    eval_metric=eval_metric,
                    callbacks=callbacks,
                    early_stopping_rounds=early_stopping_rounds,
                )
                if member_weights is not None:
                    fit_kwargs["sample_weight"] = member_weights
                estimator.fit(
                    train_X[indices],
                    member_targets[:, threshold],
                    **fit_kwargs,
                )

        self._is_fitted = True
        self.eval()
        return self

    def _member_class_probability_arrays(
        self,
        estimator_X: np.ndarray,
    ) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for group in self.estimators:
            cumulative = np.stack(
                [
                    _positive_probability(
                        estimator,
                        estimator_X,
                        model_name=type(self).__name__,
                    )
                    for estimator in group
                ],
                axis=-1,
            )
            arrays.append(_class_probs_from_cumulative(cumulative))
        return arrays


class LightGBMMixedOrdinalModel(
    _NativeCategoricalMixin,
    LightGBMOrdinalModel,
):
    """Cumulative LightGBM ordinal model with native categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self._configure_native_categorical_encoder(train_X, cat_dims, categorical_atol)


class LightGBMMixedOrdinalEnsembleModel(
    _NativeCategoricalMixin,
    LightGBMOrdinalEnsembleModel,
):
    """Bootstrap cumulative LightGBM ordinal ensemble for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self._configure_native_categorical_encoder(train_X, cat_dims, categorical_atol)


__all__ = [
    "LightGBMMixedOrdinalEnsembleModel",
    "LightGBMMixedOrdinalModel",
    "LightGBMOrdinalEnsembleModel",
    "LightGBMOrdinalModel",
]
