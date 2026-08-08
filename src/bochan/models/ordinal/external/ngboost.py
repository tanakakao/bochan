"""Cumulative NGBoost ordinal models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import numpy as np
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module

from bochan.models.classification.external.base import (
    _align_probability_columns,
    _classification_bootstrap_indices,
)
from bochan.models.external.common import _MixedCategoricalMixin

from .base import (
    _ExternalCumulativeOrdinalMixin,
    _class_probs_from_cumulative,
    _initialize_external_ordinal_model,
    _threshold_targets,
)


def _new_ngboost_binary_classifier(kwargs: Mapping[str, Any]) -> Any:
    """Create a binary NGBClassifier lazily."""
    try:
        from ngboost import NGBClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "NGBoost ordinal models require the optional dependency. "
            "Install bochan with `pip install 'bochan[ngboost]'` or install `ngboost`."
        ) from exc
    return NGBClassifier(**dict(kwargs))


def _positive_probability(estimator: Any, X: np.ndarray, *, model_name: str) -> np.ndarray:
    probabilities = _align_probability_columns(
        estimator.predict_proba(X),
        member_classes=getattr(estimator, "classes_", None),
        num_classes=2,
        model_name=model_name,
    )
    return probabilities[:, 1]


def _ngboost_fit_kwargs(
    *,
    X_val: np.ndarray | None,
    Y_val: np.ndarray | None,
    sample_weight: np.ndarray | None,
    val_sample_weight: np.ndarray | None,
    train_loss_monitor: Callable[..., Any] | None,
    val_loss_monitor: Callable[..., Any] | None,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if X_val is not None:
        kwargs["X_val"] = X_val
        kwargs["Y_val"] = Y_val
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if val_sample_weight is not None:
        kwargs["val_sample_weight"] = val_sample_weight
    if train_loss_monitor is not None:
        kwargs["train_loss_monitor"] = train_loss_monitor
    if val_loss_monitor is not None:
        kwargs["val_loss_monitor"] = val_loss_monitor
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return kwargs


class NGBoostOrdinalModel(_ExternalCumulativeOrdinalMixin, Model):
    """Ordinal model composed of K-1 probabilistic NGBoost classifiers."""

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
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__()
        _, inferred_classes = _initialize_external_ordinal_model(
            self,
            train_X=train_X,
            train_Y=train_Y,
            model_name="NGBoost ordinal model",
            num_classes=num_classes,
            input_transform=input_transform,
            latent_jitter=latent_jitter,
            weights=None,
        )
        self._ngboost_kwargs = dict(ngboost_kwargs)
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

        seed_rng = np.random.default_rng(self._ngboost_kwargs.get("random_state"))
        result = []
        for _ in range(num_thresholds):
            kwargs = dict(self._ngboost_kwargs)
            if "random_state" not in kwargs:
                kwargs["random_state"] = int(
                    seed_rng.integers(0, np.iinfo(np.int32).max)
                )
            result.append(_new_ngboost_binary_classifier(kwargs))
        return result

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
        **ignore: Any,
    ) -> Self:
        """Fit one NGBClassifier for every cumulative ordinal threshold."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this NGBoost ordinal model.")

        train_X, labels = self._prepare_training_arrays()
        targets = _threshold_targets(labels, self.num_classes)
        val_X, val_labels = self._prepare_validation_arrays(X_val, Y_val)
        val_targets = (
            None
            if val_labels is None
            else _threshold_targets(val_labels, self.num_classes)
        )

        train_weights = None if sample_weight is None else np.asarray(sample_weight)
        if train_weights is not None and train_weights.shape[0] != train_X.shape[0]:
            raise ValueError("sample_weight must contain one value per training observation.")
        validation_weights = (
            None if val_sample_weight is None else np.asarray(val_sample_weight)
        )
        if (
            validation_weights is not None
            and val_X is not None
            and validation_weights.shape[0] != val_X.shape[0]
        ):
            raise ValueError(
                "val_sample_weight must contain one value per validation observation."
            )

        for threshold, estimator in enumerate(self.estimators):
            fit_kwargs = _ngboost_fit_kwargs(
                X_val=val_X,
                Y_val=None if val_targets is None else val_targets[:, threshold],
                sample_weight=train_weights,
                val_sample_weight=validation_weights,
                train_loss_monitor=train_loss_monitor,
                val_loss_monitor=val_loss_monitor,
                early_stopping_rounds=early_stopping_rounds,
            )
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


class NGBoostOrdinalEnsembleModel(_ExternalCumulativeOrdinalMixin, Model):
    """Bootstrap ensemble of cumulative NGBoost ordinal models."""

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
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__()
        _, inferred_classes = _initialize_external_ordinal_model(
            self,
            train_X=train_X,
            train_Y=train_Y,
            model_name="NGBoost ordinal ensemble",
            num_classes=num_classes,
            input_transform=input_transform,
            latent_jitter=latent_jitter,
            weights=weights,
        )
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self._ngboost_kwargs = dict(ngboost_kwargs)
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
                f"weights must contain one value per NGBoost ordinal member; "
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
                raise ValueError(
                    "ensemble_size must match len(estimators) when both are provided."
                )
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
                    kwargs = dict(self._ngboost_kwargs)
                    if "random_state" not in kwargs:
                        kwargs["random_state"] = int(
                            seed_rng.integers(0, np.iinfo(np.int32).max)
                        )
                    estimator = _new_ngboost_binary_classifier(kwargs)
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
        train_loss_monitor: Callable[..., Any] | None = None,
        val_loss_monitor: Callable[..., Any] | None = None,
        early_stopping_rounds: int | None = None,
        **ignore: Any,
    ) -> Self:
        """Fit class-preserving bootstrap cumulative NGBoost members."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError(
                "The optional fit target must be this NGBoost ordinal ensemble."
            )

        train_X, labels = self._prepare_training_arrays()
        val_X, val_labels = self._prepare_validation_arrays(X_val, Y_val)
        val_targets = (
            None
            if val_labels is None
            else _threshold_targets(val_labels, self.num_classes)
        )
        train_weights = None if sample_weight is None else np.asarray(sample_weight)
        if train_weights is not None and train_weights.shape[0] != train_X.shape[0]:
            raise ValueError("sample_weight must contain one value per training observation.")
        validation_weights = (
            None if val_sample_weight is None else np.asarray(val_sample_weight)
        )
        if (
            validation_weights is not None
            and val_X is not None
            and validation_weights.shape[0] != val_X.shape[0]
        ):
            raise ValueError(
                "val_sample_weight must contain one value per validation observation."
            )

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
            member_weights = (
                None if train_weights is None else train_weights[indices]
            )
            for threshold, estimator in enumerate(group):
                fit_kwargs = _ngboost_fit_kwargs(
                    X_val=val_X,
                    Y_val=None if val_targets is None else val_targets[:, threshold],
                    sample_weight=member_weights,
                    val_sample_weight=validation_weights,
                    train_loss_monitor=train_loss_monitor,
                    val_loss_monitor=val_loss_monitor,
                    early_stopping_rounds=early_stopping_rounds,
                )
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


class NGBoostMixedOrdinalModel(
    _MixedCategoricalMixin,
    NGBoostOrdinalModel,
):
    """Single cumulative NGBoost ordinal model for mixed inputs."""

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
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


class NGBoostMixedOrdinalEnsembleModel(
    _MixedCategoricalMixin,
    NGBoostOrdinalEnsembleModel,
):
    """Bootstrap cumulative NGBoost ordinal ensemble for mixed inputs."""

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
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


__all__ = [
    "NGBoostMixedOrdinalEnsembleModel",
    "NGBoostMixedOrdinalModel",
    "NGBoostOrdinalEnsembleModel",
    "NGBoostOrdinalModel",
]
