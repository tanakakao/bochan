"""Cumulative Random Forest ordinal models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import numpy as np
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module

from bochan.models.classification.common.probability import _align_probability_columns
from bochan.models.external.common import _MixedCategoricalMixin

from .base import (
    _ExternalCumulativeOrdinalMixin,
    _class_probs_from_cumulative,
    _initialize_external_ordinal_model,
    _threshold_targets,
)


def _new_random_forest_classifier(kwargs: Mapping[str, Any]) -> Any:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Random Forest ordinal models require scikit-learn. "
            "Install bochan with `pip install 'bochan[tabular]'` or install `scikit-learn`."
        ) from exc
    return RandomForestClassifier(**dict(kwargs))


def _positive_probability(estimator: Any, X: np.ndarray, *, model_name: str) -> np.ndarray:
    probabilities = _align_probability_columns(
        estimator.predict_proba(X),
        member_classes=getattr(estimator, "classes_", None),
        num_classes=2,
        model_name=model_name,
    )
    return probabilities[:, 1]


class RandomForestOrdinalModel(_ExternalCumulativeOrdinalMixin, Model):
    """Ordinal model composed of K-1 cumulative Random Forest classifiers."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        input_transform: Module | None = None,
        estimators: Sequence[Any] | None = None,
        estimator_factory: Callable[[int], Any] | None = None,
        weights: Tensor | None = None,
        latent_jitter: float = 1e-6,
        **random_forest_kwargs: Any,
    ) -> None:
        super().__init__()
        _, inferred_classes = _initialize_external_ordinal_model(
            self,
            train_X=train_X,
            train_Y=train_Y,
            model_name="Random Forest ordinal model",
            num_classes=num_classes,
            input_transform=input_transform,
            latent_jitter=latent_jitter,
            weights=weights,
        )
        self._random_forest_kwargs = dict(random_forest_kwargs)
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

        kwargs = dict(self._random_forest_kwargs)
        shared_seed = kwargs.get("random_state")
        if shared_seed is None:
            shared_seed = int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        kwargs["random_state"] = shared_seed
        return [_new_random_forest_classifier(kwargs) for _ in range(num_thresholds)]

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        sample_weight: Any | None = None,
        **ignore: Any,
    ) -> Self:
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this Random Forest ordinal model.")

        train_X, labels = self._prepare_training_arrays()
        targets = _threshold_targets(labels, self.num_classes)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight)
            if weights.shape[0] != train_X.shape[0]:
                raise ValueError("sample_weight must contain one value per training observation.")
            fit_kwargs["sample_weight"] = weights

        tree_counts: list[int] = []
        for threshold, estimator in enumerate(self.estimators):
            estimator.fit(train_X, targets[:, threshold], **fit_kwargs)
            trees = getattr(estimator, "estimators_", None)
            if trees is None or len(trees) == 0:
                raise RuntimeError(
                    "Each fitted Random Forest threshold classifier must expose `estimators_`."
                )
            tree_counts.append(len(trees))

        if len(set(tree_counts)) != 1:
            raise RuntimeError(
                "All Random Forest threshold classifiers must expose the same number of trees."
            )
        configured = self._configured_member_weights
        if configured is not None and configured.numel() != tree_counts[0]:
            raise ValueError(
                f"weights must contain one value per paired tree; "
                f"expected {tree_counts[0]}, got {configured.numel()}."
            )

        self._is_fitted = True
        self.eval()
        return self

    def _member_class_probability_arrays(
        self,
        estimator_X: np.ndarray,
    ) -> list[np.ndarray]:
        tree_groups = [list(estimator.estimators_) for estimator in self.estimators]
        n_members = len(tree_groups[0])
        arrays: list[np.ndarray] = []
        for member_index in range(n_members):
            cumulative = np.stack(
                [
                    _positive_probability(
                        tree_groups[threshold][member_index],
                        estimator_X,
                        model_name=type(self).__name__,
                    )
                    for threshold in range(self.num_classes - 1)
                ],
                axis=-1,
            )
            arrays.append(_class_probs_from_cumulative(cumulative))
        return arrays


class RandomForestMixedOrdinalModel(
    _MixedCategoricalMixin,
    RandomForestOrdinalModel,
):
    """Cumulative Random Forest ordinal model for mixed inputs."""

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
    "RandomForestMixedOrdinalModel",
    "RandomForestOrdinalModel",
]
