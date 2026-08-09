"""Shared Random Forest classification implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from botorch.models.ensemble import EnsembleModel
from torch import Tensor
from torch.nn import Module

from bochan.models.external.common import (
    _check_one_to_one_input_transform,
    _require_classification_targets,
)

from .probability import _ExternalProbabilityClassifierMixin, _align_probability_columns


def _new_random_forest_classifier(kwargs: Mapping[str, Any]) -> Any:
    """Create ``RandomForestClassifier`` lazily so sklearn remains optional."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Random Forest classification requires scikit-learn. "
            "Install bochan with `pip install 'bochan[tabular]'` or install `scikit-learn`."
        ) from exc
    return RandomForestClassifier(**dict(kwargs))


class _RandomForestClassificationModel(_ExternalProbabilityClassifierMixin, EnsembleModel):
    """Shared Random Forest classifier using individual trees as probability samples."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        binary: bool,
        num_classes: int | None = None,
        input_transform: Module | None = None,
        estimator: Any | None = None,
        weights: Tensor | None = None,
        **random_forest_kwargs: Any,
    ) -> None:
        super().__init__(weights=weights)
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="Random Forest classifier",
            num_classes=requested_classes,
        )
        _check_one_to_one_input_transform(
            input_transform,
            model_name="Random Forest classifier",
        )

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.binary = bool(binary)
        self.num_classes = int(inferred_classes)
        self._configure_probability_acquisition_bridge()
        self.estimator = (
            estimator
            if estimator is not None
            else _new_random_forest_classifier(random_forest_kwargs)
        )
        self._is_fitted = False

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        sample_weight: Any | None = None,
    ) -> Self:
        """Fit the wrapped forest on the stored classification data."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError(
                "The optional fit target must be this Random Forest classification model."
            )

        train_X, train_Y = self._prepare_training_arrays()
        fit_kwargs: dict[str, Any] = {}
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
                f"Fitted Random Forest exposes classes {classes.tolist()}, "
                f"expected {expected.tolist()}."
            )
        estimators = getattr(self.estimator, "estimators_", None)
        if estimators is None or len(estimators) == 0:
            raise RuntimeError(
                "The fitted Random Forest classifier does not expose any `estimators_`."
            )
        if self.ensemble_weights is not None and self.ensemble_weights.numel() != len(estimators):
            raise ValueError("weights must contain one value per fitted tree.")

        self._is_fitted = True
        self.eval()
        return self

    def _member_probability_arrays(self, estimator_X: np.ndarray) -> list[np.ndarray]:
        arrays = []
        for tree in self.estimator.estimators_:
            probabilities = tree.predict_proba(estimator_X)
            arrays.append(
                _align_probability_columns(
                    probabilities,
                    member_classes=getattr(tree, "classes_", None),
                    num_classes=self.num_classes,
                    model_name=type(self).__name__,
                )
            )
        return arrays


__all__ = ["_RandomForestClassificationModel"]
