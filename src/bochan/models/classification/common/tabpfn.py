"""Shared TabPFN classification implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module

from bochan.models.external.common import (
    _check_one_to_one_input_transform,
    _require_classification_targets,
)

from .probability import _ExternalProbabilityClassifierMixin, _align_probability_columns


def _new_tabpfn_classifier(
    kwargs: Mapping[str, Any],
    *,
    categorical_features_indices: Sequence[int] | None,
) -> Any:
    """Create ``TabPFNClassifier`` lazily so TabPFN remains optional."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TabPFN classification requires the optional dependency. "
            "Install bochan with `pip install 'bochan[tabpfn]'` or install `tabpfn`."
        ) from exc

    options = dict(kwargs)
    if "categorical_features_indices" in options:
        raise TypeError(
            "Do not pass categorical_features_indices through TabPFN kwargs. "
            "Use bochan's cat_dims on the mixed TabPFN model instead."
        )
    options["categorical_features_indices"] = (
        None
        if categorical_features_indices is None
        else [int(dim) for dim in categorical_features_indices]
    )
    return TabPFNClassifier(**options)


def _configure_tabpfn_categorical_indices(
    estimator: Any,
    categorical_features_indices: Sequence[int] | None,
) -> None:
    indices = (
        None
        if categorical_features_indices is None
        else [int(dim) for dim in categorical_features_indices]
    )
    if hasattr(estimator, "categorical_features_indices"):
        estimator.categorical_features_indices = indices


class _TabPFNClassificationModel(_ExternalProbabilityClassifierMixin, Model):
    """Shared binary/multiclass TabPFN probability model.

    The public ``predict_proba`` result is exposed as one finite posterior member.
    TabPFN's internal inference ensemble is already aggregated by the public API,
    so bochan does not reinterpret internal preprocessing members as independent
    epistemic posterior samples.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        binary: bool,
        num_classes: int | None = None,
        input_transform: Module | None = None,
        estimator: Any | None = None,
        _categorical_features_indices: Sequence[int] | None = None,
        **tabpfn_kwargs: Any,
    ) -> None:
        super().__init__()
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="TabPFN classifier",
            num_classes=requested_classes,
        )
        _check_one_to_one_input_transform(
            input_transform,
            model_name="TabPFN classifier",
        )

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.binary = bool(binary)
        self.num_classes = int(inferred_classes)
        self._configure_probability_acquisition_bridge()
        self._tabpfn_categorical_features_indices = (
            None
            if _categorical_features_indices is None
            else [int(dim) for dim in _categorical_features_indices]
        )
        self.estimator = (
            estimator
            if estimator is not None
            else _new_tabpfn_classifier(
                tabpfn_kwargs,
                categorical_features_indices=self._tabpfn_categorical_features_indices,
            )
        )
        _configure_tabpfn_categorical_indices(
            self.estimator,
            self._tabpfn_categorical_features_indices,
        )
        self._is_fitted = False

    def fit(self, _fit_target: Any | None = None, **ignore: Any) -> Self:
        """Fit the wrapped TabPFN classifier on stored context data."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError(
                "The optional fit target must be this TabPFN classification model."
            )

        train_X, train_Y = self._prepare_training_arrays()
        self.estimator.fit(train_X, train_Y)

        classes = np.asarray(getattr(self.estimator, "classes_", []), dtype=int)
        expected = np.arange(self.num_classes)
        if not np.array_equal(classes, expected):
            raise RuntimeError(
                f"Fitted TabPFN exposes classes {classes.tolist()}, "
                f"expected {expected.tolist()}."
            )

        self._is_fitted = True
        self.eval()
        return self

    def _member_probability_arrays(self, estimator_X: np.ndarray) -> list[np.ndarray]:
        probabilities = self.estimator.predict_proba(estimator_X)
        aligned = _align_probability_columns(
            probabilities,
            member_classes=getattr(self.estimator, "classes_", None),
            num_classes=self.num_classes,
            model_name=type(self).__name__,
        )
        return [aligned]


__all__ = ["_TabPFNClassificationModel"]
