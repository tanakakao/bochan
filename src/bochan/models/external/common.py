"""Shared Tensor/NumPy preprocessing for external BoTorch-compatible estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from torch import Tensor
from torch.nn import Module


def _require_external_inputs(train_X: Tensor) -> None:
    """Validate the common input contract for sklearn-style estimators."""
    if train_X.ndim != 2:
        raise ValueError("train_X must be a 2D tensor with shape [n, d].")
    if not train_X.is_floating_point():
        raise TypeError("train_X must be a floating-point tensor.")


def _require_single_output(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    model_name: str,
) -> Tensor:
    """Validate regression tensors and return ``train_Y`` as ``n x 1``."""
    _require_external_inputs(train_X)
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError(f"{model_name} currently supports single-output regression only.")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if not train_Y.is_floating_point():
        raise TypeError("train_Y must be a floating-point tensor for regression.")
    return train_Y


def _require_classification_targets(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    model_name: str,
    num_classes: int | None = None,
) -> tuple[Tensor, int]:
    """Validate integer class labels and return ``([n, 1] long, num_classes)``."""
    _require_external_inputs(train_X)
    y = torch.as_tensor(train_Y, device=train_X.device)
    if y.ndim == 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if y.ndim != 1:
        raise ValueError(f"{model_name} class targets must have shape [n] or [n, 1].")
    if y.shape[0] != train_X.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if y.is_floating_point():
        if not torch.isfinite(y).all():
            raise ValueError(f"{model_name} class targets must be finite.")
        rounded = y.round()
        if not torch.allclose(y, rounded):
            raise ValueError(f"{model_name} class targets must be integer-valued labels.")
        y = rounded
    y = y.long()
    if (y < 0).any():
        raise ValueError(f"{model_name} class targets must be non-negative integer labels.")

    inferred = int(y.max().item()) + 1 if y.numel() else 0
    k = inferred if num_classes is None else int(num_classes)
    if k < 2:
        raise ValueError(f"{model_name} requires at least two classes.")
    if inferred > k:
        raise ValueError(f"{model_name} targets contain a label outside num_classes={k}.")
    observed = torch.unique(y).cpu()
    expected = torch.arange(k)
    if not torch.equal(observed, expected):
        raise ValueError(
            f"{model_name} class labels must cover contiguous values 0..{k - 1}; "
            f"observed {observed.tolist()}."
        )
    return y.unsqueeze(-1), k


def _validate_classification_values(
    values: Any,
    *,
    num_classes: int,
    model_name: str,
) -> np.ndarray:
    """Validate validation-set labels without requiring every class to occur."""
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[-1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{model_name} validation targets must have shape [n] or [n, 1].")
    if not np.isfinite(array).all():
        raise ValueError(f"{model_name} validation targets must be finite.")
    rounded = np.rint(array)
    if not np.allclose(array, rounded):
        raise ValueError(f"{model_name} validation targets must be integer-valued labels.")
    labels = rounded.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= int(num_classes)):
        raise ValueError(f"{model_name} validation labels must be within [0, {num_classes - 1}].")
    return labels


def _to_numpy(value: Tensor) -> np.ndarray:
    """Detach a tensor and move it to a NumPy array for sklearn-style estimators."""
    return value.detach().cpu().numpy()


def _validate_output_indices(
    output_indices: list[int] | None,
    *,
    model_name: str,
) -> None:
    if output_indices is None:
        return
    if list(output_indices) != [0]:
        raise UnsupportedError(f"{model_name} currently exposes only output index 0.")


def _input_transform_children(input_transform: Module) -> list[Module]:
    """Return nested transforms for common BoTorch composite transforms."""
    values = getattr(input_transform, "values", None)
    if callable(values):
        try:
            children = list(values())
        except TypeError:
            children = []
        if children:
            return children

    transforms = getattr(input_transform, "transforms", None)
    if transforms is not None:
        try:
            return list(transforms)
        except TypeError:
            pass
    return []


def _one_to_many_applies_on_train(input_transform: Module) -> bool:
    """Return whether a one-to-many transform can expand fitting rows."""
    if not bool(getattr(input_transform, "is_one_to_many", False)):
        return False

    one_to_many_children = [
        child
        for child in _input_transform_children(input_transform)
        if bool(getattr(child, "is_one_to_many", False))
    ]
    if one_to_many_children:
        return any(_one_to_many_applies_on_train(child) for child in one_to_many_children)

    return bool(getattr(input_transform, "transform_on_train", True))


def _check_one_to_one_input_transform(
    input_transform: Module | None,
    *,
    model_name: str,
) -> None:
    """Validate one-to-many transforms at the external-estimator boundary.

    External estimator models share one invariant: fitting and validation targets
    have one row per nominal observation. Eval-only transforms such as BoTorch
    ``InputPerturbation`` are safe because ``_preprocess_fit_inputs`` deliberately
    skips them while posterior paths flatten and batch all transformed evaluation
    rows before crossing the Tensor-to-NumPy boundary. Training-time one-to-many
    expansion is rejected for every external estimator because it would break the
    X/Y row contract.
    """
    if input_transform is None or not bool(
        getattr(input_transform, "is_one_to_many", False)
    ):
        return

    if _one_to_many_applies_on_train(input_transform):
        raise UnsupportedError(
            f"{model_name} supports one-to-many input transforms only for evaluation; "
            "training-time row expansion is not supported."
        )


class _ExternalEstimatorMixin:
    """Shared input preprocessing at the Tensor-to-NumPy estimator boundary."""

    _uses_external_fit = True
    train_X: Tensor
    train_Y: Tensor
    input_transform: Module

    def _set_transformed_inputs(self) -> None:
        """Disable GP-specific training-input mutation from ``Model.eval``."""

    def _revert_to_original_inputs(self) -> None:
        """Disable GP-specific training-input mutation from ``Model.train``."""

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self.train_X,)

    @property
    def train_targets(self) -> Tensor:
        return self.train_Y.squeeze(-1)

    def _estimator_features(self, X: Tensor) -> np.ndarray:
        """Convert transformed model inputs to estimator features."""
        return _to_numpy(X)

    def _preprocess_fit_inputs(self, X: Tensor) -> Tensor:
        """Apply only training-safe transforms to estimator fitting inputs.

        ``InputTransform.preprocess_transform`` applies transforms configured for
        training and deliberately skips eval-only one-to-many perturbations. This
        keeps training and validation row counts aligned with their targets while
        retaining normalization and other fitting-time preprocessing.
        """
        input_transform = getattr(self, "input_transform", None)
        if input_transform is None:
            return X

        preprocess = getattr(input_transform, "preprocess_transform", None)
        if callable(preprocess):
            return preprocess(X)

        was_training = bool(self.training)
        self.train()
        try:
            return self.transform_inputs(X)
        finally:
            self.train(was_training)

    def _prepare_input_array(self, X: Tensor) -> np.ndarray:
        transformed_X = self.transform_inputs(X)
        return self._estimator_features(transformed_X)


class _ExternalRegressorMixin(_ExternalEstimatorMixin):
    """Shared target transforms for external regression estimators."""

    outcome_transform: Module

    def _prepare_training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        self.train()
        transformed_X = self._preprocess_fit_inputs(self.train_X)
        transformed_Y = self.train_Y
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(transformed_Y, X=self.train_X)
        return self._estimator_features(transformed_X), _to_numpy(transformed_Y.squeeze(-1))

    def _prepare_validation_arrays(
        self,
        X_val: Any | None,
        Y_val: Any | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if X_val is None and Y_val is None:
            return None, None
        if X_val is None or Y_val is None:
            raise ValueError("X_val and Y_val must be provided together.")

        X_tensor = torch.as_tensor(X_val, dtype=self.train_X.dtype, device=self.train_X.device)
        Y_tensor = torch.as_tensor(Y_val, dtype=self.train_Y.dtype, device=self.train_Y.device)
        if Y_tensor.ndim == 1:
            Y_tensor = Y_tensor.unsqueeze(-1)
        if X_tensor.ndim != 2 or Y_tensor.ndim != 2 or Y_tensor.shape[-1] != 1:
            raise ValueError("Validation data must have shapes [n, d] and [n, 1].")

        transformed_X = self._preprocess_fit_inputs(X_tensor)
        transformed_Y = Y_tensor
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(transformed_Y, X=X_tensor)
        return self._estimator_features(transformed_X), _to_numpy(transformed_Y.squeeze(-1))


class _ExternalClassifierMixin(_ExternalEstimatorMixin):
    """Shared label handling for external classification estimators."""

    num_classes: int

    def _prepare_training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        self.train()
        transformed_X = self._preprocess_fit_inputs(self.train_X)
        labels = _to_numpy(self.train_Y.squeeze(-1)).astype(np.int64, copy=False)
        return self._estimator_features(transformed_X), labels

    def _prepare_validation_arrays(
        self,
        X_val: Any | None,
        Y_val: Any | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if X_val is None and Y_val is None:
            return None, None
        if X_val is None or Y_val is None:
            raise ValueError("X_val and Y_val must be provided together.")

        X_tensor = torch.as_tensor(X_val, dtype=self.train_X.dtype, device=self.train_X.device)
        if X_tensor.ndim != 2:
            raise ValueError("Validation X must have shape [n, d].")
        transformed_X = self._preprocess_fit_inputs(X_tensor)
        labels = _validate_classification_values(
            Y_val,
            num_classes=self.num_classes,
            model_name=type(self).__name__,
        )
        if labels.shape[0] != X_tensor.shape[0]:
            raise ValueError("X_val and Y_val must contain the same number of observations.")
        return self._estimator_features(transformed_X), labels


class _MixedCategoricalEncoder:
    """One-hot encode categorical columns while preserving the public input space."""

    def __init__(
        self,
        train_X: Tensor,
        cat_dims: Sequence[int],
        *,
        atol: float = 1e-8,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must be a 2D tensor with shape [n, d].")
        if atol < 0:
            raise ValueError("categorical_atol must be non-negative.")

        dims = [int(dim) for dim in cat_dims]
        if not dims:
            raise ValueError("cat_dims must contain at least one categorical dimension.")
        if len(set(dims)) != len(dims):
            raise ValueError("cat_dims must not contain duplicate dimensions.")
        if any(dim < 0 or dim >= train_X.shape[-1] for dim in dims):
            raise ValueError(
                f"cat_dims must be within [0, {train_X.shape[-1] - 1}], got {dims}."
            )

        raw_X = _to_numpy(train_X)
        self.input_dim = int(train_X.shape[-1])
        self.cat_dims = tuple(sorted(dims))
        self.atol = float(atol)
        self.categories: dict[int, np.ndarray] = {}
        for dim in self.cat_dims:
            values = np.asarray(raw_X[:, dim])
            if not np.isfinite(values).all():
                raise ValueError(f"Categorical dimension {dim} contains non-finite values.")
            self.categories[dim] = np.unique(values)

    @property
    def encoded_dim(self) -> int:
        continuous = self.input_dim - len(self.cat_dims)
        categorical = sum(len(self.categories[dim]) for dim in self.cat_dims)
        return continuous + categorical

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a one-hot encoded 2D array for the wrapped estimator."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected a 2D mixed input with {self.input_dim} columns, got {X.shape}."
            )

        parts: list[np.ndarray] = []
        for dim in range(self.input_dim):
            column = X[:, dim : dim + 1]
            if dim not in self.categories:
                parts.append(column)
                continue

            categories = self.categories[dim]
            matches = np.isclose(
                column,
                categories.reshape(1, -1),
                rtol=0.0,
                atol=self.atol,
            )
            match_count = matches.sum(axis=1)
            if not np.all(match_count == 1):
                bad_values = np.unique(column[match_count != 1].reshape(-1))
                raise ValueError(
                    f"Categorical dimension {dim} contains values not observed during training: "
                    f"{bad_values.tolist()}. Known values are {categories.tolist()}."
                )
            parts.append(matches.astype(X.dtype, copy=False))

        return np.concatenate(parts, axis=1)


class _MixedCategoricalMixin:
    """Internal one-hot preprocessing shared by mixed external estimators."""

    categorical_encoder: _MixedCategoricalEncoder
    cat_dims: list[int]

    def _configure_categorical_encoder(
        self,
        train_X: Tensor,
        cat_dims: Sequence[int],
        categorical_atol: float,
    ) -> None:
        self.categorical_encoder = _MixedCategoricalEncoder(
            train_X=train_X,
            cat_dims=cat_dims,
            atol=categorical_atol,
        )
        self.cat_dims = list(self.categorical_encoder.cat_dims)

    @property
    def categorical_values(self) -> dict[int, tuple[float, ...]]:
        return {
            dim: tuple(float(value) for value in values.tolist())
            for dim, values in self.categorical_encoder.categories.items()
        }

    def _estimator_features(self, X: Tensor) -> np.ndarray:
        return self.categorical_encoder.transform(_to_numpy(X))


__all__ = [
    "_ExternalClassifierMixin",
    "_ExternalEstimatorMixin",
    "_ExternalRegressorMixin",
    "_MixedCategoricalEncoder",
    "_MixedCategoricalMixin",
    "_check_one_to_one_input_transform",
    "_require_classification_targets",
    "_require_external_inputs",
    "_require_single_output",
    "_to_numpy",
    "_validate_classification_values",
    "_validate_output_indices",
]
