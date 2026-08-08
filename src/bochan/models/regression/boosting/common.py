"""Shared helpers for sklearn-style BoTorch surrogate models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from torch import Tensor
from torch.nn import Module


def _require_single_output(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    model_name: str,
) -> Tensor:
    """Validate training tensors and return ``train_Y`` as ``n x 1``."""
    if train_X.ndim != 2:
        raise ValueError("train_X must be a 2D tensor with shape [n, d].")
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError(f"{model_name} currently supports single-output regression only.")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if not train_X.is_floating_point() or not train_Y.is_floating_point():
        raise TypeError("train_X and train_Y must be floating-point tensors.")
    return train_Y


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


def _check_one_to_one_input_transform(
    input_transform: Module | None,
    *,
    model_name: str,
) -> None:
    if input_transform is not None and bool(getattr(input_transform, "is_one_to_many", False)):
        raise UnsupportedError(
            f"{model_name} currently requires one-to-one input transforms; "
            "one-to-many perturbation transforms are not supported."
        )


class _ExternalRegressorMixin:
    """Shared tensor/NumPy boundary handling for external regression estimators."""

    train_X: Tensor
    train_Y: Tensor
    input_transform: Module
    outcome_transform: Module

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

    def _prepare_training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        self.train()
        transformed_X = self.transform_inputs(self.train_X)

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

        transformed_X = self.transform_inputs(X_tensor)
        transformed_Y = Y_tensor
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(transformed_Y, X=X_tensor)
        return self._estimator_features(transformed_X), _to_numpy(transformed_Y.squeeze(-1))


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
        """Observed categorical values keyed by original input dimension."""
        return {
            dim: tuple(float(value) for value in values.tolist())
            for dim, values in self.categorical_encoder.categories.items()
        }

    def _estimator_features(self, X: Tensor) -> np.ndarray:
        return self.categorical_encoder.transform(_to_numpy(X))
