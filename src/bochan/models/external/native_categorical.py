"""Native categorical preprocessing for external estimators.

Unlike the one-hot encoder used by generic sklearn-style models, this module keeps
categorical columns in-place and maps observed values to compact zero-based integer
codes. Estimators such as LightGBM can then consume the original feature layout via
their native categorical-feature support.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch import Tensor

from .common import _to_numpy


class _NativeCategoricalEncoder:
    """Map categorical columns to compact integer codes without changing width."""

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a 2D array with native categorical columns encoded as 0..K-1."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected a 2D mixed input with {self.input_dim} columns, got {X.shape}."
            )

        encoded = np.asarray(X, dtype=float).copy()
        for dim, categories in self.categories.items():
            column = X[:, dim : dim + 1]
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
            encoded[:, dim] = matches.argmax(axis=1)
        return encoded


class _NativeCategoricalMixin:
    """Keep categorical feature positions while using estimator-native handling."""

    native_categorical_encoder: _NativeCategoricalEncoder
    cat_dims: list[int]

    def _configure_native_categorical_encoder(
        self,
        train_X: Tensor,
        cat_dims: Sequence[int],
        categorical_atol: float,
    ) -> None:
        self.native_categorical_encoder = _NativeCategoricalEncoder(
            train_X=train_X,
            cat_dims=cat_dims,
            atol=categorical_atol,
        )
        self.cat_dims = list(self.native_categorical_encoder.cat_dims)

    @property
    def categorical_values(self) -> dict[int, tuple[float, ...]]:
        """Observed categorical values keyed by original input dimension."""
        return {
            dim: tuple(float(value) for value in values.tolist())
            for dim, values in self.native_categorical_encoder.categories.items()
        }

    @property
    def categorical_feature_indices(self) -> list[int]:
        """Original feature indices passed to native categorical estimators."""
        return list(self.cat_dims)

    def _estimator_features(self, X: Tensor) -> np.ndarray:
        return self.native_categorical_encoder.transform(_to_numpy(X))


__all__ = [
    "_NativeCategoricalEncoder",
    "_NativeCategoricalMixin",
]
