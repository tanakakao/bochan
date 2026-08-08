"""Mixed-input NGBoost surrogate models with internal categorical encoding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from .ngboost import NGBoostEnsembleModel, NGBoostRegressorModel, _to_numpy


class _MixedCategoricalEncoder:
    """One-hot encode categorical columns while preserving the public input space.

    Category values are learned from the original training tensor. The encoder is
    intentionally internal to the NGBoost wrappers: BoTorch acquisition functions
    continue to operate on the original ``[..., q, d]`` mixed space.
    """

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
        """Return a one-hot encoded 2D array for the wrapped NGBoost estimator."""
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


class _MixedNGBoostMixin:
    """Shared internal one-hot preprocessing for mixed NGBoost wrappers."""

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

    def _encode_estimator_inputs(self, X: np.ndarray) -> np.ndarray:
        return self.categorical_encoder.transform(X)

    def _prepare_training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = super()._prepare_training_arrays()
        return self._encode_estimator_inputs(X), y

    def _prepare_validation_arrays(
        self,
        X_val: Any | None,
        Y_val: Any | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        X, y = super()._prepare_validation_arrays(X_val, Y_val)
        if X is None:
            return None, y
        return self._encode_estimator_inputs(X), y


class NGBoostMixedRegressorModel(_MixedNGBoostMixin, NGBoostRegressorModel):
    """BoTorch-compatible NGBoost regressor for continuous/categorical inputs.

    The public input remains in the original mixed space. Categorical columns are
    one-hot encoded only at the boundary to the wrapped NGBoost estimator, which
    keeps ``cat_dims`` and mixed candidate optimization aligned with BoTorch.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        min_scale: float = 1e-8,
        categorical_atol: float = 1e-8,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            estimator=estimator,
            min_scale=min_scale,
            **ngboost_kwargs,
        )
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)

    def _normal_parameters(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if not self._is_fitted:
            raise RuntimeError("NGBoostMixedRegressorModel is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = X.reshape(-1, X.shape[-1])
        encoded_X = self._encode_estimator_inputs(_to_numpy(flat_X))
        distribution = self.estimator.pred_dist(encoded_X)
        params = getattr(distribution, "params", None)
        if not isinstance(params, Mapping) or "loc" not in params or "scale" not in params:
            raise NotImplementedError(
                "NGBoostMixedRegressorModel currently requires a Normal-compatible "
                "predictive distribution exposing `params['loc']` and `params['scale']`."
            )

        leading_shape = X.shape[:-1]
        loc = torch.as_tensor(params["loc"], dtype=X.dtype, device=X.device).reshape(leading_shape)
        scale = torch.as_tensor(params["scale"], dtype=X.dtype, device=X.device).reshape(leading_shape)
        return loc, scale.clamp_min(self.min_scale)


class NGBoostMixedEnsembleModel(_MixedNGBoostMixin, NGBoostEnsembleModel):
    """Bootstrap NGBoost ensemble for mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        ensemble_size: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimators: Sequence[Any] | None = None,
        estimator_factory: Callable[[], Any] | None = None,
        weights: Tensor | None = None,
        categorical_atol: float = 1e-8,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            ensemble_size=ensemble_size,
            bootstrap=bootstrap,
            random_state=random_state,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            estimators=estimators,
            estimator_factory=estimator_factory,
            weights=weights,
            **ngboost_kwargs,
        )
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)

    def forward(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("NGBoostMixedEnsembleModel is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        encoded_X = self._encode_estimator_inputs(
            _to_numpy(X.reshape(-1, X.shape[-1]))
        )
        leading_shape = X.shape[:-1]
        member_values = []
        for estimator in self.estimators:
            prediction = estimator.predict(encoded_X)
            value = torch.as_tensor(
                prediction,
                dtype=X.dtype,
                device=X.device,
            ).reshape(*leading_shape, 1)
            member_values.append(value)
        return torch.stack(member_values, dim=-3)


__all__ = [
    "NGBoostMixedEnsembleModel",
    "NGBoostMixedRegressorModel",
]
