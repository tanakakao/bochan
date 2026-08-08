"""BoTorch-compatible Random Forest regression models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch
from botorch.models.ensemble import EnsembleModel
from torch import Tensor
from torch.nn import Module

from bochan.models.external.common import (
    _ExternalRegressorMixin,
    _MixedCategoricalMixin,
    _check_one_to_one_input_transform,
    _require_single_output,
)


def _new_random_forest_regressor(kwargs: Mapping[str, Any]) -> Any:
    """Create ``RandomForestRegressor`` lazily so sklearn remains optional."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Random Forest support requires scikit-learn. "
            "Install bochan with `pip install 'bochan[tabular]'` or install `scikit-learn`."
        ) from exc
    return RandomForestRegressor(**dict(kwargs))


class RandomForestRegressorModel(_ExternalRegressorMixin, EnsembleModel):
    """Expose a sklearn Random Forest through BoTorch ``EnsembleModel``.

    Each fitted decision tree contributes one ensemble prediction. The resulting
    ``EnsemblePosterior`` variance is tree-to-tree disagreement rather than a
    Gaussian-process posterior variance.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        weights: Tensor | None = None,
        **random_forest_kwargs: Any,
    ) -> None:
        super().__init__(weights=weights)
        train_Y = _require_single_output(train_X, train_Y, model_name="Random Forest")
        _check_one_to_one_input_transform(input_transform, model_name="Random Forest")

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._num_outputs = 1
        self.estimator = (
            estimator
            if estimator is not None
            else _new_random_forest_regressor(random_forest_kwargs)
        )
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        sample_weight: Any | None = None,
    ) -> Self:
        """Fit the forest using the model's stored training tensors."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError(
                "The optional fit target must be this Random Forest model instance."
            )

        train_X, train_Y = self._prepare_training_arrays()
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            sample_weight_array = np.asarray(sample_weight)
            if sample_weight_array.shape[0] != train_X.shape[0]:
                raise ValueError(
                    "sample_weight must contain one value per training observation."
                )
            fit_kwargs["sample_weight"] = sample_weight_array

        self.estimator.fit(train_X, train_Y, **fit_kwargs)
        estimators = getattr(self.estimator, "estimators_", None)
        if estimators is None or len(estimators) == 0:
            raise RuntimeError(
                "The fitted Random Forest estimator does not expose any `estimators_`."
            )

        self._is_fitted = True
        self.eval()
        return self

    def forward(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = X.reshape(-1, X.shape[-1])
        estimator_X = self._estimator_features(flat_X)
        leading_shape = X.shape[:-1]
        member_values = []
        for tree in self.estimator.estimators_:
            prediction = tree.predict(estimator_X)
            value = torch.as_tensor(
                prediction,
                dtype=X.dtype,
                device=X.device,
            ).reshape(*leading_shape, 1)
            member_values.append(value)
        return torch.stack(member_values, dim=-3)


class RandomForestMixedRegressorModel(
    _MixedCategoricalMixin,
    RandomForestRegressorModel,
):
    """Random Forest surrogate for mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        weights: Tensor | None = None,
        categorical_atol: float = 1e-8,
        **random_forest_kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            estimator=estimator,
            weights=weights,
            **random_forest_kwargs,
        )
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


__all__ = [
    "RandomForestMixedRegressorModel",
    "RandomForestRegressorModel",
]
