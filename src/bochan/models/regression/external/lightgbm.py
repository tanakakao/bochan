"""BoTorch-compatible LightGBM regression models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.ensemble import EnsembleModel
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch import Tensor
from torch.nn import Module

from bochan.models.external.common import (
    _ExternalRegressorMixin,
    _check_one_to_one_input_transform,
    _require_single_output,
    _validate_output_indices,
)
from bochan.models.external.native_categorical import _NativeCategoricalMixin


def _new_lightgbm_regressor(kwargs: Mapping[str, Any]) -> Any:
    """Create ``LGBMRegressor`` lazily so LightGBM remains optional."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LightGBM support requires the optional dependency. "
            "Install bochan with `pip install 'bochan[lightgbm]'` or install `lightgbm`."
        ) from exc
    return LGBMRegressor(**dict(kwargs))


def _lightgbm_callbacks(
    callbacks: Sequence[Callable[..., Any]] | None,
    *,
    early_stopping_rounds: int | None,
    has_validation: bool,
) -> list[Callable[..., Any]] | None:
    result = list(callbacks or [])
    if early_stopping_rounds is not None:
        if int(early_stopping_rounds) <= 0:
            raise ValueError("early_stopping_rounds must be positive.")
        if not has_validation:
            raise ValueError("early_stopping_rounds requires X_val and Y_val.")
        try:
            from lightgbm import early_stopping
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LightGBM early stopping requires the optional `lightgbm` dependency."
            ) from exc
        result.append(early_stopping(int(early_stopping_rounds), verbose=False))
    return result or None


class _LightGBMRegressorMixin(_ExternalRegressorMixin):
    """Shared LightGBM fitting helpers."""

    def _fit_kwargs(
        self,
        *,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
        val_sample_weight: Any | None,
        eval_metric: str | Callable[..., Any] | Sequence[Any] | None,
        callbacks: Sequence[Callable[..., Any]] | None,
        early_stopping_rounds: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if X_val is not None:
            kwargs["eval_set"] = [(X_val, Y_val)]
        if val_sample_weight is not None:
            if X_val is None:
                raise ValueError("val_sample_weight requires X_val and Y_val.")
            weights = np.asarray(val_sample_weight)
            if weights.shape[0] != X_val.shape[0]:
                raise ValueError(
                    "val_sample_weight must contain one value per validation observation."
                )
            kwargs["eval_sample_weight"] = [weights]
        if eval_metric is not None:
            kwargs["eval_metric"] = eval_metric
        resolved_callbacks = _lightgbm_callbacks(
            callbacks,
            early_stopping_rounds=early_stopping_rounds,
            has_validation=X_val is not None,
        )
        if resolved_callbacks is not None:
            kwargs["callbacks"] = resolved_callbacks
        cat_dims = getattr(self, "cat_dims", None)
        if cat_dims:
            kwargs["categorical_feature"] = list(cat_dims)
        return kwargs


class LightGBMRegressorModel(_LightGBMRegressorMixin, Model):
    """Single LightGBM regressor with a deterministic BoTorch posterior bridge.

    A single boosted tree model does not provide epistemic posterior variance.
    ``min_variance`` is therefore only a numerical variance floor for BoTorch
    compatibility. Use :class:`LightGBMEnsembleModel` when uncertainty is needed.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        min_variance: float = 1e-12,
        **lightgbm_kwargs: Any,
    ) -> None:
        super().__init__()
        train_Y = _require_single_output(train_X, train_Y, model_name="LightGBM")
        _check_one_to_one_input_transform(input_transform, model_name="LightGBM")
        if min_variance <= 0.0:
            raise ValueError("min_variance must be positive.")

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._num_outputs = 1
        self.min_variance = float(min_variance)
        self.estimator = (
            estimator if estimator is not None else _new_lightgbm_regressor(lightgbm_kwargs)
        )
        self._is_fitted = False

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

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
        """Fit the wrapped LightGBM regressor."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this LightGBM model instance.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        fit_kwargs = self._fit_kwargs(
            X_val=val_X,
            Y_val=val_Y,
            val_sample_weight=val_sample_weight,
            eval_metric=eval_metric,
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
        )
        if sample_weight is not None:
            weights = np.asarray(sample_weight)
            if weights.shape[0] != train_X.shape[0]:
                raise ValueError("sample_weight must contain one value per training observation.")
            fit_kwargs["sample_weight"] = weights
        self.estimator.fit(train_X, train_Y, **fit_kwargs)
        self._is_fitted = True
        self.eval()
        return self

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        del kwargs
        _validate_output_indices(output_indices, model_name="LightGBM")
        if torch.is_tensor(observation_noise) or observation_noise is not False:
            raise UnsupportedError(
                "LightGBMRegressorModel does not model observation noise."
            )
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        self.eval()
        transformed_X = self.transform_inputs(X)
        flat_X = transformed_X.reshape(-1, transformed_X.shape[-1])
        prediction = self.estimator.predict(self._estimator_features(flat_X))
        mean = torch.as_tensor(
            prediction,
            dtype=transformed_X.dtype,
            device=transformed_X.device,
        ).reshape(transformed_X.shape[:-1])
        variance = torch.full_like(mean, self.min_variance)
        posterior = GPyTorchPosterior(
            MultivariateNormal(
                mean=mean,
                covariance_matrix=DiagLinearOperator(variance),
            )
        )
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            posterior = outcome_transform.untransform_posterior(posterior, X=transformed_X)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class LightGBMEnsembleModel(_LightGBMRegressorMixin, EnsembleModel):
    """Bootstrap ensemble of full LightGBM regressors for epistemic uncertainty."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        ensemble_size: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimators: Sequence[Any] | None = None,
        estimator_factory: Callable[[], Any] | None = None,
        weights: Tensor | None = None,
        **lightgbm_kwargs: Any,
    ) -> None:
        super().__init__(weights=weights)
        train_Y = _require_single_output(train_X, train_Y, model_name="LightGBM ensemble")
        _check_one_to_one_input_transform(input_transform, model_name="LightGBM ensemble")

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._num_outputs = 1
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self._lightgbm_kwargs = dict(lightgbm_kwargs)

        if estimators is not None:
            self.estimators = list(estimators)
            if not self.estimators:
                raise ValueError("estimators must contain at least one model.")
            if ensemble_size is not None and int(ensemble_size) != len(self.estimators):
                raise ValueError("ensemble_size must match len(estimators) when both are provided.")
            self.ensemble_size = len(self.estimators)
        else:
            self.ensemble_size = 20 if ensemble_size is None else int(ensemble_size)
            if self.ensemble_size <= 0:
                raise ValueError("ensemble_size must be positive.")
            self.estimators = self._build_estimators(estimator_factory)

        if weights is not None and weights.numel() != self.ensemble_size:
            raise ValueError("weights must contain one value per ensemble member.")
        self._is_fitted = False

    def _build_estimators(self, estimator_factory: Callable[[], Any] | None) -> list[Any]:
        if estimator_factory is not None:
            return [estimator_factory() for _ in range(self.ensemble_size)]

        seed_rng = np.random.default_rng(self.random_state)
        estimators = []
        for _ in range(self.ensemble_size):
            kwargs = dict(self._lightgbm_kwargs)
            if "random_state" not in kwargs:
                kwargs["random_state"] = int(seed_rng.integers(0, np.iinfo(np.int32).max))
            estimators.append(_new_lightgbm_regressor(kwargs))
        return estimators

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

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
        """Fit full LightGBM models on bootstrap resamples of the training data."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this LightGBM ensemble instance.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        shared_kwargs = self._fit_kwargs(
            X_val=val_X,
            Y_val=val_Y,
            val_sample_weight=val_sample_weight,
            eval_metric=eval_metric,
            callbacks=callbacks,
            early_stopping_rounds=early_stopping_rounds,
        )
        base_weights = None if sample_weight is None else np.asarray(sample_weight)
        if base_weights is not None and base_weights.shape[0] != train_X.shape[0]:
            raise ValueError("sample_weight must contain one value per training observation.")

        rng = np.random.default_rng(self.random_state)
        n = train_X.shape[0]
        for estimator in self.estimators:
            indices = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            member_kwargs = dict(shared_kwargs)
            if base_weights is not None:
                member_kwargs["sample_weight"] = base_weights[indices]
            estimator.fit(train_X[indices], train_Y[indices], **member_kwargs)

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
        values = []
        for estimator in self.estimators:
            prediction = estimator.predict(estimator_X)
            values.append(
                torch.as_tensor(prediction, dtype=X.dtype, device=X.device).reshape(
                    *leading_shape,
                    1,
                )
            )
        return torch.stack(values, dim=-3)


class LightGBMMixedRegressorModel(_NativeCategoricalMixin, LightGBMRegressorModel):
    """LightGBM regressor using native categorical features for mixed inputs."""

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


class LightGBMMixedEnsembleModel(_NativeCategoricalMixin, LightGBMEnsembleModel):
    """Bootstrap LightGBM ensemble using native categorical mixed inputs."""

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
    "LightGBMEnsembleModel",
    "LightGBMMixedEnsembleModel",
    "LightGBMMixedRegressorModel",
    "LightGBMRegressorModel",
]
