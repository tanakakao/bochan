"""BoTorch-compatible NGBoost surrogate models."""

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


def _require_single_output(train_X: Tensor, train_Y: Tensor) -> Tensor:
    """Validate NGBoost training tensors and return ``train_Y`` as ``n x 1``."""
    if train_X.ndim != 2:
        raise ValueError("train_X must be a 2D tensor with shape [n, d].")
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("NGBoost models currently support single-output regression only.")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if not train_X.is_floating_point() or not train_Y.is_floating_point():
        raise TypeError("train_X and train_Y must be floating-point tensors.")
    return train_Y


def _to_numpy(value: Tensor) -> np.ndarray:
    """Detach a tensor and move it to a NumPy array for scikit-learn style models."""
    return value.detach().cpu().numpy()


def _new_ngboost_regressor(kwargs: Mapping[str, Any]) -> Any:
    """Create ``NGBRegressor`` lazily so NGBoost remains an optional dependency."""
    try:
        from ngboost import NGBRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "NGBoost support requires the optional dependency. "
            "Install bochan with `pip install 'bochan[ngboost]'` or install `ngboost`."
        ) from exc
    return NGBRegressor(**dict(kwargs))


def _validate_output_indices(output_indices: list[int] | None) -> None:
    if output_indices is None:
        return
    if list(output_indices) != [0]:
        raise UnsupportedError("NGBoost models currently expose only output index 0.")


def _check_one_to_one_input_transform(input_transform: Module | None) -> None:
    if input_transform is not None and bool(getattr(input_transform, "is_one_to_many", False)):
        raise UnsupportedError(
            "NGBoost models currently require one-to-one input transforms; "
            "one-to-many perturbation transforms are not supported."
        )


class _NGBoostModelMixin:
    """Shared data-transform and fitting helpers for external NGBoost estimators."""

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

    def _prepare_training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        self.train()
        input_transform = getattr(self, "input_transform", None)
        _check_one_to_one_input_transform(input_transform)
        transformed_X = self.transform_inputs(self.train_X)

        transformed_Y = self.train_Y
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(transformed_Y, X=self.train_X)

        return _to_numpy(transformed_X), _to_numpy(transformed_Y.squeeze(-1))

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

        input_transform = getattr(self, "input_transform", None)
        _check_one_to_one_input_transform(input_transform)
        transformed_X = self.transform_inputs(X_tensor)
        transformed_Y = Y_tensor
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(transformed_Y, X=X_tensor)
        return _to_numpy(transformed_X), _to_numpy(transformed_Y.squeeze(-1))

    @staticmethod
    def _fit_kwargs(
        *,
        X_val: np.ndarray | None,
        Y_val: np.ndarray | None,
        sample_weight: Any | None,
        val_sample_weight: Any | None,
        train_loss_monitor: Callable[..., Any] | None,
        val_loss_monitor: Callable[..., Any] | None,
        early_stopping_rounds: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if X_val is not None:
            kwargs["X_val"] = X_val
            kwargs["Y_val"] = Y_val
        if sample_weight is not None:
            kwargs["sample_weight"] = np.asarray(sample_weight)
        if val_sample_weight is not None:
            kwargs["val_sample_weight"] = np.asarray(val_sample_weight)
        if train_loss_monitor is not None:
            kwargs["train_loss_monitor"] = train_loss_monitor
        if val_loss_monitor is not None:
            kwargs["val_loss_monitor"] = val_loss_monitor
        if early_stopping_rounds is not None:
            kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        return kwargs


class NGBoostRegressorModel(_NGBoostModelMixin, Model):
    """Wrap ``NGBRegressor`` behind the standard BoTorch ``Model`` interface.

    The NGBoost predictive Normal distribution is represented as a diagonal
    ``MultivariateNormal`` across query points. This yields a standard
    ``GPyTorchPosterior`` and therefore works with BoTorch's normal MC samplers
    without a custom sampler registration.

    Notes:
        ``posterior(..., observation_noise=...)`` always represents NGBoost's
        predictive outcome distribution. A boolean ``observation_noise`` argument
        is accepted for BoTorch API compatibility but does not change the result.
        Tensor-valued observation noise is unsupported.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        min_scale: float = 1e-8,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__()
        train_Y = _require_single_output(train_X, train_Y)
        _check_one_to_one_input_transform(input_transform)
        if min_scale <= 0:
            raise ValueError("min_scale must be positive.")

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._num_outputs = 1
        self.min_scale = float(min_scale)
        self.estimator = estimator if estimator is not None else _new_ngboost_regressor(ngboost_kwargs)
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
        train_loss_monitor: Callable[..., Any] | None = None,
        val_loss_monitor: Callable[..., Any] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Self:
        """Fit the wrapped estimator using the model's stored training tensors."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this NGBoost model instance.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        fit_kwargs = self._fit_kwargs(
            X_val=val_X,
            Y_val=val_Y,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.estimator.fit(train_X, train_Y, **fit_kwargs)
        self._is_fitted = True
        self.eval()
        return self

    def _normal_parameters(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if not self._is_fitted:
            raise RuntimeError("NGBoostRegressorModel is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = X.reshape(-1, X.shape[-1])
        distribution = self.estimator.pred_dist(_to_numpy(flat_X))
        params = getattr(distribution, "params", None)
        if not isinstance(params, Mapping) or "loc" not in params or "scale" not in params:
            raise NotImplementedError(
                "NGBoostRegressorModel currently requires a Normal-compatible "
                "predictive distribution exposing `params['loc']` and `params['scale']`."
            )

        leading_shape = X.shape[:-1]
        loc = torch.as_tensor(params["loc"], dtype=X.dtype, device=X.device).reshape(leading_shape)
        scale = torch.as_tensor(params["scale"], dtype=X.dtype, device=X.device).reshape(leading_shape)
        scale = scale.clamp_min(self.min_scale)
        return loc, scale

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior:
        _validate_output_indices(output_indices)
        if isinstance(observation_noise, Tensor):
            raise UnsupportedError("Tensor-valued observation noise is not supported by NGBoostRegressorModel.")

        self.eval()
        transformed_X = self.transform_inputs(X)
        loc, scale = self._normal_parameters(transformed_X)
        posterior = GPyTorchPosterior(
            MultivariateNormal(
                mean=loc,
                covariance_matrix=DiagLinearOperator(scale.square()),
            )
        )

        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            posterior = outcome_transform.untransform_posterior(posterior, X=transformed_X)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class NGBoostEnsembleModel(_NGBoostModelMixin, EnsembleModel):
    """Bootstrap NGBoost ensemble exposed through BoTorch ``EnsembleModel``.

    Each ensemble member contributes its predictive mean. Consequently the
    ``EnsemblePosterior`` variance represents disagreement between bootstrap
    models (an epistemic uncertainty estimate), rather than the per-member
    NGBoost predictive scale.
    """

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
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(weights=weights)
        train_Y = _require_single_output(train_X, train_Y)
        _check_one_to_one_input_transform(input_transform)

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._num_outputs = 1
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self._ngboost_kwargs = dict(ngboost_kwargs)

        if estimators is not None:
            self.estimators = list(estimators)
            if not self.estimators:
                raise ValueError("estimators must contain at least one model.")
            if ensemble_size is not None and ensemble_size != len(self.estimators):
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
            kwargs = dict(self._ngboost_kwargs)
            if "random_state" not in kwargs and self.random_state is not None:
                kwargs["random_state"] = int(seed_rng.integers(0, np.iinfo(np.int32).max))
            estimators.append(_new_ngboost_regressor(kwargs))
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
        train_loss_monitor: Callable[..., Any] | None = None,
        val_loss_monitor: Callable[..., Any] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Self:
        """Fit each NGBoost member on a bootstrap resample of the stored data."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this NGBoost ensemble instance.")

        train_X, train_Y = self._prepare_training_arrays()
        self.eval()
        val_X, val_Y = self._prepare_validation_arrays(X_val, Y_val)
        shared_kwargs = self._fit_kwargs(
            X_val=val_X,
            Y_val=val_Y,
            sample_weight=None,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )

        rng = np.random.default_rng(self.random_state)
        n = train_X.shape[0]
        base_sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        if base_sample_weight is not None and base_sample_weight.shape[0] != n:
            raise ValueError("sample_weight must contain one value per training observation.")

        for estimator in self.estimators:
            indices = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            member_kwargs = dict(shared_kwargs)
            if base_sample_weight is not None:
                member_kwargs["sample_weight"] = base_sample_weight[indices]
            estimator.fit(train_X[indices], train_Y[indices], **member_kwargs)

        self._is_fitted = True
        self.eval()
        return self

    def forward(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("NGBoostEnsembleModel is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = _to_numpy(X.reshape(-1, X.shape[-1]))
        leading_shape = X.shape[:-1]
        member_values = []
        for estimator in self.estimators:
            prediction = estimator.predict(flat_X)
            value = torch.as_tensor(prediction, dtype=X.dtype, device=X.device).reshape(*leading_shape, 1)
            member_values.append(value)
        return torch.stack(member_values, dim=-3)
