"""TabPFN foundation-model surrogates for single-output regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
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


def _new_tabpfn_regressor(
    kwargs: Mapping[str, Any],
    *,
    categorical_features_indices: Sequence[int] | None,
) -> Any:
    """Create ``TabPFNRegressor`` lazily so TabPFN remains optional."""
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TabPFN regression requires the optional dependency. "
            "Install bochan with `pip install 'bochan[tabpfn]'` or install `tabpfn`."
        ) from exc

    options = dict(kwargs)
    if "categorical_features_indices" in options:
        raise TypeError(
            "Do not pass categorical_features_indices through TabPFN kwargs. "
            "Use bochan's cat_dims on TabPFNMixedRegressorModel instead."
        )
    options["categorical_features_indices"] = (
        None
        if categorical_features_indices is None
        else [int(dim) for dim in categorical_features_indices]
    )
    return TabPFNRegressor(**options)


def _configure_tabpfn_categorical_indices(
    estimator: Any,
    categorical_features_indices: Sequence[int] | None,
) -> None:
    """Keep injected estimators aligned with bochan's mixed-input contract."""
    indices = (
        None
        if categorical_features_indices is None
        else [int(dim) for dim in categorical_features_indices]
    )
    if hasattr(estimator, "categorical_features_indices"):
        estimator.categorical_features_indices = indices


def _tabpfn_regression_variance(
    result: Mapping[str, Any],
    *,
    ref: Tensor,
    min_variance: float,
) -> Tensor:
    """Evaluate the official TabPFN bar-distribution variance from ``full`` output."""
    if "criterion" not in result or "logits" not in result:
        raise RuntimeError(
            "TabPFNRegressor.predict(output_type='full') must return `criterion` and `logits`."
        )

    criterion = result["criterion"]
    variance_fn = getattr(criterion, "variance", None)
    if not callable(variance_fn):
        raise RuntimeError("TabPFN regression criterion does not expose variance(logits).")

    logits = result["logits"]
    if not torch.is_tensor(logits):
        borders = getattr(criterion, "borders", None)
        if torch.is_tensor(borders):
            logits = torch.as_tensor(logits, device=borders.device, dtype=borders.dtype)
        else:
            logits = torch.as_tensor(logits)
    else:
        borders = getattr(criterion, "borders", None)
        if torch.is_tensor(borders):
            logits = logits.to(device=borders.device, dtype=borders.dtype)

    with torch.no_grad():
        variance = variance_fn(logits)
    variance = torch.as_tensor(variance, device=ref.device, dtype=ref.dtype)
    return variance.clamp_min(float(min_variance))


class TabPFNRegressorModel(_ExternalRegressorMixin, Model):
    """TabPFN regression surrogate with a BoTorch moment-matched posterior bridge.

    TabPFN's native predictive distribution is a bar distribution. The wrapped
    estimator remains the source of truth: :meth:`tabpfn_distribution` exposes
    its official ``output_type='full'`` result. For compatibility with generic
    BoTorch acquisitions, :meth:`posterior` maps the exact marginal mean and
    variance to an independent Gaussian posterior across candidate points.

    The Gaussian bridge is a marginal moment match, not a claim that TabPFN's
    predictive distribution itself is Gaussian or jointly independent.
    """

    posterior_family = "tabpfn_bar"

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        min_variance: float = 1e-12,
        _categorical_features_indices: Sequence[int] | None = None,
        **tabpfn_kwargs: Any,
    ) -> None:
        super().__init__()
        train_Y = _require_single_output(train_X, train_Y, model_name="TabPFN")
        _check_one_to_one_input_transform(input_transform, model_name="TabPFN")
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
        self._tabpfn_categorical_features_indices = (
            None
            if _categorical_features_indices is None
            else [int(dim) for dim in _categorical_features_indices]
        )
        self.estimator = (
            estimator
            if estimator is not None
            else _new_tabpfn_regressor(
                tabpfn_kwargs,
                categorical_features_indices=self._tabpfn_categorical_features_indices,
            )
        )
        _configure_tabpfn_categorical_indices(
            self.estimator,
            self._tabpfn_categorical_features_indices,
        )
        self._is_fitted = False

    def make_mll(self, **kwargs: Any) -> None:
        """TabPFN uses in-context fitting rather than a BoTorch marginal likelihood."""
        del kwargs
        return None

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, _fit_target: Any | None = None, **ignore: Any) -> Self:
        """Fit TabPFN's sklearn-style estimator on the stored context data."""
        del ignore
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this TabPFN model instance.")

        train_X, train_Y = self._prepare_training_arrays()
        self.estimator.fit(train_X, train_Y)
        self._is_fitted = True
        self.eval()
        return self

    def _full_distribution_from_transformed(self, X: Tensor) -> Mapping[str, Any]:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = X.reshape(-1, X.shape[-1])
        estimator_X = self._estimator_features(flat_X)
        result = self.estimator.predict(estimator_X, output_type="full")
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "TabPFNRegressor.predict(output_type='full') must return a mapping."
            )
        if "mean" not in result:
            raise RuntimeError(
                "TabPFNRegressor.predict(output_type='full') did not return `mean`."
            )
        return result

    def tabpfn_distribution(self, X: Tensor) -> Mapping[str, Any]:
        """Return TabPFN's official ``output_type='full'`` predictive result.

        When an ``outcome_transform`` is configured, this result is in the
        transformed target space used to fit the TabPFN estimator. Use
        :meth:`posterior` for bochan-scale predictive moments.
        """
        self.eval()
        transformed_X = self.transform_inputs(X)
        return self._full_distribution_from_transformed(transformed_X)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        del kwargs
        _validate_output_indices(output_indices, model_name="TabPFN")
        if torch.is_tensor(observation_noise) or observation_noise is not False:
            raise UnsupportedError("TabPFNRegressorModel does not expose observation_noise.")

        self.eval()
        transformed_X = self.transform_inputs(X)
        result = self._full_distribution_from_transformed(transformed_X)

        mean = torch.as_tensor(
            np.asarray(result["mean"]),
            device=transformed_X.device,
            dtype=transformed_X.dtype,
        ).reshape(transformed_X.shape[:-1])
        variance = _tabpfn_regression_variance(
            result,
            ref=transformed_X,
            min_variance=self.min_variance,
        ).reshape(transformed_X.shape[:-1])

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


class TabPFNMixedRegressorModel(
    _NativeCategoricalMixin,
    TabPFNRegressorModel,
):
    """TabPFN regression surrogate with estimator-native categorical features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        min_variance: float = 1e-12,
        categorical_atol: float = 1e-8,
        **tabpfn_kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            estimator=estimator,
            min_variance=min_variance,
            _categorical_features_indices=cat_dims,
            **tabpfn_kwargs,
        )
        self._configure_native_categorical_encoder(
            train_X,
            cat_dims,
            categorical_atol,
        )


__all__ = [
    "TabPFNMixedRegressorModel",
    "TabPFNRegressorModel",
]
