"""Shared cumulative-ordinal wrapper for external probability estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.nn import Module

from bochan.models.ordinal.likelihood import OrdinalLogitLikelihood
from bochan.models.external.common import (
    _ExternalClassifierMixin,
    _check_one_to_one_input_transform,
    _require_classification_targets,
)
from bochan.posteriors.ordinal_ensemble import OrdinalEnsemblePosterior


def _require_ordinal_targets(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    model_name: str,
    num_classes: int | None = None,
) -> tuple[Tensor, int]:
    train_Y, inferred = _require_classification_targets(
        train_X,
        train_Y,
        model_name=model_name,
        num_classes=num_classes,
    )
    if inferred < 3:
        raise ValueError(f"{model_name} requires at least three ordinal classes.")
    return train_Y, inferred


def _threshold_targets(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    thresholds = np.arange(int(num_classes) - 1, dtype=np.int64)
    return (labels[:, None] > thresholds[None, :]).astype(np.int64, copy=False)


def _pava_nonincreasing_row(values: np.ndarray) -> np.ndarray:
    row = np.clip(np.asarray(values, dtype=float).reshape(-1), 0.0, 1.0)
    levels: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    for index, value in enumerate(row.tolist()):
        levels.append(float(value))
        weights.append(1)
        starts.append(index)
        ends.append(index)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / total_weight
            levels[-2] = merged
            weights[-2] = total_weight
            ends[-2] = ends[-1]
            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    out = np.empty_like(row)
    for level, start, end in zip(levels, starts, ends, strict=True):
        out[start : end + 1] = level
    return out


def _project_nonincreasing_probabilities(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 1:
        raise ValueError("Cumulative probability array must have at least one dimension.")
    flat = array.reshape(-1, array.shape[-1])
    projected = np.stack([_pava_nonincreasing_row(row) for row in flat], axis=0)
    return projected.reshape(array.shape)


def _class_probs_from_cumulative(
    cumulative: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    t = _project_nonincreasing_probabilities(cumulative)
    first = 1.0 - t[..., :1]
    middle = t[..., :-1] - t[..., 1:]
    last = t[..., -1:]
    probs = np.concatenate([first, middle, last], axis=-1)
    probs = np.clip(probs, 0.0, None)
    denom = probs.sum(axis=-1, keepdims=True)
    return probs / np.clip(denom, eps, None)


def _cumulative_from_class_probs(probabilities: Tensor) -> Tensor:
    cdf = probabilities.cumsum(dim=-1)
    return (1.0 - cdf[..., :-1]).clamp(0.0, 1.0)


class _ExternalCumulativeOrdinalMixin(_ExternalClassifierMixin):
    """Common BoTorch API for K-1 cumulative binary ordinal estimators."""

    num_classes: int
    _is_fitted: bool
    likelihood: OrdinalLogitLikelihood
    latent_jitter: float
    _configured_member_weights: Tensor | None

    def make_mll(self, **kwargs: Any) -> None:
        del kwargs
        return None

    @property
    def ordinal_likelihood(self) -> OrdinalLogitLikelihood:
        return self.likelihood

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _configure_ordinal_bridge(
        self,
        *,
        num_classes: int,
        latent_jitter: float,
        weights: Tensor | None,
    ) -> None:
        if latent_jitter <= 0.0:
            raise ValueError("latent_jitter must be positive.")
        self.num_classes = int(num_classes)
        self.latent_jitter = float(latent_jitter)
        self.likelihood = OrdinalLogitLikelihood(
            num_classes=self.num_classes,
            init_gap=1.0,
            fix_first_cutpoint=True,
        )
        self.likelihood.raw_gaps.requires_grad_(False)
        if weights is None:
            self.register_buffer("_configured_member_weights", None)
        else:
            weights = torch.as_tensor(weights, dtype=self.train_X.dtype).reshape(-1)
            if weights.numel() == 0 or (weights < 0).any() or float(weights.sum()) <= 0.0:
                raise ValueError("weights must contain non-negative values with positive sum.")
            self.register_buffer("_configured_member_weights", weights)

    def _member_class_probability_arrays(
        self,
        estimator_X: np.ndarray,
    ) -> list[np.ndarray]:
        raise NotImplementedError

    def _member_weights(self, num_members: int, *, ref: Tensor) -> Tensor:
        configured = self._configured_member_weights
        if configured is None:
            return torch.full(
                (num_members,),
                1.0 / float(num_members),
                device=ref.device,
                dtype=ref.dtype,
            )
        if configured.numel() != num_members:
            raise ValueError(
                f"weights must contain one value per ordinal member; "
                f"expected {num_members}, got {configured.numel()}."
            )
        weights = configured.to(device=ref.device, dtype=ref.dtype)
        return weights / weights.sum()

    def _probability_values_from_transformed(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = X.reshape(-1, X.shape[-1])
        estimator_X = self._estimator_features(flat_X)
        arrays = self._member_class_probability_arrays(estimator_X)
        if not arrays:
            raise RuntimeError(f"{type(self).__name__} produced no ordinal probability members.")

        leading_shape = X.shape[:-1]
        values = []
        for array in arrays:
            probabilities = np.asarray(array, dtype=float)
            expected_shape = (flat_X.shape[0], self.num_classes)
            if probabilities.shape != expected_shape:
                raise RuntimeError(
                    f"{type(self).__name__} member probabilities must have shape "
                    f"{expected_shape}, got {probabilities.shape}."
                )
            if not np.isfinite(probabilities).all():
                raise RuntimeError(
                    f"{type(self).__name__} member probabilities contain non-finite values."
                )
            probs = torch.as_tensor(
                probabilities,
                device=X.device,
                dtype=X.dtype,
            ).reshape(*leading_shape, self.num_classes)
            values.append(probs)
        return torch.stack(values, dim=-3)

    def ordinal_probability_posterior(
        self,
        X: Tensor,
        *,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> OrdinalEnsemblePosterior:
        del kwargs
        self.eval()
        transformed_X = self.transform_inputs(X)
        values = self._probability_values_from_transformed(transformed_X)
        weights = self._member_weights(values.shape[-3], ref=values)
        posterior = OrdinalEnsemblePosterior(values=values, weights=weights)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def probability_posterior(
        self,
        X: Tensor,
        **kwargs: Any,
    ) -> OrdinalEnsemblePosterior:
        return self.ordinal_probability_posterior(X, **kwargs)

    def _member_latent_scores(self, probability_values: Tensor) -> Tensor:
        cumulative = _cumulative_from_class_probs(probability_values)
        eps = max(float(self.latent_jitter), 1e-8)
        cumulative = cumulative.clamp(eps, 1.0 - eps)
        cuts = self.likelihood.cutpoints.to(
            device=probability_values.device,
            dtype=probability_values.dtype,
        )
        implied = torch.logit(cumulative) + cuts
        return implied.mean(dim=-1)

    def _latent_distribution_from_transformed(self, X: Tensor) -> MultivariateNormal:
        values = self._probability_values_from_transformed(X)
        member_latent = self._member_latent_scores(values)
        weights = self._member_weights(member_latent.shape[-2], ref=member_latent)

        weight_shape = [1] * member_latent.ndim
        weight_shape[-2] = int(weights.numel())
        expanded_weights = weights.view(*weight_shape)
        mean = (expanded_weights * member_latent).sum(dim=-2)
        centered = member_latent - mean.unsqueeze(-2)
        covariance = (
            expanded_weights.unsqueeze(-1)
            * centered.unsqueeze(-1)
            * centered.unsqueeze(-2)
        ).sum(dim=-3)

        q = int(mean.shape[-1])
        eye = torch.eye(q, device=mean.device, dtype=mean.dtype)
        covariance = covariance + eye * self.latent_jitter
        return MultivariateNormal(mean=mean, covariance_matrix=covariance)

    def forward(self, X: Tensor) -> MultivariateNormal:
        return self._latent_distribution_from_transformed(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        del kwargs
        if output_indices is not None and list(output_indices) != [0]:
            raise UnsupportedError(
                f"{type(self).__name__} exposes only ordinal output index 0."
            )
        if torch.is_tensor(observation_noise) or observation_noise is not False:
            raise UnsupportedError(
                f"{type(self).__name__} does not support observation_noise."
            )

        self.eval()
        transformed_X = self.transform_inputs(X)
        posterior = GPyTorchPosterior(
            self._latent_distribution_from_transformed(transformed_X)
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def latent_posterior(self, X: Tensor, **kwargs: Any) -> GPyTorchPosterior:
        return self.posterior(X, **kwargs)

    def class_probs(self, X: Tensor) -> Tensor:
        return self.ordinal_probability_posterior(X).mean

    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)

    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        return self.ordinal_probability_posterior(X).expected_utility(utilities)


def _initialize_external_ordinal_model(
    model: Model,
    *,
    train_X: Tensor,
    train_Y: Tensor,
    model_name: str,
    num_classes: int | None,
    input_transform: Module | None,
    latent_jitter: float,
    weights: Tensor | None,
) -> tuple[Tensor, int]:
    train_Y, inferred_classes = _require_ordinal_targets(
        train_X,
        train_Y,
        model_name=model_name,
        num_classes=num_classes,
    )
    _check_one_to_one_input_transform(input_transform, model_name=model_name)

    model.register_buffer("train_X", train_X.detach().clone())
    model.register_buffer("train_Y", train_Y.detach().clone())
    if input_transform is not None:
        model.input_transform = input_transform

    model._is_fitted = False
    model._configure_ordinal_bridge(
        num_classes=inferred_classes,
        latent_jitter=latent_jitter,
        weights=weights,
    )
    return train_Y, inferred_classes


__all__ = [
    "_ExternalCumulativeOrdinalMixin",
    "_class_probs_from_cumulative",
    "_cumulative_from_class_probs",
    "_initialize_external_ordinal_model",
    "_project_nonincreasing_probabilities",
    "_require_ordinal_targets",
    "_threshold_targets",
]
