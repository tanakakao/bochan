"""Shared BoTorch-facing base for external probability classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Tensor
from torch.nn import Module

from bochan.models.external.common import _ExternalClassifierMixin
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior


class _ProbabilityPassthroughLikelihood(Module):
    """Treat probability-space compatibility samples as Bernoulli probabilities."""

    def forward(self, values: Tensor):
        return torch.distributions.Bernoulli(probs=values.clamp(1e-9, 1.0 - 1e-9))


def _align_probability_columns(
    probabilities: Any,
    *,
    member_classes: Any | None,
    num_classes: int,
    model_name: str,
) -> np.ndarray:
    """Align estimator probability columns to class labels ``0..K-1``."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise RuntimeError(f"{model_name} predict_proba must return a 2D array, got {probs.shape}.")
    if not np.isfinite(probs).all():
        raise RuntimeError(f"{model_name} predict_proba returned non-finite values.")

    if member_classes is None:
        if probs.shape[1] != int(num_classes):
            raise RuntimeError(
                f"{model_name} predict_proba returned {probs.shape[1]} columns for "
                f"num_classes={num_classes}."
            )
        aligned = probs
    else:
        classes = np.asarray(member_classes).reshape(-1)
        if classes.shape[0] != probs.shape[1]:
            raise RuntimeError(
                f"{model_name} classes_ and predict_proba columns do not match: "
                f"{classes.shape[0]} vs {probs.shape[1]}."
            )
        aligned = np.zeros((probs.shape[0], int(num_classes)), dtype=probs.dtype)
        for column, class_value in enumerate(classes.tolist()):
            class_index = int(class_value)
            if class_index < 0 or class_index >= int(num_classes):
                raise RuntimeError(
                    f"{model_name} exposes unexpected class label {class_value}; "
                    f"expected 0..{num_classes - 1}."
                )
            aligned[:, class_index] = probs[:, column]

    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError(f"{model_name} predict_proba returned a zero-probability row.")
    aligned = np.clip(aligned / row_sums, 0.0, 1.0)
    return aligned


def _classification_bootstrap_indices(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap: bool,
    num_classes: int,
) -> np.ndarray:
    """Bootstrap rows while ensuring every known class remains represented."""
    n = int(labels.shape[0])
    if not bootstrap:
        return np.arange(n)
    indices = rng.integers(0, n, size=n)
    sampled = labels[indices]
    for class_index in range(int(num_classes)):
        if np.any(sampled == class_index):
            continue
        candidates = np.flatnonzero(labels == class_index)
        if candidates.size == 0:
            raise RuntimeError(f"Training labels do not contain class {class_index}.")
        replace_at = int(rng.integers(0, n))
        indices[replace_at] = int(candidates[int(rng.integers(0, candidates.size))])
        sampled[replace_at] = class_index
    return indices


class _ExternalProbabilityClassifierMixin(_ExternalClassifierMixin):
    """Expose estimator/member class probabilities through BoTorch posterior APIs."""

    num_classes: int
    binary: bool
    _is_fitted: bool

    def _configure_probability_acquisition_bridge(self) -> None:
        """Install the probability-space compatibility likelihood for binary AL."""
        if self.binary:
            self.likelihood = _ProbabilityPassthroughLikelihood()

    @property
    def num_outputs(self) -> int:
        return 1 if self.binary else int(self.num_classes)

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _member_probability_arrays(self, estimator_X: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def forward(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        flat_X = X.reshape(-1, X.shape[-1])
        estimator_X = self._estimator_features(flat_X)
        member_probs = self._member_probability_arrays(estimator_X)
        if not member_probs:
            raise RuntimeError(f"{type(self).__name__} produced no classifier members.")

        leading_shape = X.shape[:-1]
        values = []
        for probabilities in member_probs:
            probs = torch.as_tensor(probabilities, dtype=X.dtype, device=X.device).reshape(
                *leading_shape,
                self.num_classes,
            )
            if self.binary:
                probs = probs[..., 1:2]
            values.append(probs)
        return torch.stack(values, dim=-3)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> ClassificationEnsemblePosterior:
        if output_indices is not None:
            if not self.binary or list(output_indices) != [0]:
                raise UnsupportedError(
                    f"{type(self).__name__} does not support output_indices={output_indices}."
                )
        if torch.is_tensor(observation_noise):
            raise UnsupportedError(
                f"{type(self).__name__} does not support tensor-valued observation_noise."
            )

        self.eval()
        transformed_X = self.transform_inputs(X)
        values = self.forward(transformed_X)
        posterior = ClassificationEnsemblePosterior(
            values=values,
            weights=getattr(self, "weights", None),
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def probability_posterior(self, X: Tensor, **kwargs: Any) -> ClassificationEnsemblePosterior:
        """Return the standard probability posterior used by bochan prediction APIs."""
        return self.posterior(X, **kwargs)

    def epistemic_probability_posterior(
        self,
        X: Tensor,
        **kwargs: Any,
    ) -> ClassificationEnsemblePosterior:
        """Return member probability samples for epistemic active learning."""
        return self.posterior(X, **kwargs)

    def _multiclass_log_probability_posterior(self, X: Tensor) -> GPyTorchPosterior:
        """Gaussian approximation to ensemble log-probabilities for legacy MC samplers."""
        probability_posterior = self.posterior(X)
        values = probability_posterior.values.clamp_min(1e-9).log()
        flat_values = values.flatten(start_dim=-2)
        weights = probability_posterior.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * flat_values.ndim
        shape[-2] = int(weights.numel())
        weights = weights.view(*shape)
        flat_mean = (weights * flat_values).sum(dim=-2)
        centered = flat_values - flat_mean.unsqueeze(-2)
        covariance = (
            weights.unsqueeze(-1)
            * centered.unsqueeze(-1)
            * centered.unsqueeze(-2)
        ).sum(dim=-3)
        event_size = int(flat_values.shape[-1])
        jitter = torch.eye(event_size, dtype=values.dtype, device=values.device) * 1e-8
        covariance = covariance + jitter
        mean = flat_mean.reshape(*values.shape[:-3], values.shape[-2], values.shape[-1])
        return GPyTorchPosterior(
            MultitaskMultivariateNormal(
                mean=mean,
                covariance_matrix=covariance,
                interleaved=True,
            )
        )

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ):
        """Compatibility posterior used by existing classification acquisitions.

        Binary models expose finite member probabilities directly and install a
        passthrough Bernoulli likelihood. Multiclass models expose a Gaussian
        approximation in log-probability space; the existing multiclass
        acquisition softmax then maps those samples back to the simplex.
        """
        if output_indices is not None:
            if not self.binary or list(output_indices) != [0]:
                raise UnsupportedError(
                    f"{type(self).__name__} does not support output_indices={output_indices}."
                )
        if self.binary:
            if getattr(self, "likelihood", None) is None:
                self._configure_probability_acquisition_bridge()
            posterior = self.posterior(X, output_indices=output_indices)
        else:
            posterior = self._multiclass_log_probability_posterior(X)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def class_probs(self, X: Tensor) -> Tensor:
        mean = self.posterior(X).mean
        if self.binary:
            p1 = mean[..., 0]
            return torch.stack([1.0 - p1, p1], dim=-1)
        return mean

    def class_probs_list(self, X: Tensor, output_indices: Any | None = None) -> list[Tensor]:
        del output_indices
        return [self.class_probs(X)]

    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)


__all__ = [
    "_ExternalProbabilityClassifierMixin",
    "_ProbabilityPassthroughLikelihood",
    "_align_probability_columns",
    "_classification_bootstrap_indices",
]
