"""Epistemic probability utilities for binary classification.

The binary model's regular ``posterior`` represents the predictive Bernoulli
label distribution. Its variance is therefore usually ``p * (1 - p)``. This
module instead samples the latent GP posterior and maps each latent sample
through the model likelihood. Variation between those probability samples is
model (epistemic) uncertainty.
"""

from __future__ import annotations

import inspect
from math import prod
from typing import Any, Sequence

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor

from bochan.acquisition.binary._probability import (
    latent_samples_to_binary_probabilities,
)


def _call_posterior_accessor(
    accessor: Any,
    X: Tensor,
    *,
    output_indices: Sequence[int] | None = None,
) -> Posterior:
    """Call a latent posterior accessor without assuming its exact signature."""

    if output_indices is None:
        return accessor(X)
    try:
        signature = inspect.signature(accessor)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "output_indices" in signature.parameters:
        return accessor(X, output_indices=output_indices)
    return accessor(X)


def get_binary_latent_posterior(
    model: Any,
    X: Tensor,
    *,
    output_indices: Sequence[int] | None = None,
) -> Posterior:
    """Return the latent ``f`` posterior and never fall back to label variance."""

    accessor = getattr(model, "latent_posterior", None)
    if callable(accessor):
        return _call_posterior_accessor(
            accessor,
            X,
            output_indices=output_indices,
        )

    inner = getattr(model, "model", None)
    if inner is not None:
        accessor = getattr(inner, "latent_posterior", None)
        if callable(accessor):
            return _call_posterior_accessor(
                accessor,
                X,
                output_indices=output_indices,
            )

    raise AttributeError(
        f"{type(model).__name__} does not expose a latent posterior. "
        "Binary epistemic uncertainty requires latent_posterior(X)."
    )


def _probability_mean(
    model: Any,
    X: Tensor,
    *,
    output_indices: Sequence[int] | None = None,
    eps: float,
) -> Tensor | None:
    """Return the analytic predictive probability mean when available."""

    for name in ("probability_posterior", "posterior"):
        accessor = getattr(model, name, None)
        if not callable(accessor):
            continue
        try:
            posterior = _call_posterior_accessor(
                accessor,
                X,
                output_indices=output_indices,
            )
            mean = getattr(posterior, "mean", None)
            if torch.is_tensor(mean):
                return mean.clamp(eps, 1.0 - eps)
        except Exception:
            continue
    return None


def _find_contiguous_shape(
    actual: tuple[int, ...],
    target: tuple[int, ...],
) -> list[int] | None:
    """Return indices of the first contiguous ``target`` occurrence in ``actual``."""

    if not target:
        return []
    width = len(target)
    for start in range(len(actual) - width + 1):
        if actual[start : start + width] == target:
            return list(range(start, start + width))
    return None


def _align_probability_samples(
    posterior: "BinaryEpistemicProbabilityPosterior",
    probabilities: Tensor,
    sample_shape: torch.Size,
) -> Tensor:
    """Align latent-derived probability samples with the analytic mean layout."""

    probability_mean = posterior._probability_mean
    if not torch.is_tensor(probability_mean):
        return probabilities

    sample_shape = torch.Size(sample_shape)
    target_shape = torch.Size(probability_mean.shape)
    expected_shape = sample_shape + target_shape
    if probabilities.shape == expected_shape:
        return probabilities

    sample_ndim = len(sample_shape)
    if torch.Size(probabilities.shape[:sample_ndim]) != sample_shape:
        raise RuntimeError(
            "Binary epistemic probability samples do not preserve the requested "
            "sample shape. "
            f"sample_shape={tuple(sample_shape)}, "
            f" samples.shape={tuple(probabilities.shape)}, "
            f"probability_mean.shape={tuple(target_shape)}."
        )
    if len(target_shape) < 2:
        raise RuntimeError(
            "Binary epistemic probability mean must have q and output axes. "
            f"Got probability_mean.shape={tuple(target_shape)}."
        )

    values = probabilities
    body_shape = tuple(int(size) for size in values.shape[sample_ndim:])
    num_outputs = int(target_shape[-1])

    if not body_shape or body_shape[-1] != num_outputs:
        output_candidates = [
            index for index, size in enumerate(body_shape) if size == num_outputs
        ]
        if len(output_candidates) != 1:
            raise RuntimeError(
                "Could not identify the objective-output axis in binary epistemic "
                "probability samples. "
                f"Expected num_outputs={num_outputs}, "
                f"samples.shape={tuple(probabilities.shape)}, "
                f"probability_mean.shape={tuple(target_shape)}, "
                f"matching_axes={output_candidates}."
            )
        values = values.movedim(sample_ndim + output_candidates[0], -1)

    body_without_output = tuple(
        int(size) for size in values.shape[sample_ndim:-1]
    )
    expected_batch_shape = tuple(int(size) for size in target_shape[:-2])
    q_like = int(target_shape[-2])
    batch_indices = _find_contiguous_shape(
        body_without_output,
        expected_batch_shape,
    )
    if batch_indices is None:
        raise RuntimeError(
            "Could not identify the t-batch axes in binary epistemic probability "
            "samples. "
            f"Expected t_batch_shape={expected_batch_shape}, "
            f"samples.shape={tuple(probabilities.shape)}, "
            f"probability_mean.shape={tuple(target_shape)}."
        )

    remaining_indices = [
        index
        for index in range(len(body_without_output))
        if index not in batch_indices
    ]
    permutation = (
        list(range(sample_ndim))
        + [sample_ndim + index for index in batch_indices]
        + [sample_ndim + index for index in remaining_indices]
        + [values.ndim - 1]
    )
    values = values.permute(*permutation)

    point_start = sample_ndim + len(expected_batch_shape)
    point_and_extra_shape = tuple(int(size) for size in values.shape[point_start:-1])
    point_product = prod(point_and_extra_shape) if point_and_extra_shape else 1

    if point_product == q_like:
        return values.reshape(*sample_shape, *target_shape)

    q_candidates = [
        index for index, size in enumerate(point_and_extra_shape) if size == q_like
    ]
    if len(q_candidates) == 1:
        values = values.movedim(point_start + q_candidates[0], point_start)
        extra_size = prod(tuple(int(size) for size in values.shape[point_start + 1 : -1]))
        values = values.reshape(
            *sample_shape,
            *expected_batch_shape,
            q_like,
            extra_size,
            num_outputs,
        )
        return values.mean(dim=-2)

    if q_like > 0 and point_product % q_like == 0:
        extra_size = point_product // q_like
        values = values.reshape(
            *sample_shape,
            *expected_batch_shape,
            q_like,
            extra_size,
            num_outputs,
        )
        return values.mean(dim=-2)

    raise RuntimeError(
        "Could not align binary epistemic probability samples with the analytic "
        "probability posterior. "
        f"samples.shape={tuple(probabilities.shape)}, "
        f"probability_mean.shape={tuple(target_shape)}, "
        f"sample_shape={tuple(sample_shape)}."
    )


def binary_probability_samples(
    model: Any,
    X: Tensor,
    *,
    sample_shape: torch.Size | None = None,
    num_samples: int | None = None,
    output_indices: Sequence[int] | None = None,
    eps: float = 1e-6,
    base_samples: Tensor | None = None,
) -> Tensor:
    """Draw probability samples induced only by latent posterior uncertainty."""

    if sample_shape is None:
        if num_samples is None:
            num_samples = 128
        sample_shape = torch.Size([int(num_samples)])
    else:
        sample_shape = torch.Size(sample_shape)

    latent_posterior = get_binary_latent_posterior(
        model,
        X,
        output_indices=output_indices,
    )
    if base_samples is not None and callable(
        getattr(latent_posterior, "rsample_from_base_samples", None)
    ):
        latent_samples = latent_posterior.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )
    else:
        latent_samples = latent_posterior.rsample(sample_shape)

    return latent_samples_to_binary_probabilities(
        model,
        latent_samples,
        eps=eps,
        name="latent posterior samples",
        output_dim=-1,
    )


def binary_probability_moments(
    model: Any,
    X: Tensor,
    *,
    num_samples: int = 256,
    output_indices: Sequence[int] | None = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return probability mean and decomposed predictive variances."""

    probability_samples = binary_probability_samples(
        model,
        X,
        num_samples=num_samples,
        output_indices=output_indices,
        eps=eps,
    )
    sample_mean = probability_samples.mean(dim=0)
    probability_mean = _probability_mean(
        model,
        X,
        output_indices=output_indices,
        eps=eps,
    )
    if probability_mean is None or probability_mean.shape != sample_mean.shape:
        probability_mean = sample_mean

    epistemic_variance = probability_samples.var(dim=0, unbiased=False)
    aleatoric_variance = (
        probability_samples * (1.0 - probability_samples)
    ).mean(dim=0)
    total_label_variance = probability_mean * (1.0 - probability_mean)
    return (
        probability_mean,
        epistemic_variance.clamp_min(0.0),
        aleatoric_variance.clamp_min(0.0),
        total_label_variance.clamp_min(0.0),
    )


class BinaryEpistemicProbabilityPosterior(Posterior):
    """A probability posterior whose samples come from the latent GP posterior."""

    def __init__(
        self,
        *,
        model: Any,
        X: Tensor,
        latent_posterior: Posterior,
        probability_mean: Tensor | None = None,
        moment_samples: int = 256,
        eps: float = 1e-6,
    ) -> None:
        self.model = model
        self.X = X
        self.latent_posterior = latent_posterior
        self._probability_mean = probability_mean
        self.moment_samples = int(moment_samples)
        self.eps = float(eps)
        self._moment_cache: tuple[Tensor, Tensor] | None = None

    def _moments(self) -> tuple[Tensor, Tensor]:
        if self._moment_cache is None:
            samples = self.rsample(torch.Size([self.moment_samples]))
            sample_mean = samples.mean(dim=0)
            mean = self._probability_mean
            if mean is None or mean.shape != sample_mean.shape:
                mean = sample_mean
            variance = samples.var(dim=0, unbiased=False).clamp_min(0.0)
            self._moment_cache = (mean, variance)
        return self._moment_cache

    @property
    def mean(self) -> Tensor:
        return self._moments()[0]

    @property
    def variance(self) -> Tensor:
        return self._moments()[1]

    @property
    def device(self) -> torch.device:
        return self.latent_posterior.device

    @property
    def dtype(self) -> torch.dtype:
        return self.latent_posterior.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self.latent_posterior.event_shape

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size(
            getattr(self.latent_posterior, "batch_shape", torch.Size())
        )

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.latent_posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return self.latent_posterior.batch_range

    def _extended_shape(
        self,
        sample_shape: torch.Size | None = None,
    ) -> torch.Size:
        resolved = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        return self.latent_posterior._extended_shape(sample_shape=resolved)

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        resolved = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        latent_samples = self.latent_posterior.rsample(resolved)
        probabilities = latent_samples_to_binary_probabilities(
            self.model,
            latent_samples,
            eps=self.eps,
            name="latent posterior samples",
            output_dim=-1,
        )
        return _align_probability_samples(self, probabilities, resolved)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        resolved = torch.Size(sample_shape)
        latent_samples = self.latent_posterior.rsample_from_base_samples(
            sample_shape=resolved,
            base_samples=base_samples,
        )
        probabilities = latent_samples_to_binary_probabilities(
            self.model,
            latent_samples,
            eps=self.eps,
            name="latent posterior base samples",
            output_dim=-1,
        )
        return _align_probability_samples(self, probabilities, resolved)


class BinaryEpistemicProbabilityModel(Model):
    """BoTorch model adapter exposing epistemic probability posterior samples."""

    def __init__(
        self,
        model: Model,
        *,
        moment_samples: int = 256,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model = model
        self.moment_samples = int(moment_samples)
        self.eps = float(eps)

    @property
    def num_outputs(self) -> int:
        return int(getattr(self.model, "num_outputs", 1))

    @property
    def batch_shape(self) -> torch.Size:
        return getattr(self.model, "batch_shape", torch.Size())

    @property
    def train_inputs(self):
        return getattr(self.model, "train_inputs")

    def posterior(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        if observation_noise is not False:
            raise ValueError(
                "BinaryEpistemicProbabilityModel excludes observation noise. "
                "Use the original binary model posterior for Bernoulli label variance."
            )
        latent_posterior = get_binary_latent_posterior(
            self.model,
            X,
            output_indices=output_indices,
        )
        probability_mean = _probability_mean(
            self.model,
            X,
            output_indices=output_indices,
            eps=self.eps,
        )
        posterior = BinaryEpistemicProbabilityPosterior(
            model=self.model,
            X=X,
            latent_posterior=latent_posterior,
            probability_mean=probability_mean,
            moment_samples=self.moment_samples,
            eps=self.eps,
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        conditioned = self.model.condition_on_observations(X=X, Y=Y, **kwargs)
        return type(self)(
            conditioned,
            moment_samples=self.moment_samples,
            eps=self.eps,
        )


def as_epistemic_probability_model(
    model: Model,
    *,
    moment_samples: int = 256,
    eps: float = 1e-6,
) -> BinaryEpistemicProbabilityModel:
    """Wrap one binary model unless it is already epistemic-probability aware."""

    if isinstance(model, BinaryEpistemicProbabilityModel):
        return model
    return BinaryEpistemicProbabilityModel(
        model,
        moment_samples=moment_samples,
        eps=eps,
    )


__all__ = [
    "BinaryEpistemicProbabilityModel",
    "BinaryEpistemicProbabilityPosterior",
    "as_epistemic_probability_model",
    "binary_probability_moments",
    "binary_probability_samples",
    "get_binary_latent_posterior",
]
