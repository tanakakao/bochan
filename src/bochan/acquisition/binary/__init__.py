"""Binary-classification acquisition utilities.

This package registers the custom binary probability posteriors with BoTorch's
sampler dispatcher. The posteriors expose the normal base-sample interface
required by BoTorch, so standard normal MC samplers can be reused by qEHVI,
qNEHVI, and other Monte Carlo acquisition functions.
"""

from __future__ import annotations

from math import prod

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init
from bochan.models.multioutput.binary import (
    MultiOutputBernoulliPosterior,
)

from .epistemic import BinaryEpistemicProbabilityPosterior

# The high-level registry may resolve the generic BoTorch qNEHVI class directly.
# Patch it here as well as the bochan-specific wrappers so models such as the
# Kronecker binary classifier can disable the insupported cached-Cholesky path.
patch_nehvi_cache_root_init(qNoisyExpectedHypervolumeImprovement)


def _normalize_binary_epistemic_extended_shape(
    self: BinaryEpistemicProbabilityPosterior,
    sample_shape: torch.Size | None = None,
) -> torch.Size:
    """Delegate shape calculation using BoTorch's empty sample-shape default.

    BoTorch may call ``posterior._extended_shape()`` without an explicit
    ``sample_shape``. ``GPyTorchPosterior`` expects a ``torch.Size`` rather than
    ``None``, so normalize the optional argument before delegation.
    """
    resolved_sample_shape = (
        torch.Size() if sample_shape is None else torch.Size(sample_shape)
    )
    return self.latent_posterior._extended_shape(
        sample_shape=resolved_sample_shape,
    )


def _binary_epistemic_batch_shape(
    self: BinaryEpistemicProbabilityPosterior,
) -> torch.Size:
    """Return the latent posterior batch shape for BoTorch normal samplers."""
    return torch.Size(getattr(self.latent_posterior, "batch_shape", torch.Size()))


def _multioutput_bernoulli_batch_shape(
    self: MultiOutputBernoulliPosterior,
) -> torch.Size:
    """Return dimensions preceding the posterior's ``q`` and output axes."""
    return torch.Size(self.mean.shape[:-2])


def _find_contiguous_shape(
    actual: tuple[int, ...],
    target: tuple[int, ...],
) -> list[int] | None:
    """Return indices of the first contiguous ``target`` occurrence in ``actual``."""
    if len(target) == 0:
        return []
    width = len(target)
    for start in range(len(actual) - width + 1):
        if actual[start : start + width] == target:
            return list(range(start, start + width))
    return None


def _align_epistemic_probability_samples(
    self: BinaryEpistemicProbabilityPosterior,
    probabilities: Tensor,
    sample_shape: torch.Size,
) -> Tensor:
    """Align latent-derived probabilities with the analytic probability mean.

    Projected PCA / REMBO classifiers may expose latent posterior model-batch
    dimensions that are not present in their probability posterior. qEHVI needs
    samples with the canonical shape
    ``sample_shape x t_batch_shape x q_like x num_outputs``. The analytic
    probability mean is the authoritative reference for that layout.

    Extra latent model-batch dimensions are averaged only after the expected
    output and point axes have been identified. Split point axes whose product is
    ``q_like`` are reshaped without reduction.
    """
    probability_mean = getattr(self, "_probability_mean", None)
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
            f"samples.shape={tuple(probabilities.shape)}, "
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

    # The likelihood conversion must preserve an explicit output axis. Move the
    # unique matching axis to the end when a projected latent model placed it in
    # a model-batch position.
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

    all_body_indices = list(range(len(body_without_output)))
    remaining_indices = [
        index for index in all_body_indices if index not in batch_indices
    ]
    permutation = (
        list(range(sample_ndim))
        + [sample_ndim + index for index in batch_indices]
        + [sample_ndim + index for index in remaining_indices]
        + [values.ndim - 1]
    )
    values = values.permute(*permutation)

    point_and_extra_shape = tuple(
        int(size)
        for size in values.shape[
            sample_ndim + len(expected_batch_shape) : -1
        ]
    )
    point_product = prod(point_and_extra_shape) if point_and_extra_shape else 1

    # Multiple adjacent point axes such as q x n_w can be flattened exactly.
    if point_product == q_like:
        return values.reshape(*sample_shape, *target_shape)

    # Otherwise identify the explicit q_like axis and average only the remaining
    # projected / model-batch axes.
    q_candidates = [
        index for index, size in enumerate(point_and_extra_shape) if size == q_like
    ]
    if len(q_candidates) == 1:
        q_index = q_candidates[0]
        point_start = sample_ndim + len(expected_batch_shape)
        values = values.movedim(point_start + q_index, point_start)
        extra_size = prod(
            tuple(int(size) for size in values.shape[point_start + 1 : -1])
        )
        values = values.reshape(
            *sample_shape,
            *expected_batch_shape,
            q_like,
            extra_size,
            num_outputs,
        )
        return values.mean(dim=-2)

    # A flattened latent layout can still be normalized when its point/model
    # dimensions contain an integer number of q_like groups.
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


_original_binary_epistemic_rsample = BinaryEpistemicProbabilityPosterior.rsample
_original_binary_epistemic_rsample_from_base_samples = (
    BinaryEpistemicProbabilityPosterior.rsample_from_base_samples
)


def _aligned_binary_epistemic_rsample(
    self: BinaryEpistemicProbabilityPosterior,
    sample_shape: torch.Size | None = None,
) -> Tensor:
    """Draw and align latent-derived probability samples."""
    resolved_sample_shape = (
        torch.Size() if sample_shape is None else torch.Size(sample_shape)
    )
    probabilities = _original_binary_epistemic_rsample(
        self,
        sample_shape=resolved_sample_shape,
    )
    return _align_epistemic_probability_samples(
        self,
        probabilities,
        resolved_sample_shape,
    )


def _aligned_binary_epistemic_rsample_from_base_samples(
    self: BinaryEpistemicProbabilityPosterior,
    sample_shape: torch.Size,
    base_samples: Tensor,
) -> Tensor:
    """Draw from base samples and align the resulting probability tensor."""
    resolved_sample_shape = torch.Size(sample_shape)
    probabilities = _original_binary_epistemic_rsample_from_base_samples(
        self,
        sample_shape=resolved_sample_shape,
        base_samples=base_samples,
    )
    return _align_epistemic_probability_samples(
        self,
        probabilities,
        resolved_sample_shape,
    )


# Keep support with BoTorch versions whose Posterior API calls
# ``_extended_shape()`` with the default ``None`` value and whose normal sampler
# reads ``posterior.batch_shape`` while updating cached base samples.
BinaryEpistemicProbabilityPosterior._extended_shape = (
    _normalize_binary_epistemic_extended_shape
)
BinaryEpistemicProbabilityPosterior.batch_shape = property(
    _binary_epistemic_batch_shape
)
BinaryEpistemicProbabilityPosterior.rsample = _aligned_binary_epistemic_rsample
BinaryEpistemicProbabilityPosterior.rsample_from_base_samples = (
    _aligned_binary_epistemic_rsample_from_base_samples
)
MultiOutputBernoulliPosterior.batch_shape = property(
    _multioutput_bernoulli_batch_shape
)


@GetSampler.register(BinaryEpistemicProbabilityPosterior)
def _get_binary_epistemic_probability_sampler(
    posterior: BinaryEpistemicProbabilityPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return BoTorch's standard normal sampler for the custom posterior.

    ``SobolQMCNormalSampler`` implements the complete ``MCSampler`` contract,
    including ``_update_base_samples`` used by qNEHVI's cached-Cholesky path.
    The custom posterior forwards the base-sample interface to its latent
    Gaussian posterior.
    """
    del posterior
    return SobolQMCNormalSampler(
        sample_shape=torch.Size(sample_shape),
        seed=seed,
    )


@GetSampler.register(MultiOutputBernoulliPosterior)
def _get_multioutput_bernoulli_sampler(
    posterior: MultiOutputBernoulliPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return a normal QMC sampler for the continuous Bernoulli proxy posterior.

    ``MultiOutputBernoulliPosterior`` implements ``base_sample_shape`` and
    ``rsample_from_base_samples``. Registering the standard Sobol normal sampler
    lets BoTorch's generic MC acquisitions obtain reparameterized probability
    samples without requiring a posterior-specific sampler implementation.
    """
    del posterior
    return SobolQMCNormalSampler(
        sample_shape=torch.Size(sample_shape),
        seed=seed,
    )


__all__ = [
    "BinaryEpistemicProbabilityPosterior",
    "MultiOutputBernoulliPosterior",
]
