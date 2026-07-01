"""Binary-classification acquisition utilities.

This package registers the custom epistemic probability posterior with
BoTorch's sampler dispatcher. The posterior exposes the normal base-sample
interface required by BoTorch, so standard normal MC samplers can be reused by
qEHVI, qNEHVI, and other cached Monte Carlo acquisition functions.
"""

from __future__ import annotations

import torch
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from .epistemic import BinaryEpistemicProbabilityPosterior


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


# Keep compatibility with BoTorch versions whose Posterior API calls
# ``_extended_shape()`` with the default ``None`` value.
BinaryEpistemicProbabilityPosterior._extended_shape = (
    _normalize_binary_epistemic_extended_shape
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
    The custom posterior already forwards ``base_sample_shape``, ``batch_range``,
    and ``rsample_from_base_samples`` to its latent Gaussian posterior.
    """
    del posterior
    return SobolQMCNormalSampler(
        sample_shape=torch.Size(sample_shape),
        seed=seed,
    )


__all__ = ["BinaryEpistemicProbabilityPosterior"]
