"""Binary-classification acquisition utilities.

This package registers the custom binary probability posteriors with BoTorch's
sampler dispatcher. The posteriors expose the normal base-sample interface
required by BoTorch, so standard normal MC samplers can be reused by qEHVI,
qNEHVI, and other Monte Carlo acquisition functions.
"""

from __future__ import annotations

import torch
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.classification.binary.base.multioutput import (
    MultiOutputBernoulliPosterior,
)

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


# Keep compatibility with BoTorch versions whose Posterior API calls
# ``_extended_shape()`` with the default ``None`` value and whose normal sampler
# reads ``posterior.batch_shape`` while updating cached base samples.
BinaryEpistemicProbabilityPosterior._extended_shape = (
    _normalize_binary_epistemic_extended_shape
)
BinaryEpistemicProbabilityPosterior.batch_shape = property(
    _binary_epistemic_batch_shape
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
