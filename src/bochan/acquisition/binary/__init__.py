"""Binary-classification acquisition utilities.

Custom binary probability posteriors are registered with BoTorch's sampler
dispatcher through its public ``GetSampler.register`` extension point. Importing
this package does not mutate acquisition classes or posterior class methods.
"""

from __future__ import annotations

import torch
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.multioutput.binary import MultiOutputBernoulliPosterior

from .epistemic import BinaryEpistemicProbabilityPosterior


@GetSampler.register(BinaryEpistemicProbabilityPosterior)
def _get_binary_epistemic_probability_sampler(
    posterior: BinaryEpistemicProbabilityPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return BoTorch's standard normal sampler for the custom posterior."""

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
    """Return a normal QMC sampler for the continuous Bernoulli proxy posterior."""

    del posterior
    return SobolQMCNormalSampler(
        sample_shape=torch.Size(sample_shape),
        seed=seed,
    )


__all__ = [
    "BinaryEpistemicProbabilityPosterior",
    "MultiOutputBernoulliPosterior",
]
