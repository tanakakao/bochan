from __future__ import annotations

import torch

from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.components.beta import BetaPosterior
from bochan.models.components.gamma import GammaPosterior
from bochan.models.components.negative_binomial import NegativeBinomialPosterior
from bochan.models.components.poisson import PoissonPosterior


@GetSampler.register(BetaPosterior)
@GetSampler.register(GammaPosterior)
@GetSampler.register(NegativeBinomialPosterior)
@GetSampler.register(PoissonPosterior)
def _get_sampler_for_non_gaussian_posterior(
    posterior: BetaPosterior | GammaPosterior | NegativeBinomialPosterior | PoissonPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return the default MC sampler for custom non-Gaussian posteriors.

    These posteriors wrap a latent Gaussian posterior and expose BoTorch's
    posterior sampling protocol (`base_sample_shape`, `batch_range`, and
    `rsample`). Registering them here lets MC acquisition functions construct a
    default sampler when the user does not pass one explicitly.
    """
    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


__all__ = ["_get_sampler_for_non_gaussian_posterior"]
