from __future__ import annotations

import torch
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.regression.beta._components import BetaPosterior
from bochan.models.regression.count.negative_binomial._components import NegativeBinomialPosterior
from bochan.models.regression.count.poisson._components import PoissonPosterior
from bochan.models.regression.gamma._components import GammaPosterior

NonGaussianPosterior = (
    BetaPosterior | GammaPosterior | NegativeBinomialPosterior | PoissonPosterior
)


@GetSampler.register(BetaPosterior)
@GetSampler.register(GammaPosterior)
@GetSampler.register(NegativeBinomialPosterior)
@GetSampler.register(PoissonPosterior)
def _get_sampler_for_non_gaussian_posterior(
    posterior: NonGaussianPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """Return the default MC sampler for custom non-Gaussian posteriors.

    These posteriors wrap a latent Gaussian posterior and expose BoTorch's
    posterior sampling protocol (`base_sample_shape`, `batch_range`, and
    `rsample_from_base_samples`). Registering them here lets MC acquisition
    functions construct a default sampler when the user does not pass one
    explicitly.
    """
    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


__all__ = [
    "_get_sampler_for_non_gaussian_posterior",
]
