from __future__ import annotations

import torch
from torch import Tensor

from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.components.beta import BetaPosterior
from bochan.models.components.gamma import GammaPosterior
from bochan.models.components.negative_binomial import NegativeBinomialPosterior
from bochan.models.components.poisson import PoissonPosterior

NonGaussianPosterior = BetaPosterior | GammaPosterior | NegativeBinomialPosterior | PoissonPosterior


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


def _rsample_mean_from_base_samples(
    posterior: NonGaussianPosterior,
    sample_shape: torch.Size,
    base_samples: Tensor,
) -> Tensor:
    """Sample the latent Gaussian posterior and map samples to the mean scale."""
    f_samples = posterior.latent_posterior.rsample_from_base_samples(
        sample_shape=sample_shape,
        base_samples=base_samples,
    )

    if isinstance(posterior, PoissonPosterior):
        return posterior.likelihood.rate_from_f(f_samples)

    return posterior.likelihood.mean_from_f(f_samples)


def _rsample_from_base_samples(
    self: NonGaussianPosterior,
    sample_shape: torch.Size,
    base_samples: Tensor,
) -> Tensor:
    return _rsample_mean_from_base_samples(
        posterior=self,
        sample_shape=sample_shape,
        base_samples=base_samples,
    )


# BoTorch's NormalMCSampler calls `posterior.rsample_from_base_samples(...)`
# directly. The custom posterior classes already expose the required latent
# Gaussian posterior, so add the method here when the sampler registration module
# is imported.
BetaPosterior.rsample_from_base_samples = _rsample_from_base_samples
GammaPosterior.rsample_from_base_samples = _rsample_from_base_samples
NegativeBinomialPosterior.rsample_from_base_samples = _rsample_from_base_samples
PoissonPosterior.rsample_from_base_samples = _rsample_from_base_samples


__all__ = [
    "_get_sampler_for_non_gaussian_posterior",
    "_rsample_mean_from_base_samples",
]
