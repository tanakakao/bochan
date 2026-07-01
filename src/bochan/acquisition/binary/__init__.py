"""Binary-classification acquisition utilities.

This package also registers the custom epistemic probability posterior with
BoTorch's sampler dispatcher. The posterior already implements ``rsample`` and
``rsample_from_base_samples``; the registration below allows standard MC
acquisition functions such as qEHVI and qNEHVI to obtain samples from it.
"""

from __future__ import annotations

import torch
from botorch.posteriors.posterior import Posterior
from botorch.sampling.get_sampler import GetSampler

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


class _BinaryEpistemicPosteriorSampler:
    """Minimal BoTorch-compatible sampler for the custom probability posterior."""

    def __init__(self, sample_shape: torch.Size, seed: int | None = None) -> None:
        self.sample_shape = torch.Size(sample_shape)
        self.seed = seed

    def __call__(self, posterior: Posterior) -> torch.Tensor:
        if self.seed is None:
            return posterior.rsample(self.sample_shape)

        devices = []
        device = getattr(posterior, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            devices = [device]

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            return posterior.rsample(self.sample_shape)


@GetSampler.register(BinaryEpistemicProbabilityPosterior)
def _get_binary_epistemic_probability_sampler(
    posterior: BinaryEpistemicProbabilityPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> _BinaryEpistemicPosteriorSampler:
    """Return a sampler understood by BoTorch's MC acquisition functions."""
    del posterior
    return _BinaryEpistemicPosteriorSampler(
        sample_shape=torch.Size(sample_shape),
        seed=seed,
    )


__all__ = ["BinaryEpistemicProbabilityPosterior"]
