"""Finite latent-score posterior for ordinal Deep Ensembles."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class OrdinalEnsemblePosterior(EnsemblePosterior):
    """Deep Ensemble latent posterior with a moment-matched Gaussian bridge.

    ``values`` retain the finite ensemble samples used by BoTorch's
    ``EnsemblePosterior``. For latent-score posteriors (``m=1``), the
    ``distribution`` property and Normal-sampler protocol expose a Gaussian
    moment match across ensemble members so bochan's existing ordinal
    likelihood quadrature and MC acquisitions can consume the posterior.

    Probability-space ordinal posteriors can also reuse this class with
    ``m > 1``. Those keep the regular finite-ensemble sampling semantics and do
    not expose the Gaussian Normal-sampler bridge.
    """

    def _requires_latent_score(self) -> None:
        if self.values.shape[-1] != 1:
            raise RuntimeError(
                "OrdinalEnsemblePosterior Gaussian bridge requires one latent score per point."
            )

    @property
    def distribution(self) -> MultivariateNormal:
        self._requires_latent_score()

        values = self.values[..., 0]
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-2] = int(weights.numel())
        weights = weights.view(*shape)

        mean = (weights * values).sum(dim=-2)
        centered = values - mean.unsqueeze(-2)
        covariance = (
            weights.unsqueeze(-1)
            * centered.unsqueeze(-1)
            * centered.unsqueeze(-2)
        ).sum(dim=-3)

        q = int(values.shape[-1])
        jitter = torch.eye(q, dtype=values.dtype, device=values.device) * 1e-8
        covariance = covariance + jitter
        return MultivariateNormal(mean, covariance)

    def _gaussian_bridge(self) -> GPyTorchPosterior:
        return GPyTorchPosterior(self.distribution)

    @property
    def base_sample_shape(self) -> torch.Size:
        """Normal base-sample shape for latent-score ordinal acquisitions."""
        self._requires_latent_score()
        return self._gaussian_bridge().base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        """t-batch range used by BoTorch Normal samplers."""
        self._requires_latent_score()
        return self._gaussian_bridge().batch_range

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """Draw reparameterized latent samples from the Gaussian bridge.

        For probability-space posteriors (``m > 1``), preserve the inherited
        finite-ensemble index-sampling behavior.
        """
        if self.values.shape[-1] != 1:
            return super().rsample_from_base_samples(
                sample_shape=sample_shape,
                base_samples=base_samples,
            )
        return self._gaussian_bridge().rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        """Sample latent scores from the Gaussian bridge when ``m=1``."""
        if self.values.shape[-1] != 1:
            return super().rsample(sample_shape=sample_shape)
        return self._gaussian_bridge().rsample(sample_shape=sample_shape)

    @property
    def epistemic_variance(self) -> Tensor:
        values = self.values
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-3] = int(weights.numel())
        weights = weights.view(*shape)
        mean = (weights * values).sum(dim=-3)
        return (weights * (values - mean.unsqueeze(-3)).square()).sum(dim=-3).clamp_min(0.0)


__all__ = ["OrdinalEnsemblePosterior"]
