"""Finite latent-score posterior for ordinal Deep Ensembles."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class OrdinalEnsemblePosterior(EnsemblePosterior):
    """Deep Ensemble latent posterior with a moment-matched Gaussian bridge.

    ``values`` retain the finite ensemble samples used by BoTorch's
    ``EnsemblePosterior``. The ``distribution`` property provides a Gaussian
    moment match across ensemble members so bochan's existing ordinal
    likelihood quadrature can keep using ``posterior.distribution``.
    """

    @property
    def distribution(self) -> MultivariateNormal:
        if self.values.shape[-1] != 1:
            raise RuntimeError("OrdinalEnsemblePosterior requires one latent score per point.")

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
