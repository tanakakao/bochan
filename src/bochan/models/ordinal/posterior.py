"""Finite ordinal ensemble posterior utilities and latent-score bridge."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class OrdinalEnsemblePosterior(EnsemblePosterior):
    """Finite ordinal posterior with probability and latent-score semantics.

    ``values`` follow BoTorch's ensemble layout
    ``batch_shape x ensemble_size x q x m``.

    For ordinal class-probability posteriors, ``m=num_classes`` and the
    inherited ``mean`` is the mean class-probability vector. Probability-space
    helpers expose epistemic disagreement, expected utility, and class
    prediction directly from those finite members.

    For latent-score posteriors, ``m=1``. The ``distribution`` property and
    Normal-sampler protocol expose a Gaussian moment match across ensemble
    members so bochan's existing ordinal likelihood quadrature and MC
    acquisitions can consume the posterior.
    """

    def _ensemble_weights_for_values(self) -> Tensor:
        values = self.values
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-3] = int(weights.numel())
        return weights.view(*shape)

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
        """Draw samples using the appropriate ordinal posterior semantics."""
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
        """Population variance across finite ordinal ensemble members."""
        values = self.values
        weights = self._ensemble_weights_for_values()
        mean = self.mean.unsqueeze(-3)
        return (weights * (values - mean).square()).sum(dim=-3).clamp_min(0.0)

    def expected_utility(self, utilities: Tensor) -> Tensor:
        """Return expected utility under the mean ordinal class distribution."""
        utilities = torch.as_tensor(
            utilities,
            device=self.mean.device,
            dtype=self.mean.dtype,
        ).reshape(-1)
        if utilities.numel() != self.mean.shape[-1]:
            raise ValueError(
                f"utilities must have length {self.mean.shape[-1]}, "
                f"got {utilities.numel()}."
            )
        return (self.mean * utilities).sum(dim=-1)

    def member_expected_utility(self, utilities: Tensor) -> Tensor:
        """Return expected utility for every finite ordinal ensemble member."""
        utilities = torch.as_tensor(
            utilities,
            device=self.values.device,
            dtype=self.values.dtype,
        ).reshape(-1)
        if utilities.numel() != self.values.shape[-1]:
            raise ValueError(
                f"utilities must have length {self.values.shape[-1]}, "
                f"got {utilities.numel()}."
            )
        return (self.values * utilities).sum(dim=-1)

    def utility_epistemic_variance(self, utilities: Tensor) -> Tensor:
        """Return member-disagreement variance in expected-utility space."""
        values = self.member_expected_utility(utilities)
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-2] = int(weights.numel())
        weights = weights.view(*shape)
        mean = (weights * values).sum(dim=-2)
        return (
            weights * (values - mean.unsqueeze(-2)).square()
        ).sum(dim=-2).clamp_min(0.0)

    def class_probs(self) -> Tensor:
        """Return the mean ordinal class-probability vector."""
        return self.mean

    def predict_class(self) -> Tensor:
        """Return the maximum-probability ordinal class."""
        return self.mean.argmax(dim=-1)


__all__ = ["OrdinalEnsemblePosterior"]
