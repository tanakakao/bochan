"""Finite class-probability posterior for cumulative ordinal ensembles."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor


class OrdinalEnsemblePosterior(EnsemblePosterior):
    """Ordinal class-probability posterior backed by finite model members.

    ``values`` follow BoTorch's ensemble layout
    ``batch_shape x ensemble_size x q x num_classes``. The regular inherited
    ``mean`` is therefore the mean class probability vector. This class exposes
    probability-space epistemic disagreement separately so callers do not
    confuse it with the scalar latent posterior used by the compatibility
    bridge for existing ordinal acquisitions.
    """

    def _ensemble_weights_for_values(self) -> Tensor:
        values = self.values
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-3] = int(weights.numel())
        return weights.view(*shape)

    @property
    def epistemic_variance(self) -> Tensor:
        """Population variance of member class probabilities."""
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
        """Return expected utility for every finite ensemble member."""
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
        return (weights * (values - mean.unsqueeze(-2)).square()).sum(dim=-2)

    def class_probs(self) -> Tensor:
        """Return the mean ordinal class probability vector."""
        return self.mean

    def predict_class(self) -> Tensor:
        """Return the maximum-probability ordinal class."""
        return self.mean.argmax(dim=-1)


__all__ = ["OrdinalEnsemblePosterior"]
