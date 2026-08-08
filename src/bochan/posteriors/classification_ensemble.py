"""Probability posteriors backed by finite classifier ensembles."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor


class ClassificationEnsemblePosterior(EnsemblePosterior):
    """Ensemble probability posterior with observation/epistemic decomposition.

    ``values`` follow BoTorch's ensemble layout ``batch x s x q x m``. For
    binary classification ``m=1`` and values are class-1 probabilities. For
    multiclass classification ``m=C`` and values are full class-probability
    vectors.

    The regular ``variance`` property intentionally follows bochan's existing
    classification convention and returns marginal label variance ``p(1-p)``.
    Model disagreement is exposed separately as ``epistemic_variance`` and is
    also represented by ``rsample()``, which samples ensemble members.
    """

    def _ensemble_weights_for_values(self) -> Tensor:
        values = self.values
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-3] = int(weights.numel())
        return weights.view(*shape)

    @property
    def variance(self) -> Tensor:
        p = self.mean.clamp(0.0, 1.0)
        return p * (1.0 - p)

    @property
    def epistemic_variance(self) -> Tensor:
        values = self.values
        mean = self.mean
        weights = self._ensemble_weights_for_values()
        return (weights * (values - mean.unsqueeze(-3)).square()).sum(dim=-3).clamp_min(0.0)

    @property
    def aleatoric_variance(self) -> Tensor:
        values = self.values.clamp(0.0, 1.0)
        weights = self._ensemble_weights_for_values()
        return (weights * values * (1.0 - values)).sum(dim=-3).clamp_min(0.0)

    @property
    def total_label_variance(self) -> Tensor:
        return self.variance

    @property
    def covariance_matrix(self) -> Tensor:
        """Population covariance across q points for binary probability samples."""
        if self.values.shape[-1] != 1:
            raise NotImplementedError(
                "covariance_matrix is currently defined only for binary probability outputs."
            )
        values = self.values[..., 0]
        weights = self.weights.to(device=values.device, dtype=values.dtype)
        shape = [1] * values.ndim
        shape[-2] = int(weights.numel())
        weights = weights.view(*shape)
        mean = (weights * values).sum(dim=-2)
        centered = values - mean.unsqueeze(-2)
        return (
            weights.unsqueeze(-1)
            * centered.unsqueeze(-1)
            * centered.unsqueeze(-2)
        ).sum(dim=-3).clamp_min(0.0)

    def class_probs(self) -> Tensor:
        return self.mean

    def predict_class(self) -> Tensor:
        if self.mean.shape[-1] == 1:
            return (self.mean[..., 0] >= 0.5).long()
        return self.mean.argmax(dim=-1)


__all__ = ["ClassificationEnsemblePosterior"]
