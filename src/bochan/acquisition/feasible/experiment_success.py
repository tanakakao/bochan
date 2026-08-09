"""Feasibility composition for an independently learned experiment-success model."""

from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


def _probability_posterior(model: Any, X: Tensor):
    accessor = getattr(model, "probability_posterior", None)
    if callable(accessor):
        return accessor(X)
    return model.posterior(X)


def _positive_class_probability(model: Any, X: Tensor) -> Tensor:
    posterior = _probability_posterior(model, X)
    probability = getattr(posterior, "mean", None)
    if probability is None:
        raise AttributeError("Experiment success model posterior must expose mean probability.")
    if probability.ndim >= X.ndim and probability.shape[-1] == 1:
        probability = probability.squeeze(-1)
    return probability.clamp(0.0, 1.0)


def _sign_safe_weight(base_value: Tensor, feasibility: Tensor) -> Tensor:
    """Apply bounded monotone feasibility weighting to signed acquisition values.

    Positive utilities use the usual multiplicative probability weighting.
    Negative utilities are multiplied by ``2 - feasibility`` so reducing
    feasibility can never make the score larger merely because the base value is
    negative.  The negative-side multiplier stays in ``[1, 2]`` and therefore
    avoids division by very small probabilities.
    """

    return torch.where(
        base_value >= 0,
        base_value * feasibility,
        base_value * (2.0 - feasibility),
    )


class ExperimentSuccessWeightedAcquisition(AcquisitionFunction):
    """Compose an acquisition with a separately learned success classifier.

    ``success_model`` is trained from all completed experiments, including failed
    rows whose objective values are unavailable.  The base acquisition continues
    to use only objective-model observations.
    """

    def __init__(
        self,
        acqf: AcquisitionFunction,
        success_model: Any,
        *,
        min_success_probability: float = 0.5,
        eta: float = 0.05,
        reduce_q: str = "prod",
    ) -> None:
        super().__init__(model=acqf.model)
        probability = float(min_success_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("min_success_probability must be between 0 and 1.")
        if float(eta) <= 0.0:
            raise ValueError("eta must be positive.")
        if reduce_q not in {"prod", "min", "mean"}:
            raise ValueError("reduce_q must be 'prod', 'min', or 'mean'.")

        self.acqf = acqf
        self.success_model = success_model
        self.min_success_probability = probability
        self.eta = float(eta)
        self.reduce_q = str(reduce_q)
        self.set_X_pending(getattr(acqf, "X_pending", None))

    def success_probability_per_point(self, X: Tensor) -> Tensor:
        """Return posterior ``P(experiment succeeds | X)`` for each q point."""

        return _positive_class_probability(self.success_model, X)

    def feasibility_per_point(self, X: Tensor) -> Tensor:
        """Return a smooth probability-of-meeting-success-threshold score."""

        probability = self.success_probability_per_point(X)
        return torch.sigmoid(
            (probability - self.min_success_probability) / self.eta
        )

    def feasibility(self, X: Tensor) -> Tensor:
        """Reduce per-point success feasibility over a q batch."""

        feasibility = self.feasibility_per_point(X)
        if self.reduce_q == "prod":
            return feasibility.prod(dim=-1)
        if self.reduce_q == "min":
            return feasibility.min(dim=-1).values
        return feasibility.mean(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        base_value = self.acqf(X)
        feasibility = self.feasibility(X)
        return _sign_safe_weight(base_value, feasibility)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        self.X_pending = X_pending


__all__ = [
    "ExperimentSuccessWeightedAcquisition",
]
