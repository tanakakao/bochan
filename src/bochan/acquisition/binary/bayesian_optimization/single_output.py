"""Binary probability-of-feasibility acquisition.

Standard binary qEI / qPI / qUCB live in ``standard.py`` and use BoTorch joint
q-batch semantics. This module contains only the binary feasibility acquisition
that has task-specific probability semantics.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary._probability import latent_samples_to_binary_probabilities
from bochan.acquisition.binary.base import ReductionType, _BinaryClassificationAcqBase

from ._utils import apply_score_objective

PoFMode = Literal["mc_likelihood", "mc_sigmoid", "latent_cdf"]


class qBinaryProbabilityOfFeasibility(_BinaryClassificationAcqBase):
    """Probability of feasibility for binary classification."""

    def __init__(
        self,
        model,
        num_samples: int = 32,
        threshold: float = 0.0,
        mode: PoFMode = "mc_likelihood",
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            objective=objective,
        )
        self.num_samples = int(num_samples)
        self.threshold = float(threshold)
        self.mode = mode

    def _mc_likelihood_prob(self, latent_dist, orig: torch.Size) -> Tensor:
        f_samples = latent_dist.rsample(torch.Size([self.num_samples]))
        expected = self.num_samples * math.prod(orig)
        if f_samples.numel() != expected:
            raise RuntimeError(
                f"Unexpected sample shape: got {tuple(f_samples.shape)}, "
                f"numel={f_samples.numel()}, expected={expected}"
            )
        f_samples = f_samples.reshape(self.num_samples, *orig)
        return latent_samples_to_binary_probabilities(
            self.model,
            f_samples,
            eps=self.eps,
            name="f_samples via binary likelihood",
        ).clamp(self.eps, 1.0 - self.eps).mean(dim=0)

    def _latent_cdf_prob(self, latent_dist, orig: torch.Size) -> Tensor:
        mu = self._reshape_pointwise_tensor(latent_dist.mean, orig)
        var = self._reshape_pointwise_tensor(latent_dist.variance, orig).clamp_min(self.eps)
        sigma = var.sqrt()
        z = (mu - self.threshold) / sigma
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return normal.cdf(z).clamp(self.eps, 1.0 - self.eps)

    def _pointwise_pof_from_latent_dist(self, latent_dist, orig: torch.Size) -> Tensor:
        if self.mode in {"mc_likelihood", "mc_sigmoid"}:
            return self._mc_likelihood_prob(latent_dist, orig)
        if self.mode == "latent_cdf":
            return self._latent_cdf_prob(latent_dist, orig)
        raise ValueError(f"Unknown mode: {self.mode}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        score = self._pointwise_pof_from_latent_dist(latent_dist, orig)

        penalty = self._candidate_penalty_per_point(Xt)
        if penalty.shape == score.shape:
            score = score - penalty
        elif penalty.numel() == score.numel():
            score = score - penalty.reshape_as(score)
        elif self.pending_penalty_weight > 0:
            raise RuntimeError(
                "Pending penalty shape mismatch: "
                f"score={tuple(score.shape)}, penalty={tuple(penalty.shape)}"
            )

        score = apply_score_objective(
            self,
            score,
            X=X,
            attr_name="objective",
            name="PoF",
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "PoF")
        return out


__all__ = ["PoFMode", "qBinaryProbabilityOfFeasibility"]
