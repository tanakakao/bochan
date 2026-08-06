from __future__ import annotations

from typing import Callable, Optional

import torch
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from ._utils import _apply_objective_to_pointwise_score
from bochan.acquisition.binary.epistemic import binary_probability_moments
from .single_output import (
    qBinaryProbabilityVariance,
    _apply_input_transform_for_ipv,
    _ensure_q_batch_for_ipv,
)


class qBinaryIntegratedPosteriorVarianceProxy(qBinaryProbabilityVariance):
    """Binary IPV proxy based on probability epistemic variance."""

    def __init__(
        self,
        model: Model,
        *,
        mc_points: Optional[Tensor] = None,
        integration_beta: float = 25.0,
        num_epistemic_samples: int = 128,
        local_weight: Optional[float] = None,
        integrated_weight: float = 1.0,
        reduction: str = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            num_samples=num_epistemic_samples,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            objective=objective,
        )
        if integration_beta <= 0.0:
            raise ValueError("integration_beta must be positive.")
        if mc_points is not None:
            if mc_points.ndim != 2:
                raise ValueError(
                    "mc_points must have shape [n_mc, d]. "
                    f"Got {tuple(mc_points.shape)}."
                )
            train_inputs = getattr(model, "train_inputs", None)
            ref_X = train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs
            if ref_X is not None:
                mc_points = mc_points.to(device=ref_X.device, dtype=ref_X.dtype)
            self.register_buffer("mc_points", mc_points.detach().clone())
        else:
            self.mc_points = None

        self.integration_beta = float(integration_beta)
        self.num_epistemic_samples = int(num_epistemic_samples)
        self.local_weight = (
            1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        )
        self.integrated_weight = float(integrated_weight)

    def _epistemic_stats(self, X: Tensor) -> tuple[Tensor, Tensor]:
        mean, epistemic_var, _, _ = binary_probability_moments(
            self.model,
            X,
            num_samples=self.num_epistemic_samples,
            eps=self.eps,
        )
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        if epistemic_var.shape[-1] == 1:
            epistemic_var = epistemic_var.squeeze(-1)
        return mean, epistemic_var

    def _integrated_score(self, Xt: Tensor) -> Tensor:
        if self.mc_points is None:
            return Xt.new_zeros(Xt.shape[:-1])

        _, mc_uncertainty = self._epistemic_stats(self.mc_points)
        mc_uncertainty = mc_uncertainty.reshape(-1)
        mc_transformed = _apply_input_transform_for_ipv(
            self.model,
            self.mc_points,
        ).reshape(-1, Xt.shape[-1])

        if mc_uncertainty.numel() != mc_transformed.shape[-2]:
            if mc_transformed.shape[-2] % mc_uncertainty.numel() == 0:
                mc_uncertainty = mc_uncertainty.repeat_interleave(
                    mc_transformed.shape[-2] // mc_uncertainty.numel()
                )
            elif mc_uncertainty.numel() % mc_transformed.shape[-2] == 0:
                mc_uncertainty = mc_uncertainty.reshape(
                    mc_transformed.shape[-2], -1
                ).mean(dim=-1)
            else:
                raise RuntimeError("Could not align mc_points and binary uncertainty.")

        d2 = torch.cdist(
            Xt.reshape(-1, Xt.shape[-1]),
            mc_transformed.detach(),
        ).pow(2)
        weights = torch.exp(-self.integration_beta * d2)
        score = (
            weights * mc_uncertainty.detach().reshape(1, -1)
        ).sum(dim=-1) / weights.sum(dim=-1).clamp_min(self.eps)
        return score.reshape(*Xt.shape[:-1])

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = _ensure_q_batch_for_ipv(X)
        original_batch_shape = raw_X.shape[:-2]
        Xt = _apply_input_transform_for_ipv(self.model, raw_X)

        _, local_score = self._epistemic_stats(raw_X)
        integrated_score = self._integrated_score(Xt)
        if integrated_score.shape != local_score.shape:
            integrated_score = integrated_score.reshape_as(local_score)

        score = self.local_weight * local_score + self.integrated_weight * integrated_score
        score = score - self._candidate_penalty_per_point(Xt)
        score = _apply_objective_to_pointwise_score(
            self,
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="BinaryIntegratedPosteriorVarianceProxy",
        )
        out = self._reduce_q(score)
        self._check_output_shape(
            out,
            original_batch_shape,
            "BinaryIntegratedPosteriorVarianceProxy",
        )
        return out


__all__ = ["qBinaryIntegratedPosteriorVarianceProxy"]
