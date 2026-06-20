from __future__ import annotations

from typing import Callable, Optional

import torch
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from ._utils import _apply_objective_to_pointwise_score
from .single_output import (
    qBinaryProbabilityVariance,
    _apply_input_transform_for_ipv,
    _binary_values_to_probability_for_ipv,
    _ensure_q_batch_for_ipv,
)


class qBinaryIntegratedPosteriorVarianceProxy(qBinaryProbabilityVariance):
    """Binary IPV proxy using the same score design as multiclass IPV."""

    def __init__(
        self,
        model: Model,
        *,
        mc_points: Optional[Tensor] = None,
        integration_beta: float = 25.0,
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
        self.local_weight = (
            1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        )
        self.integrated_weight = float(integrated_weight)

    def _probability(self, X: Tensor, *, name: str) -> Tensor:
        prob_fn = getattr(self.model, "probability_posterior", None)
        posterior = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        probability = _binary_values_to_probability_for_ipv(self.model, posterior.mean, apply_sigmoid_if_needed=self.apply_sigmoid_if_needed, eps=self.eps, name=name)
        return probability.squeeze(-1) if probability.shape[-1] == 1 else probability

    def _integrated_score(self, Xt: Tensor) -> Tensor:
        if self.mc_points is None:
            return Xt.new_zeros(Xt.shape[:-1])

        mc_probability = self._probability(
            self.mc_points,
            name="binary mc_points posterior mean",
        ).reshape(-1)
        mc_uncertainty = mc_probability * (1.0 - mc_probability)
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

        probability = self._probability(
            raw_X,
            name="binary candidate posterior mean",
        )
        local_score = probability * (1.0 - probability)
        integrated_score = self._integrated_score(Xt)
        if integrated_score.shape != local_score.shape:
            integrated_score = integrated_score.reshape_as(local_score)

        score = self.local_weight * local_score + self.integrated_weight * integrated_score
        score = score - self._pending_penalty_per_point(Xt)
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
