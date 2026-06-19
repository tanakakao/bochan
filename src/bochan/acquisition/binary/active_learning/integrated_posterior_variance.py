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
    """Differentiable integrated probability-variance proxy for binary models.

    This acquisition evaluates binary probability uncertainty ``p(1-p)`` on
    ``mc_points`` and scores candidate points by differentiable RBF coverage of
    those uncertain reference regions. Unlike the fantasy/refit NIPV variant,
    this proxy remains differentiable with respect to candidate ``X`` and can be
    optimized with BoTorch's standard ``optimize_acqf`` / L-BFGS-B backend.

    Args:
        model: Binary classification model.
        mc_points: Integration/reference points with shape ``n_mc x d``.
        kernel_lengthscale: RBF lengthscale in transformed input space.
        normalize_weights: Normalize RBF weights over ``mc_points``.
        reduction: q-batch reduction inherited from
            :class:`qBinaryProbabilityVariance`.
        pending_penalty_weight: Penalty applied near pending points.
        pending_penalty_beta: Distance decay for pending-point penalty.
        apply_sigmoid_if_needed: Convert latent posterior means to
            probabilities when necessary.
        eps: Numerical stability constant.
        objective: Optional classification score objective, including
            InputPerturbation risk aggregation.
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        *,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
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
        if mc_points.ndim != 2:
            raise ValueError(
                "mc_points must have shape [n_mc, d]. "
                f"Got shape={tuple(mc_points.shape)}."
            )
        if kernel_lengthscale <= 0.0:
            raise ValueError("kernel_lengthscale must be positive.")

        ref_X = getattr(model, "train_X", None)
        if ref_X is None:
            train_inputs = getattr(model, "train_inputs", None)
            if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
                ref_X = train_inputs[0]
        if ref_X is not None:
            mc_points = mc_points.to(device=ref_X.device, dtype=ref_X.dtype)

        self.register_buffer("mc_points", mc_points.detach().clone())
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    def _reference_points_and_uncertainty(self) -> tuple[Tensor, Tensor]:
        """Return transformed integration points and detached uncertainty."""
        self._prepare_eval()

        prob_fn = getattr(self.model, "probability_posterior", None)
        posterior = (
            prob_fn(self.mc_points)
            if callable(prob_fn)
            else self.model.posterior(self.mc_points)
        )
        prob = _binary_values_to_probability_for_ipv(
            posterior.mean,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="binary mc_points posterior mean",
        ).reshape(-1)

        transformed = _apply_input_transform_for_ipv(
            self.model,
            self.mc_points,
        ).reshape(-1, self.mc_points.shape[-1])

        n_ref = int(transformed.shape[-2])
        if prob.numel() == n_ref:
            aligned_prob = prob
        elif n_ref % prob.numel() == 0:
            aligned_prob = prob.repeat_interleave(n_ref // prob.numel())
        elif prob.numel() % n_ref == 0:
            aligned_prob = prob.reshape(n_ref, prob.numel() // n_ref).mean(dim=-1)
        else:
            raise RuntimeError(
                "Could not align binary probability values with transformed "
                "mc_points. "
                f"prob.shape={tuple(prob.shape)}, "
                f"transformed.shape={tuple(transformed.shape)}."
            )

        uncertainty = aligned_prob * (1.0 - aligned_prob)
        return transformed.detach(), uncertainty.detach()

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = _ensure_q_batch_for_ipv(X)
        original_batch_shape = raw_X.shape[:-2]

        Xt = _apply_input_transform_for_ipv(self.model, raw_X)
        ref_points, ref_uncertainty = self._reference_points_and_uncertainty()
        ref_points = ref_points.to(device=Xt.device, dtype=Xt.dtype)
        ref_uncertainty = ref_uncertainty.to(device=Xt.device, dtype=Xt.dtype)

        X2d = Xt.reshape(-1, Xt.shape[-1])
        d2 = torch.cdist(X2d, ref_points).pow(2)
        d2 = d2.reshape(*Xt.shape[:-1], ref_points.shape[-2])

        lengthscale2 = max(self.kernel_lengthscale**2, self.eps)
        weights = torch.exp(-0.5 * d2 / lengthscale2)
        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        view_shape = (1,) * (weights.ndim - 1) + (ref_uncertainty.numel(),)
        score = (weights * ref_uncertainty.view(*view_shape)).sum(dim=-1)
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
