from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary.active_learning.multi_output import (
    MultiOutputMode,
    ReductionType,
    _MultiOutputBinaryClassificationAcqBase,
)


JointUncertaintyMode = Literal["logdet1p", "logdet", "sqrt_trace"]
JointBoundaryMode = Literal["mean_abs", "l2_mean", "max_abs"]


class _MultiOutputLatentStraddleBase(_MultiOutputBinaryClassificationAcqBase):
    """Binary multi-output LSE base sharing Active Learning duplicate controls.

    Hard duplicate exclusion, observed-X resolution, X_pending handling and the
    pointwise objective pipeline are inherited from the common binary
    multi-output Active Learning base. This class adds only LSE-specific latent
    posterior helpers.
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
        )

    def _get_multioutput_latent_stats(self, X: Tensor) -> tuple[Tensor, Tensor]:
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        posterior = self._get_latent_posterior(raw_X)
        mean = self._normalize_mean_shape(posterior.mean, Xt)
        if hasattr(posterior, "variance"):
            variance = posterior.variance
        else:
            dist = getattr(posterior, "distribution", posterior)
            variance = dist.variance
        variance = self._normalize_mean_shape(variance, Xt).clamp_min(self.eps)
        return mean, variance

    def _get_multioutput_probability_mean(
        self,
        X: Tensor,
        *,
        apply_sigmoid_if_needed: bool,
    ) -> Tensor:
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        posterior = self._get_probability_posterior(raw_X)
        mean = self._normalize_mean_shape(posterior.mean, Xt)
        return self._to_probability(
            mean,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            name="probability posterior mean",
        )

    @staticmethod
    def _threshold_vector(
        thresholds: float | Tensor,
        m: int,
        device,
        dtype,
    ) -> Tensor:
        if isinstance(thresholds, (float, int)):
            return torch.full((m,), float(thresholds), device=device, dtype=dtype)
        out = torch.as_tensor(thresholds, device=device, dtype=dtype).reshape(-1)
        if out.numel() == 1:
            return out.expand(m)
        if out.numel() != m:
            raise ValueError(
                f"thresholds must be scalar or shape ({m},), got {tuple(out.shape)}."
            )
        return out

    def _finalize_pointwise_score(
        self,
        score: Tensor,
        raw_X: Tensor,
        Xt: Tensor,
        *,
        name: str,
    ) -> Tensor:
        score = score - self._candidate_penalty_per_point(Xt)
        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name=name,
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, raw_X.shape[:-2], name)
        return out


class _MultiOutputLatentStraddleAcquisition(_MultiOutputLatentStraddleBase):
    """Pointwise multi-output binary latent straddle implementation."""

    def __init__(
        self,
        model,
        beta: float = 1.0,
        thresholds: float | Tensor = 0.0,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        smooth_abs_eps: float = 1e-8,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
        )
        self.beta = float(beta)
        self.thresholds = thresholds
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.smooth_abs_eps = float(smooth_abs_eps)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        mean, variance = self._get_multioutput_latent_stats(raw_X)
        m = int(mean.shape[-1])
        threshold = self._threshold_vector(
            self.thresholds,
            m,
            mean.device,
            mean.dtype,
        ).view(*((1,) * (mean.ndim - 1)), m)
        score_per_output = self.beta * variance.sqrt() - torch.sqrt(
            (mean - threshold).pow(2) + self.smooth_abs_eps
        )
        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )
        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qMultiOutputBinaryLatentStraddleAcquisition",
        )


class qMultiOutputBinaryLatentStraddleAcquisition(_MultiOutputLatentStraddleAcquisition):
    """Multi-output binary latent straddle with common duplicate controls."""


class qMultiOutputBinaryClassEntropyAcquisition(_MultiOutputLatentStraddleBase):
    """Multi-output binary class entropy LSE acquisition."""

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        apply_sigmoid_if_needed: bool = False,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
        )
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._get_multioutput_probability_mean(
            raw_X,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
        )
        score_per_output = -(
            probs * probs.clamp_min(self.eps).log()
            + (1.0 - probs) * (1.0 - probs).clamp_min(self.eps).log()
        )
        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )
        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qMultiOutputBinaryClassEntropyAcquisition",
        )


class qMultiOutputBinaryICUAcquisition(_MultiOutputLatentStraddleBase):
    """Multi-output binary ICU / contour uncertainty acquisition."""

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        apply_sigmoid_if_needed: bool = False,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
        )
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._get_multioutput_probability_mean(
            raw_X,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
        )
        score = self._aggregate_outputs(
            4.0 * probs * (1.0 - probs),
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )
        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qMultiOutputBinaryICUAcquisition",
        )


class qMultiOutputBinaryBoundaryVarianceAcquisition(_MultiOutputLatentStraddleBase):
    """Boundary-weighted latent posterior variance for multi-output binary LSE."""

    def __init__(
        self,
        model,
        thresholds: float | Tensor = 0.0,
        tau: float = 1.0,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
        )
        self.thresholds = thresholds
        self.tau = float(tau)
        self.output_mode = output_mode
        self.output_weights = output_weights
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        mean, variance = self._get_multioutput_latent_stats(raw_X)
        m = int(mean.shape[-1])
        threshold = self._threshold_vector(
            self.thresholds,
            m,
            mean.device,
            mean.dtype,
        ).view(*((1,) * (mean.ndim - 1)), m)
        tau = torch.as_tensor(self.tau, device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        boundary_weight = torch.exp(-0.5 * ((mean - threshold) / tau).pow(2))
        score = self._aggregate_outputs(
            variance * boundary_weight,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )
        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qMultiOutputBinaryBoundaryVarianceAcquisition",
        )


class qMultiOutputBinaryJointLatentStraddleAcquisition(_MultiOutputLatentStraddleBase):
    """Joint q-batch multi-output binary latent straddle acquisition."""

    def __init__(
        self,
        model,
        beta: float = 2.0,
        thresholds: float | Tensor = 0.0,
        uncertainty_mode: JointUncertaintyMode = "logdet1p",
        boundary_mode: JointBoundaryMode = "l2_mean",
        tau: float = 1.0,
        jitter: float = 1e-6,
        marginalize_pending: bool = True,
        same_batch_penalty_weight: float = 0.1,
        pending_penalty_weight: float = 0.1,
        observed_penalty_weight: float = 0.0,
        distance_beta: float = 20.0,
        duplicate_tol: float = 1e-6,
        hard_duplicate_penalty: float = 1e6,
        hard_duplicate_tol: Optional[float] = None,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-10,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        resolved_tol = float(duplicate_tol if hard_duplicate_tol is None else hard_duplicate_tol)
        super().__init__(
            model=model,
            reduction="sum",
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=distance_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=distance_beta,
            hard_duplicate_tol=resolved_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
        )
        self.beta = float(beta)
        self.thresholds = thresholds
        self.uncertainty_mode = uncertainty_mode
        self.boundary_mode = boundary_mode
        self.tau = float(tau)
        self.jitter = float(jitter)
        self.marginalize_pending = bool(marginalize_pending)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.distance_beta = float(distance_beta)
        self.duplicate_tol = resolved_tol
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self._set_multioutput_classification_objective(objective)

    def _joint_mean_cov(self, X: Tensor) -> tuple[Tensor, Tensor]:
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        posterior = self._get_latent_posterior(raw_X)
        mean = self._normalize_mean_shape(posterior.mean, Xt)
        q = int(mean.shape[-2])
        m = int(mean.shape[-1])

        dist = getattr(posterior, "distribution", None)
        cov = None
        if dist is not None and hasattr(dist, "covariance_matrix"):
            cov = dist.covariance_matrix
        elif hasattr(posterior, "mvn") and hasattr(posterior.mvn, "covariance_matrix"):
            cov = posterior.mvn.covariance_matrix

        if cov is None:
            variance = getattr(posterior, "variance", None)
            if variance is None and dist is not None:
                variance = getattr(dist, "variance", None)
            if variance is None:
                raise AttributeError("Could not extract latent variance for joint binary LSE.")
            variance = self._normalize_mean_shape(variance, Xt).clamp_min(self.eps)
            cov = torch.diag_embed(variance.reshape(*variance.shape[:-2], q * m))
        else:
            target = mean.shape[:-2] + torch.Size([q * m, q * m])
            if cov.shape != target:
                if cov.numel() == math.prod(target):
                    cov = cov.reshape(target)
                else:
                    variance = getattr(posterior, "variance", None)
                    if variance is None and dist is not None:
                        variance = getattr(dist, "variance", None)
                    variance = self._normalize_mean_shape(variance, Xt).clamp_min(self.eps)
                    cov = torch.diag_embed(variance.reshape(*variance.shape[:-2], q * m))

        eye = torch.eye(q * m, dtype=cov.dtype, device=cov.device)
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + self.jitter * eye
        return mean, cov

    def _joint_uncertainty(self, cov: Tensor) -> Tensor:
        n = int(cov.shape[-1])
        eye = torch.eye(n, dtype=cov.dtype, device=cov.device)
        if self.uncertainty_mode == "logdet1p":
            sign, logdet = torch.linalg.slogdet(eye + cov / max(self.tau**2, self.eps))
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite matrix encountered in logdet1p.")
            return 0.5 * logdet
        if self.uncertainty_mode == "logdet":
            sign, logdet = torch.linalg.slogdet(cov)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite covariance encountered in logdet.")
            return 0.5 * logdet
        if self.uncertainty_mode == "sqrt_trace":
            return torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(self.eps).sqrt()
        raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode!r}.")

    def _joint_score(self, X: Tensor) -> Tensor:
        mean, cov = self._joint_mean_cov(X)
        m = int(mean.shape[-1])
        threshold = self._threshold_vector(
            self.thresholds,
            m,
            mean.device,
            mean.dtype,
        ).view(*((1,) * (mean.ndim - 1)), m)
        diff = mean - threshold
        if self.boundary_mode == "mean_abs":
            boundary = diff.abs().mean(dim=(-2, -1))
        elif self.boundary_mode == "l2_mean":
            boundary = diff.pow(2).mean(dim=(-2, -1)).sqrt()
        elif self.boundary_mode == "max_abs":
            boundary = diff.abs().amax(dim=(-2, -1))
        else:
            raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")
        return self.beta * self._joint_uncertainty(cov) - boundary

    def _same_batch_soft_penalty(self, Xt: Tensor) -> Tensor:
        q = int(Xt.shape[-2])
        if q <= 1 or self.same_batch_penalty_weight <= 0.0:
            return Xt.new_zeros(Xt.shape[:-2])
        dist = torch.cdist(Xt, Xt)
        eye = torch.eye(q, dtype=torch.bool, device=Xt.device)
        while eye.ndim < dist.ndim:
            eye = eye.unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        nearest = dist.min(dim=-1).values
        return self.same_batch_penalty_weight * torch.exp(
            -self.distance_beta * nearest
        ).sum(dim=-1)

    def _joint_repulsion_penalty(self, X: Tensor) -> Tensor:
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        penalty = self._same_batch_soft_penalty(Xt)
        penalty = penalty + self._candidate_penalty_per_point(Xt).sum(dim=-1)
        return penalty

    @staticmethod
    def _expand_pending_to_batch(X_pending: Tensor, batch_shape: torch.Size) -> Tensor:
        if X_pending.ndim == 1:
            X_pending = X_pending.view(1, -1)
        if X_pending.ndim == 2:
            m, d = X_pending.shape
            return X_pending.view(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        m, d = X_pending.shape[-2:]
        return X_pending.reshape(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        batch_shape = raw_X.shape[:-2]
        Xp = getattr(self, "X_pending", None)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            value = self._joint_score(raw_X) - self._joint_repulsion_penalty(raw_X)
        else:
            Xp_batch = self._expand_pending_to_batch(
                Xp.to(device=raw_X.device, dtype=raw_X.dtype),
                batch_shape,
            )
            value = (
                self._joint_score(torch.cat([Xp_batch, raw_X], dim=-2))
                - self._joint_score(Xp_batch)
                - self._joint_repulsion_penalty(raw_X)
            )

        if self.objective is not None:
            try:
                value = self.objective(value, X=raw_X)
            except TypeError:
                value = self.objective(value)
            if not torch.is_tensor(value):
                raise TypeError(
                    "qMultiOutputBinaryJointLatentStraddleAcquisition objective must return Tensor."
                )
        self._check_output_shape(
            value,
            batch_shape,
            "qMultiOutputBinaryJointLatentStraddleAcquisition",
        )
        return value


# Backward-supported internal implementation name.
_JointMultiOutputLatentStraddleAcquisition = qMultiOutputBinaryJointLatentStraddleAcquisition


__all__ = [
    "qMultiOutputBinaryLatentStraddleAcquisition",
    "qMultiOutputBinaryJointLatentStraddleAcquisition",
    "qMultiOutputBinaryClassEntropyAcquisition",
    "qMultiOutputBinaryICUAcquisition",
    "qMultiOutputBinaryBoundaryVarianceAcquisition",
]
