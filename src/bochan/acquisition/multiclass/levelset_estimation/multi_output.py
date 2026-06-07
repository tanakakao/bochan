from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.multi_output import (
    OutputModeType,
    OutputReductionType,
    ReductionType,
    _DirectMultiOutputMulticlassAcqBase,
)
from bochan.acquisition.multiclass.base import ClassReductionType

UncertaintyMode = Literal["bernoulli", "posterior", "combined"]
JointUncertaintyMode = Literal["logdet1p", "logdet", "sqrt_trace", "trace"]
JointBoundaryMode = Literal["mean_abs", "l2_mean", "max_abs"]


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _align_pointwise_to_reference(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    """Align a pointwise tensor to a reference tensor."""

    if value.ndim >= 1 and value.shape[-1] == 1 and reference.ndim >= 1 and reference.shape[-1] != 1:
        value = value.squeeze(-1)

    while value.ndim > reference.ndim:
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        else:
            value = value.mean(dim=-1)

    if value.shape == reference.shape:
        return value.to(reference)
    if value.shape == reference.shape[:-1]:
        return value.unsqueeze(-1).expand_as(reference).to(reference)
    if value.ndim == reference.ndim and value.shape[:-1] == reference.shape[:-1]:
        q_ref = reference.shape[-1]
        q_value = value.shape[-1]
        if q_ref % q_value == 0:
            return value.repeat_interleave(q_ref // q_value, dim=-1).to(reference)
        if q_value % q_ref == 0:
            return value.reshape(*reference.shape[:-1], q_ref, q_value // q_ref).mean(dim=-1).to(reference)
    if value.numel() == reference.numel():
        return value.reshape_as(reference).to(reference)
    if value.numel() == 1:
        return value.reshape(()).expand_as(reference).to(reference)
    raise RuntimeError(
        f"{name}: cannot align pointwise value. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


def _class_entropy(probs: Tensor, *, eps: float) -> Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _boundary_weight(value: Tensor, threshold: float, *, bandwidth: float, eps: float) -> Tensor:
    return torch.exp(-((value - threshold).abs() / max(float(bandwidth), eps)))


class _MultiOutputMulticlassTargetProbabilityBase(_DirectMultiOutputMulticlassAcqBase):
    """Complete base for multi-output multiclass level-set acquisitions.

    The base operates on multiclass probabilities with shape
    ``batch_shape x q_like x m x C`` and computes one score per output before
    output aggregation. It also mirrors the single-output complete version with
    observed / pending / same-batch penalties and optional score objectives.
    """

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        threshold: float = 0.5,
        class_reduction: ClassReductionType = "mean",
        reduction: ReductionType = "mean",
        output_mode: OutputModeType = "mean",
        output_reduction: OutputReductionType | None = None,
        output_weights: Tensor | Sequence[float] | None = None,
        normalize_output_weights: bool = True,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        num_samples: int = 128,
        eps: float = 1e-8,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            output_mode=output_mode,
            output_reduction=output_reduction,
            output_weights=output_weights,
            normalize_output_weights=normalize_output_weights,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            objective=None,
        )
        self.target_class = target_class
        self.output_target_classes = None if output_target_classes is None else [int(i) for i in output_target_classes]
        self.threshold = float(threshold)
        self.class_reduction = class_reduction
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()
        self.num_samples = int(num_samples)
        self.levelset_objective = objective

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()

    def _reduce_classes(self, selected: Tensor) -> Tensor:
        if self.class_reduction == "mean":
            return selected.mean(dim=-1)
        if self.class_reduction == "sum":
            return selected.sum(dim=-1)
        if self.class_reduction == "max":
            return selected.max(dim=-1).values
        if self.class_reduction == "min":
            return selected.min(dim=-1).values
        if self.class_reduction == "prod":
            return selected.prod(dim=-1)
        raise ValueError(f"Unknown class_reduction: {self.class_reduction!r}.")

    def _target_prob_per_output(self, probs: Tensor) -> Tensor:
        """Select target probabilities from ``... x m x C`` probabilities."""

        n_outputs = int(probs.shape[-2])
        if self.output_target_classes is not None:
            if len(self.output_target_classes) != n_outputs:
                raise ValueError(
                    "output_target_classes length must match number of outputs. "
                    f"Got {len(self.output_target_classes)} and {n_outputs}."
                )
            idx = torch.as_tensor(self.output_target_classes, device=probs.device, dtype=torch.long)
            gather_idx = idx.view(*([1] * (probs.ndim - 2)), n_outputs, 1).expand(*probs.shape[:-1], 1)
            return torch.gather(probs, dim=-1, index=gather_idx).squeeze(-1)

        if self.target_class is None:
            return probs.max(dim=-1).values
        if isinstance(self.target_class, int):
            return probs[..., int(self.target_class)]
        indices = [int(i) for i in self.target_class]
        selected = probs[..., indices]
        return self._reduce_classes(selected)

    def _target_prob_mean_per_output(self, X: Tensor) -> Tensor:
        return self._target_prob_per_output(self._mean_probs(X))

    def _target_prob_samples_per_output(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        return self._target_prob_per_output(self._sample_probs(X, num_samples=int(num_samples or self.num_samples)))

    def _target_uncertainty(self, X: Tensor, p: Tensor, *, mode: UncertaintyMode) -> Tensor:
        if mode == "bernoulli":
            return (p * (1.0 - p)).clamp_min(self.eps).sqrt()
        samples = self._target_prob_samples_per_output(X)
        posterior_std = samples.std(dim=0, unbiased=False).clamp_min(self.eps)
        if mode == "posterior":
            return posterior_std
        if mode == "combined":
            bernoulli_var = (p * (1.0 - p)).clamp_min(0.0)
            return (posterior_std.pow(2) + bernoulli_var).clamp_min(self.eps).sqrt()
        raise ValueError(f"Unknown uncertainty_mode: {mode!r}.")

    def _reference_points_transformed(self, X_ref, *, ref: Tensor) -> Tensor | None:
        if X_ref is None:
            return None
        if isinstance(X_ref, (list, tuple)):
            pieces = [self._reference_points_transformed(x, ref=ref) for x in X_ref if x is not None]
            pieces = [p for p in pieces if p is not None and p.numel() > 0]
            return None if len(pieces) == 0 else torch.cat(pieces, dim=-2)
        X_ref = torch.as_tensor(X_ref, device=ref.device, dtype=ref.dtype)
        if X_ref.numel() == 0:
            return None
        Xt = self._ensure_q_batch(self._apply_input_transform(X_ref))
        return Xt.reshape(-1, Xt.shape[-1]).to(ref)

    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.observed_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-1])
        Xobs = self._reference_points_transformed(self.X_observed, ref=Xt)
        if Xobs is None:
            return Xt.new_zeros(Xt.shape[:-1])
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xobs).min(dim=-1).values
        return self.observed_penalty_weight * torch.exp(-self.observed_penalty_beta * dist.reshape(Xt.shape[:-1]))

    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.same_batch_penalty_weight <= 0 or Xt.shape[-2] <= 1:
            return Xt.new_zeros(Xt.shape[:-2])
        Xb = Xt.reshape(-1, Xt.shape[-2], Xt.shape[-1])
        d = torch.cdist(Xb, Xb)
        q = Xt.shape[-2]
        eye = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, float("inf"))
        penalty = 0.5 * self.same_batch_penalty_weight * torch.exp(-self.same_batch_penalty_beta * d).sum(dim=(-1, -2))
        return penalty.reshape(*Xt.shape[:-2])

    def _apply_levelset_objective(self, score: Tensor, X: Tensor, *, name: str) -> Tensor:
        if self.levelset_objective is None:
            return score
        try:
            out = self.levelset_objective(score, X=X)
        except TypeError:
            out = self.levelset_objective(score)
        if not torch.is_tensor(out):
            raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
        return out

    def _score_to_value(self, score_per_output: Tensor, raw_X: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        score = self._aggregate_outputs(score_per_output)
        pending = _align_pointwise_to_reference(self._pending_penalty_per_point(Xt), score, name=f"{name}.pending_penalty")
        observed = _align_pointwise_to_reference(self._observed_penalty_per_point(Xt), score, name=f"{name}.observed_penalty")
        score = score - pending - observed
        score = self._apply_levelset_objective(score, raw_X, name=name)
        if score.shape == raw_X.shape[:-2]:
            value = score
        else:
            value = self._reduce_q(score)
        value = value - self._same_batch_penalty(Xt)
        return self._finalize(value, raw_X, name=name)


class qMultiOutputMulticlassLatentStraddleAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Multi-output target-class probability straddle acquisition."""

    def __init__(self, model, *, beta: float = 1.0, uncertainty_mode: UncertaintyMode = "combined", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = float(beta)
        self.uncertainty_mode = uncertainty_mode

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score_per_output = self.beta * uncertainty - (p - self.threshold).abs()
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qMultiOutputMulticlassJointLatentStraddleAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Joint q-batch multi-output target-probability straddle acquisition."""

    def __init__(
        self,
        model,
        *,
        beta: float = 2.0,
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
        sampler: Optional[MCSampler] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        super().__init__(
            model=model,
            reduction="sum",
            sampler=sampler,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=distance_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=distance_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=distance_beta,
            **kwargs,
        )
        self.beta = float(beta)
        self.uncertainty_mode = uncertainty_mode
        self.boundary_mode = boundary_mode
        self.tau = float(tau)
        self.jitter = float(jitter)
        self.marginalize_pending = bool(marginalize_pending)
        self.distance_beta = float(distance_beta)
        self.duplicate_tol = float(duplicate_tol)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)

    @staticmethod
    def _expand_pending_to_batch(X_pending: Tensor, batch_shape: torch.Size) -> Tensor:
        if X_pending.ndim == 1:
            X_pending = X_pending.view(1, -1)
        if X_pending.ndim == 2:
            m, d = X_pending.shape
            return X_pending.view(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        if X_pending.ndim >= 3:
            m, d = X_pending.shape[-2], X_pending.shape[-1]
            leading = X_pending.shape[:-2]
            if leading == batch_shape:
                return X_pending
            return X_pending.reshape(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        raise ValueError(f"Unexpected X_pending shape: {tuple(X_pending.shape)}")

    def _flatten_samples(self, target_samples: Tensor) -> Tensor:
        # target_samples: S x batch_shape x q x m in common case.
        if target_samples.ndim < 4:
            return target_samples.unsqueeze(0)
        # active base currently uses one sample dimension. This keeps future-proof support for multiple sample dims.
        q_dim = -2
        m_dim = -1
        prefix = target_samples.shape[:-2]
        if len(prefix) <= 1:
            return target_samples
        # Assume all leading dims before batch are sample dims only when explicit sampler has >1 sample dims.
        return target_samples

    def _sample_covariance_per_output(self, target_samples: Tensor) -> tuple[Tensor, Tensor]:
        # samples: S x batch_shape x q x m
        samples = self._flatten_samples(target_samples)
        mean = samples.mean(dim=0)
        centered = samples - mean.unsqueeze(0)
        sample_count = max(1, int(samples.shape[0]))
        batch_shape = mean.shape[:-2]
        q = int(mean.shape[-2])
        m = int(mean.shape[-1])
        # B x m x q x S
        flat = centered.reshape(sample_count, -1, q, m).permute(1, 3, 2, 0)
        cov = torch.matmul(flat, flat.transpose(-1, -2)) / max(sample_count - 1, 1)
        cov = cov.reshape(*batch_shape, m, q, q)
        eye = torch.eye(q, dtype=cov.dtype, device=cov.device)
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + self.jitter * eye
        return mean, cov

    def _joint_uncertainty(self, cov: Tensor) -> Tensor:
        q = int(cov.shape[-1])
        eye = torch.eye(q, dtype=cov.dtype, device=cov.device)
        if self.uncertainty_mode == "logdet1p":
            tau2 = max(self.tau**2, self.eps)
            sign, logabsdet = torch.linalg.slogdet(eye + cov / tau2)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite matrix encountered in logdet1p.")
            return 0.5 * logabsdet
        if self.uncertainty_mode == "logdet":
            sign, logabsdet = torch.linalg.slogdet(cov)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite covariance encountered in logdet.")
            return 0.5 * logabsdet
        if self.uncertainty_mode == "sqrt_trace":
            return torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(self.eps).sqrt()
        if self.uncertainty_mode == "trace":
            return torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
        raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode!r}.")

    def _boundary_distance(self, mean: Tensor) -> Tensor:
        # mean: batch_shape x q x m -> batch_shape x m
        diff = mean - self.threshold
        if self.boundary_mode == "mean_abs":
            return diff.abs().mean(dim=-2)
        if self.boundary_mode == "l2_mean":
            return diff.pow(2).mean(dim=-2).sqrt()
        if self.boundary_mode == "max_abs":
            return diff.abs().max(dim=-2).values
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    def _joint_score_per_output(self, X: Tensor) -> Tensor:
        Xq = self._ensure_q_batch(X)
        samples = self._target_prob_samples_per_output(Xq)
        mean, cov = self._sample_covariance_per_output(samples)
        uncertainty = self._joint_uncertainty(cov)  # batch_shape x m
        boundary = self._boundary_distance(mean)   # batch_shape x m
        return self.beta * uncertainty - boundary

    def _repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(self._apply_input_transform(self._ensure_q_batch(X)))
        penalty = self._same_batch_penalty(Xt)
        penalty = penalty + self._pending_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._observed_penalty_per_point(Xt).sum(dim=-1)
        return penalty

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        batch_shape = raw_X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach().to(device=raw_X.device, dtype=raw_X.dtype)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            score_per_output = self._joint_score_per_output(raw_X)
            value = self._aggregate_outputs(score_per_output)
            value = value - self._repulsion_penalty(raw_X)
            value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
            return self._finalize(value, raw_X, name=self.__class__.__name__)

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        pending_score = self._joint_score_per_output(Xp_batch)
        all_score = self._joint_score_per_output(torch.cat([Xp_batch, raw_X], dim=-2))
        score_per_output = all_score - pending_score
        value = self._aggregate_outputs(score_per_output)
        value = value - self._repulsion_penalty(raw_X)
        value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassICUAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Integrated contour uncertainty style acquisition for multi-output multiclass."""

    def __init__(self, model, *, bandwidth: float = 0.10, uncertainty_mode: UncertaintyMode = "bernoulli", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)
        self.uncertainty_mode = uncertainty_mode

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score_per_output = uncertainty.pow(2) * contour_weight
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qMultiOutputMulticlassBoundaryVarianceAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Boundary-weighted target-class variance acquisition."""

    def __init__(self, model, *, bandwidth: float = 0.15, uncertainty_mode: UncertaintyMode = "bernoulli", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)
        self.uncertainty_mode = uncertainty_mode

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score_per_output = uncertainty.pow(2) * _boundary_weight(p, self.threshold, bandwidth=self.bandwidth, eps=self.eps)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qMultiOutputMulticlassClassEntropyAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Per-output class entropy aggregated over outputs."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        if self.target_class is None and self.output_target_classes is None:
            score_per_output = _class_entropy(probs, eps=self.eps)
        else:
            selected = self._target_prob_per_output(probs)
            score_per_output = -(selected.clamp_min(self.eps) * selected.clamp_min(self.eps).log())
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qMultiOutputMulticlassProbabilityOfExceedance(_MultiOutputMulticlassTargetProbabilityBase):
    """Smooth probability-space exceedance score for target-class probability."""

    def __init__(self, model, *, tau: float = 0.02, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        score_per_output = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qMultiOutputMulticlassLevelSetUncertainty(qMultiOutputMulticlassICUAcquisition):
    """Alias for multi-output level-set uncertainty around target-class threshold."""

    pass


__all__ = [
    "UncertaintyMode",
    "JointUncertaintyMode",
    "JointBoundaryMode",
    "OutputReductionType",
    "OutputModeType",
    "_MultiOutputMulticlassTargetProbabilityBase",
    "qMultiOutputMulticlassLatentStraddleAcquisition",
    "qMultiOutputMulticlassJointLatentStraddleAcquisition",
    "qMultiOutputMulticlassICUAcquisition",
    "qMultiOutputMulticlassBoundaryVarianceAcquisition",
    "qMultiOutputMulticlassClassEntropyAcquisition",
    "qMultiOutputMulticlassProbabilityOfExceedance",
    "qMultiOutputMulticlassLevelSetUncertainty",
]
