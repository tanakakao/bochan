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

    def _align_score_per_output_to_raw(self, score: Tensor, raw_X: Tensor, *, name: str) -> Tensor:
        """Align q_like / perturbation-expanded per-output score to raw_X q.

        With InputPerturbation, model probabilities may be evaluated on q_like = q * n_w
        points while optimize_acqf still expects acquisition values for raw q.
        The inherited helper averages q_like / q perturbations and returns
        ``batch_shape x q x m``.
        """
        return self._align_score_per_output_to_raw_X(score, raw_X, name=name)

    def _target_uncertainty(self, X: Tensor, p: Tensor, *, mode: UncertaintyMode) -> Tensor:
        p = self._align_score_per_output_to_raw(p, X, name=f"{self.__class__.__name__}.target_prob")
        if mode == "bernoulli":
            return (p * (1.0 - p)).clamp_min(self.eps).sqrt()

        samples = self._target_prob_samples_per_output(X)
        posterior_std = samples.std(dim=0, unbiased=False).clamp_min(self.eps)
        posterior_std = self._align_score_per_output_to_raw(
            posterior_std,
            X,
            name=f"{self.__class__.__name__}.posterior_std",
        )

        if mode == "posterior":
            return posterior_std
        if mode == "combined":
            bernoulli_var = (p * (1.0 - p)).clamp_min(0.0)
            posterior_std = _align_pointwise_to_reference(
                posterior_std,
                bernoulli_var,
                name=f"{self.__class__.__name__}.posterior_std_combined",
            )
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
        score_per_output = self._align_score_per_output_to_raw(
            score_per_output,
            raw_X,
            name=f"{name}.score_per_output",
        )
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
        p = self._align_score_per_output_to_raw(p, raw_X, name=f"{self.__class__.__name__}.target_prob")
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
            pending_penalty_weight=pending_penalty_weight,
            observed_penalty_weight=observed_penalty_weight,
            same_batch_penalty_weight=same_batch_penalty_weight,
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
        self.sampler = sampler

    def _sample_target_probs(self, X: Tensor) -> Tensor:
        posterior = self._get_multiclass_probability_posterior(X)
        return self._target_prob_per_output(posterior.rsample(self.sampler.sample_shape))

    def _joint_uncertainty(self, samples: Tensor, *, q: int) -> Tensor:
        # samples: S x batch x q_like x m
        s = samples.shape[0]
        if samples.shape[-2] != q:
            samples = samples.reshape(s, *samples.shape[1:-2], q, samples.shape[-2] // q, samples.shape[-1]).mean(dim=-2)

    def _joint_uncertainty(self, samples: Tensor, *, q: int) -> Tensor:
        # samples: S x batch x q_like x m
        s = samples.shape[0]
        if samples.shape[-2] != q:
            samples = samples.reshape(s, *samples.shape[1:-2], q, samples.shape[-2] // q, samples.shape[-1]).mean(dim=-2)

        y = samples.permute(*range(1, samples.ndim - 2), -2, -1, 0)  # batch x q x m x S
        y = y.reshape(*y.shape[:-2], y.shape[-2] * y.shape[-1])  # batch x q x (m*S)
        y = y - y.mean(dim=-2, keepdim=True)
        cov = torch.matmul(y, y.transpose(-1, -2)) / max(y.shape[-1] - 1, 1)
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        cov = cov + self.jitter * eye
