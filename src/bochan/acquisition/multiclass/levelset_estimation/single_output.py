from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ClassReductionType, ReductionType
from bochan.acquisition.multiclass.bayesian_optimization.single_output import (
    _MulticlassProbabilityBOBase,
    _finalize_multiclass_acq_output_to_batch,
    _mean_over_sample_dims,
    _select_class_probs,
    _std_over_sample_dims,
    ensure_q_batch,
)

UncertaintyMode = Literal["bernoulli", "posterior", "combined"]
JointUncertaintyMode = Literal["logdet1p", "logdet", "sqrt_trace", "trace"]
JointBoundaryMode = Literal["mean_abs", "l2_mean", "max_abs"]


def _align_pointwise_to_reference(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    """Align a pointwise tensor to a reference tensor.

    This is mainly used when an input transform such as InputPerturbation expands
    q into q*n_w and a score / penalty needs to be broadcast or reduced back to
    the current pointwise score shape.
    """

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


class _MulticlassTargetProbabilityLevelSetBase(_MulticlassProbabilityBOBase):
    """Complete base for multiclass target-probability level-set acquisitions.

    Unlike the previous lightweight implementation, this base reuses the
    complete multiclass probability BO base so that latent logits,
    probability-posterior models, input transforms, pending / observed /
    same-batch penalties, and score objectives are handled consistently.
    """

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        threshold: float = 0.5,
        class_reduction: ClassReductionType = "mean",
        reduction: ReductionType = "mean",
        sampler: Optional[MCSampler] = None,
        apply_softmax_if_needed: bool = True,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        eps: float = 1e-8,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            target_class=target_class,
            class_reduction=class_reduction,
            apply_softmax_if_needed=apply_softmax_if_needed,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_observed=X_observed,
            eps=eps,
            objective=None,
        )
        self.threshold = float(threshold)
        self.levelset_objective = objective

    def _target_prob(self, X: Tensor) -> Tensor:
        return self._target_prob_mean(X)

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

    def _score_to_value(self, score: Tensor, raw_X: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        pending = _align_pointwise_to_reference(self._pending_penalty_per_point(Xt), score, name=f"{name}.pending_penalty")
        observed = _align_pointwise_to_reference(self._observed_penalty_per_point(Xt), score, name=f"{name}.observed_penalty")
        score = score - pending - observed
        score = self._apply_levelset_objective(score, raw_X, name=name)
        if score.shape == raw_X.shape[:-2]:
            value = score
        else:
            value = self._reduce_q(score)
        value = value - self._same_batch_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=name)

    def _target_uncertainty(self, X: Tensor, p: Tensor, *, mode: UncertaintyMode) -> Tensor:
        if mode == "bernoulli":
            return (p * (1.0 - p)).clamp_min(self.eps).sqrt()
        samples = self._target_prob_samples(X)
        posterior_std = _std_over_sample_dims(samples, self.sampler, eps=self.eps)
        if mode == "posterior":
            return posterior_std
        if mode == "combined":
            bernoulli_var = (p * (1.0 - p)).clamp_min(0.0)
            return (posterior_std.pow(2) + bernoulli_var).clamp_min(self.eps).sqrt()
        raise ValueError(f"Unknown uncertainty_mode: {mode!r}.")


class qMulticlassLatentStraddleAcquisition(_MulticlassTargetProbabilityLevelSetBase):
    """Target-class probability straddle acquisition.

    Scores points close to ``p(target_class | x) = threshold`` and with high
    uncertainty. ``uncertainty_mode`` controls whether uncertainty is based on
    Bernoulli variance, posterior sample variance, or both.
    """

    def __init__(
        self,
        model,
        *,
        beta: float = 1.0,
        uncertainty_mode: UncertaintyMode = "combined",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = float(beta)
        self.uncertainty_mode = uncertainty_mode

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score = self.beta * uncertainty - (p - self.threshold).abs()
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassJointLatentStraddleAcquisition(_MulticlassTargetProbabilityLevelSetBase):
    """Joint q-batch target-probability straddle acquisition.

    The joint uncertainty term is estimated from Monte Carlo target-probability
    samples across the q-batch. This is the multiclass-probability analogue of
    the binary joint straddle implementation.
    """

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

    def _flatten_samples(self, target_samples: Tensor, X: Tensor) -> Tensor:
        # target_samples: sample_shape x batch_shape x q
        sample_ndim = len(getattr(self.sampler, "sample_shape", torch.Size([1])))
        if sample_ndim <= 0:
            return target_samples.unsqueeze(0)
        if sample_ndim == 1:
            return target_samples
        sample_shape = target_samples.shape[:sample_ndim]
        batch_q_shape = target_samples.shape[sample_ndim:]
        return target_samples.reshape(int(torch.tensor(sample_shape).prod().item()), *batch_q_shape)

    def _sample_covariance(self, target_samples: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
        samples = self._flatten_samples(target_samples, X)
        mean = samples.mean(dim=0)
        centered = samples - mean.unsqueeze(0)
        sample_count = max(1, int(samples.shape[0]))
        batch_shape = mean.shape[:-1]
        q = int(mean.shape[-1])
        flat = centered.reshape(sample_count, -1, q).permute(1, 2, 0)
        cov = torch.matmul(flat, flat.transpose(-1, -2)) / max(sample_count - 1, 1)
        cov = cov.reshape(*batch_shape, q, q)
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
        diff = mean - self.threshold
        if self.boundary_mode == "mean_abs":
            return diff.abs().mean(dim=-1)
        if self.boundary_mode == "l2_mean":
            return diff.pow(2).mean(dim=-1).sqrt()
        if self.boundary_mode == "max_abs":
            return diff.abs().max(dim=-1).values
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    def _joint_score(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        samples = self._target_prob_samples(Xq)
        mean, cov = self._sample_covariance(samples, Xq)
        return self.beta * self._joint_uncertainty(cov) - self._boundary_distance(mean)

    def _repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = self._apply_input_transform(ensure_q_batch(X))
        penalty = self._same_batch_penalty(Xt)
        penalty = penalty + self._pending_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._observed_penalty_per_point(Xt).sum(dim=-1)
        return penalty

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        batch_shape = raw_X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach().to(device=raw_X.device, dtype=raw_X.dtype)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            value = self._joint_score(raw_X)
            value = value - self._repulsion_penalty(raw_X)
            value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
            return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        pending_score = self._joint_score(Xp_batch)
        all_score = self._joint_score(torch.cat([Xp_batch, raw_X], dim=-2))
        value = all_score - pending_score
        value = value - self._repulsion_penalty(raw_X)
        value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassICUAcquisition(_MulticlassTargetProbabilityLevelSetBase):
    """Integrated contour uncertainty style acquisition for target-class probability."""

    def __init__(self, model, *, bandwidth: float = 0.10, uncertainty_mode: UncertaintyMode = "bernoulli", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)
        self.uncertainty_mode = uncertainty_mode

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score = uncertainty.pow(2) * contour_weight
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassBoundaryVarianceAcquisition(_MulticlassTargetProbabilityLevelSetBase):
    """Boundary-weighted target-class variance acquisition."""

    def __init__(self, model, *, bandwidth: float = 0.15, uncertainty_mode: UncertaintyMode = "bernoulli", **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)
        self.uncertainty_mode = uncertainty_mode

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score = uncertainty.pow(2) * _boundary_weight(p, self.threshold, bandwidth=self.bandwidth, eps=self.eps)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassClassEntropyAcquisition(_MulticlassTargetProbabilityLevelSetBase):
    """Class entropy acquisition for multiclass boundary exploration."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._posterior_mean_probs(raw_X)
        if self.target_class is None:
            score = _class_entropy(probs, eps=self.eps)
        else:
            selected = _select_class_probs(
                probs,
                target_class=self.target_class,
                class_reduction=self.class_reduction,
            )
            score = -(selected.clamp_min(self.eps) * selected.clamp_min(self.eps).log())
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassProbabilityOfExceedance(_MulticlassTargetProbabilityLevelSetBase):
    """Smooth probability-space exceedance score for target-class probability."""

    def __init__(self, model, *, tau: float = 0.02, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        score = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassLevelSetUncertainty(qMulticlassICUAcquisition):
    """Alias for level-set uncertainty around target-class threshold."""

    pass


__all__ = [
    "UncertaintyMode",
    "JointUncertaintyMode",
    "JointBoundaryMode",
    "_MulticlassTargetProbabilityLevelSetBase",
    "qMulticlassLatentStraddleAcquisition",
    "qMulticlassJointLatentStraddleAcquisition",
    "qMulticlassICUAcquisition",
    "qMulticlassBoundaryVarianceAcquisition",
    "qMulticlassClassEntropyAcquisition",
    "qMulticlassProbabilityOfExceedance",
    "qMulticlassLevelSetUncertainty",
]
