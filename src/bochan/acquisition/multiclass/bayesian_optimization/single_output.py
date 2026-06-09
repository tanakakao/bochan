from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ClassReductionType, ReductionType

QFeasMode = Literal["prod", "mean", "min", "max"]


def ensure_q_batch(X: Tensor) -> Tensor:
    """Normalize input tensor to ``batch_shape x q x d``.

    - ``d`` -> ``1 x 1 x d``
    - ``n x d`` -> ``1 x n x d``
    - ``batch_shape x q x d`` -> unchanged
    """

    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _finalize_multiclass_acq_output_to_batch(value: Tensor, X: Tensor, *, name: str) -> Tensor:
    """Align acquisition output to BoTorch t-batch shape."""

    Xq = ensure_q_batch(X)
    target = tuple(Xq.shape[:-2])
    out = value

    if out.shape == target:
        return out
    if len(target) == 0:
        return out.mean() if out.ndim > 0 else out
    if out.ndim == 0:
        return out.expand(*target)

    while out.ndim > len(target):
        out = out.mean(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out
    if out.numel() == _prod(target):
        return out.reshape(target)
    if out.ndim == 1 and len(target) == 1:
        if out.shape[0] == target[0]:
            return out
        return out.mean().expand(*target)

    raise RuntimeError(
        f"{name}: could not align acquisition output to t-batch shape. "
        f"value.shape={tuple(value.shape)}, target={target}, X.shape={tuple(Xq.shape)}."
    )


def _model_num_classes(model: Model) -> int | None:
    for attr in ("num_classes", "num_outputs"):
        value = getattr(model, attr, None)
        if value is not None:
            try:
                value_int = int(value)
            except (TypeError, ValueError):
                continue
            if value_int > 1:
                return value_int
    return None


def _move_class_dim_to_last(values: Tensor, *, num_classes: int | None, name: str) -> Tensor:
    """Move a class-batch dimension to the final output dimension.

    Multiclass latent GPs are represented as class-wise batched GPs, so the raw
    latent posterior often has shape like ``batch x C x q x 1`` instead of
    ``batch x q x C``. Acquisition code expects class probabilities in the final
    dimension. This helper canonicalizes both mean and MC sample tensors.
    """

    if num_classes is None:
        return values
    c = int(num_classes)
    if values.ndim < 1:
        raise RuntimeError(f"{name}: expected tensor with class dimension. Got scalar tensor.")

    out = values
    while out.ndim >= 2 and out.shape[-1] == 1 and out.shape[-2] != c:
        out = out.squeeze(-1)

    if out.shape[-1] == c:
        return out

    for dim in range(out.ndim - 2, -1, -1):
        if out.shape[dim] == c:
            return out.movedim(dim, -1)

    return values


def _align_class_probs_to_X(
    probs: Tensor,
    X: Tensor,
    *,
    num_classes: int | None,
    sample_ndim: int = 0,
    name: str,
) -> Tensor:
    """Canonicalize class-probability shape to sample_shape x batch_shape x q x C."""

    Xq = ensure_q_batch(X)
    out = _move_class_dim_to_last(probs, num_classes=num_classes, name=name)
    target_ndim = int(sample_ndim) + Xq.ndim
    while out.ndim > target_ndim and out.shape[-2] == 1:
        out = out.squeeze(-2)
    return out


def _reduce_extra_leading_dims_to_raw_X(
    value: Tensor,
    raw_X: Tensor,
    *,
    sample_ndim: int = 0,
    name: str,
) -> Tensor:
    """Reduce sample-like leading dims so a score/value matches raw_X.

    Multiclass DeepGP posteriors can leave an additional leading sample-like
    dimension in target-probability tensors. This can appear either before the
    t-batch/q axes, e.g. ``S x 10 x batch x q``, or after sampler dimensions have
    already been reduced, e.g. ``10 x batch``. This helper preserves explicit
    sampler dimensions and averages only the extra dimensions between sampler
    dimensions and the raw-X t-batch/q axes.
    """

    raw_X = ensure_q_batch(raw_X)
    sample_ndim = int(sample_ndim)
    if sample_ndim < 0:
        raise ValueError(f"sample_ndim must be non-negative. Got {sample_ndim}.")
    if sample_ndim > value.ndim:
        return value

    batch_shape = tuple(raw_X.shape[:-2])
    q = int(raw_X.shape[-2])

    if tuple(value.shape[sample_ndim:]) == batch_shape:
        return value

    if (
        value.ndim >= sample_ndim + 1
        and tuple(value.shape[sample_ndim:-1]) == batch_shape
        and value.shape[-1] == q
    ):
        return value

    reduce_start = sample_ndim

    if len(batch_shape) > 0:
        batch_ndim = len(batch_shape)
        if value.ndim >= sample_ndim + batch_ndim and tuple(value.shape[-batch_ndim:]) == batch_shape:
            extra_ndim = value.ndim - sample_ndim - batch_ndim
            if extra_ndim > 0:
                return value.mean(dim=tuple(range(reduce_start, reduce_start + extra_ndim)))
            return value

        target_with_q = batch_shape + (q,)
        target_ndim = len(target_with_q)
        if value.ndim >= sample_ndim + target_ndim and tuple(value.shape[-target_ndim:]) == target_with_q:
            extra_ndim = value.ndim - sample_ndim - target_ndim
            if extra_ndim > 0:
                return value.mean(dim=tuple(range(reduce_start, reduce_start + extra_ndim)))
            return value

    elif value.ndim >= sample_ndim + 2 and value.shape[-1] == q:
        extra_ndim = value.ndim - sample_ndim - 1
        if extra_ndim > 0:
            return value.mean(dim=tuple(range(reduce_start, reduce_start + extra_ndim)))

    return value


def _normalize_class_probs(probs: Tensor, *, eps: float, name: str) -> Tensor:
    if probs.ndim < 1 or probs.shape[-1] <= 1:
        raise RuntimeError(f"{name}: multiclass probability tensor must have class dim C >= 2. Got {tuple(probs.shape)}.")
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _select_class_probs(
    probs: Tensor,
    *,
    target_class: int | Sequence[int] | None,
    class_reduction: ClassReductionType = "mean",
) -> Tensor:
    if target_class is None:
        return probs.max(dim=-1).values
    if isinstance(target_class, int):
        return probs[..., int(target_class)]
    indices = [int(i) for i in target_class]
    selected = probs[..., indices]
    if class_reduction == "mean":
        return selected.mean(dim=-1)
    if class_reduction == "sum":
        return selected.sum(dim=-1)
    if class_reduction == "max":
        return selected.max(dim=-1).values
    if class_reduction == "min":
        return selected.min(dim=-1).values
    if class_reduction == "prod":
        return selected.prod(dim=-1)
    raise ValueError(f"Unknown class_reduction: {class_reduction!r}.")


def _sample_ndim_from_sampler(sampler: Optional[MCSampler]) -> int:
    sample_shape = getattr(sampler, "sample_shape", torch.Size([1]))
    return len(sample_shape)


def _mean_over_sample_dims(values: Tensor, sampler: Optional[MCSampler]) -> Tensor:
    sample_ndim = _sample_ndim_from_sampler(sampler)
    if sample_ndim <= 0:
        return values
    return values.mean(dim=tuple(range(sample_ndim)))


def _std_over_sample_dims(values: Tensor, sampler: Optional[MCSampler], *, eps: float) -> Tensor:
    sample_ndim = _sample_ndim_from_sampler(sampler)
    if sample_ndim <= 0:
        return torch.zeros_like(values)
    return values.std(dim=tuple(range(sample_ndim)), unbiased=False).clamp_min(eps)


def _coerce_reference_tensor(X_ref, *, ref: Tensor | None = None) -> Tensor | None:
    if X_ref is None:
        return None
    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        tensors = [_coerce_reference_tensor(x, ref=ref) for x in X_ref if x is not None]
        tensors = [x for x in tensors if x is not None and x.numel() > 0]
        if len(tensors) == 0:
            return None
        if len(tensors) == 1:
            out = tensors[0]
        else:
            out = torch.cat([x.reshape(-1, x.shape[-1]) for x in tensors], dim=-2)
    else:
        raise TypeError(f"Reference points must be Tensor, sequence of Tensors, or None. Got {type(X_ref)}.")
    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


def _resolve_observed_X(model: Model, X_observed: Tensor | None = None) -> Tensor | None:
    if X_observed is not None:
        return X_observed
    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x[0] if isinstance(x, tuple) else x
    x = getattr(model, "train_inputs", None)
    if isinstance(x, tuple) and len(x) > 0:
        return x[0]
    return None


def compute_multiclass_target_probability_values(
    model: Model,
    X: Tensor,
    *,
    target_class: int | Sequence[int] | None,
    class_reduction: ClassReductionType = "mean",
    apply_softmax_if_needed: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute target-class probabilities on ``X``.

    This helper mirrors the acquisition behavior and is intended for computing
    ``best_f`` from observed / baseline points.
    """

    if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
        model = model.models[0]
    model.eval()
    Xq = ensure_q_batch(X)
    num_classes = _model_num_classes(model)
    with torch.no_grad():
        if apply_softmax_if_needed and hasattr(model, "latent_posterior"):
            logits = model.latent_posterior(Xq).mean
            logits = _align_class_probs_to_X(logits, Xq, num_classes=num_classes, name="latent posterior mean")
            probs = torch.softmax(logits, dim=-1)
        elif hasattr(model, "class_probs") and callable(getattr(model, "class_probs")):
            probs = model.class_probs(Xq)
            probs = _align_class_probs_to_X(probs, Xq, num_classes=num_classes, name="class_probs")
        elif hasattr(model, "probability_posterior") and callable(getattr(model, "probability_posterior")):
            probs = model.probability_posterior(Xq).mean
            probs = _align_class_probs_to_X(probs, Xq, num_classes=num_classes, name="probability_posterior.mean")
        else:
            mean = model.posterior(Xq).mean
            mean = _align_class_probs_to_X(mean, Xq, num_classes=num_classes, name="posterior.mean")
            if apply_softmax_if_needed and (mean.min() < -eps or mean.max() > 1.0 + eps):
                probs = torch.softmax(mean, dim=-1)
            else:
                probs = _normalize_class_probs(mean, eps=eps, name="posterior.mean")
        probs = _normalize_class_probs(probs, eps=eps, name="class probabilities")
        values = _select_class_probs(probs, target_class=target_class, class_reduction=class_reduction)
        values = _reduce_extra_leading_dims_to_raw_X(
            values,
            Xq,
            sample_ndim=0,
            name="compute_multiclass_target_probability_values",
        )
    return values.detach().reshape(-1)


def compute_multiclass_target_probability_best_f(
    model: Model,
    train_X: Tensor,
    *,
    target_class: int | Sequence[int] | None,
    class_reduction: ClassReductionType = "mean",
    apply_softmax_if_needed: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute best target-class probability on observed points."""

    values = compute_multiclass_target_probability_values(
        model=model,
        X=train_X,
        target_class=target_class,
        class_reduction=class_reduction,
        apply_softmax_if_needed=apply_softmax_if_needed,
        eps=eps,
    )
    return values.max().detach()


class _MulticlassProbabilityBOBase(MCAcquisitionFunction):
    """Base class for multiclass single-output BO acquisitions.

    The base accepts models whose posterior is already class-probability based,
    as well as models exposing ``latent_posterior``. In the latter case,
    ``apply_softmax_if_needed=True`` samples latent logits and maps them to class
    probabilities with softmax.
    """

    def __init__(
        self,
        model: Model,
        *,
        sampler: Optional[MCSampler] = None,
        target_class: int | Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        apply_softmax_if_needed: bool = True,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        eps: float = 1e-8,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        **kwargs,
    ) -> None:
        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
            model = model.models[0]
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        super().__init__(
            model=model,
            sampler=sampler,
            objective=IdentityMCObjective(),
            **kwargs,
        )
        self.target_class = target_class
        self.class_reduction = class_reduction
        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.eps = float(eps)
        self.score_objective = objective
        self.num_classes = _model_num_classes(model)
        self.X_observed = _resolve_observed_X(model, X_observed)
        self.set_X_pending(None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None:
            likelihood.eval()

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = ensure_q_batch(X)
        input_transform = getattr(self.model, "input_transform", None)
        if input_transform is not None:
            Xt = input_transform(X)
            return ensure_q_batch(Xt[0] if isinstance(Xt, tuple) else Xt)
        return X

    def _align_class_tensor(self, values: Tensor, X: Tensor, *, sample_ndim: int = 0, name: str) -> Tensor:
        return _align_class_probs_to_X(
            values,
            X,
            num_classes=self.num_classes,
            sample_ndim=sample_ndim,
            name=name,
        )

    def _posterior_samples_as_probs(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        sample_ndim = _sample_ndim_from_sampler(self.sampler)
        if self.apply_softmax_if_needed and hasattr(self.model, "latent_posterior"):
            post = self.model.latent_posterior(Xq)
            logits = self.get_posterior_samples(post)
            logits = self._align_class_tensor(logits, Xq, sample_ndim=sample_ndim, name="latent posterior samples")
            probs = torch.softmax(logits, dim=-1)
        elif hasattr(self.model, "probability_posterior") and callable(getattr(self.model, "probability_posterior")):
            post = self.model.probability_posterior(Xq)
            probs = self.get_posterior_samples(post)
            probs = self._align_class_tensor(probs, Xq, sample_ndim=sample_ndim, name="probability posterior samples")
        elif hasattr(self.model, "class_probs") and callable(getattr(self.model, "class_probs")):
            mean_probs = self.model.class_probs(Xq)
            mean_probs = self._align_class_tensor(mean_probs, Xq, name="class_probs")
            sample_shape = getattr(self.sampler, "sample_shape", torch.Size([1]))
            probs = mean_probs.expand(*sample_shape, *mean_probs.shape)
        else:
            post = self.model.posterior(Xq)
            samples = self.get_posterior_samples(post)
            samples = self._align_class_tensor(samples, Xq, sample_ndim=sample_ndim, name="posterior samples")
            if self.apply_softmax_if_needed and (samples.min() < -self.eps or samples.max() > 1.0 + self.eps):
                probs = torch.softmax(samples, dim=-1)
            else:
                probs = samples
        return _normalize_class_probs(probs, eps=self.eps, name="posterior samples")

    def _posterior_mean_probs(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        if self.apply_softmax_if_needed and hasattr(self.model, "latent_posterior"):
            logits = self.model.latent_posterior(Xq).mean
            logits = self._align_class_tensor(logits, Xq, name="latent posterior mean")
            probs = torch.softmax(logits, dim=-1)
        elif hasattr(self.model, "class_probs") and callable(getattr(self.model, "class_probs")):
            probs = self.model.class_probs(Xq)
            probs = self._align_class_tensor(probs, Xq, name="class_probs")
        elif hasattr(self.model, "probability_posterior") and callable(getattr(self.model, "probability_posterior")):
            probs = self.model.probability_posterior(Xq).mean
            probs = self._align_class_tensor(probs, Xq, name="probability_posterior.mean")
        else:
            mean = self.model.posterior(Xq).mean
            mean = self._align_class_tensor(mean, Xq, name="posterior.mean")
            if self.apply_softmax_if_needed and (mean.min() < -self.eps or mean.max() > 1.0 + self.eps):
                probs = torch.softmax(mean, dim=-1)
            else:
                probs = mean
        return _normalize_class_probs(probs, eps=self.eps, name="posterior mean")

    def _target_prob_samples(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        sample_ndim = _sample_ndim_from_sampler(self.sampler)
        probs = self._posterior_samples_as_probs(Xq)
        values = _select_class_probs(
            probs,
            target_class=self.target_class,
            class_reduction=self.class_reduction,
        )
        values = _reduce_extra_leading_dims_to_raw_X(
            values,
            Xq,
            sample_ndim=sample_ndim,
            name="target probability samples",
        )
        if self.score_objective is not None:
            try:
                values = self.score_objective(values, X=Xq)
            except TypeError:
                values = self.score_objective(values)
            values = _reduce_extra_leading_dims_to_raw_X(
                values,
                Xq,
                sample_ndim=sample_ndim,
                name="target probability samples objective",
            )
        return values

    def _target_prob_mean(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        probs = self._posterior_mean_probs(Xq)
        values = _select_class_probs(
            probs,
            target_class=self.target_class,
            class_reduction=self.class_reduction,
        )
        values = _reduce_extra_leading_dims_to_raw_X(
            values,
            Xq,
            sample_ndim=0,
            name="target probability mean",
        )
        if self.score_objective is not None:
            try:
                values = self.score_objective(values, X=Xq)
            except TypeError:
                values = self.score_objective(values)
            values = _reduce_extra_leading_dims_to_raw_X(
                values,
                Xq,
                sample_ndim=0,
                name="target probability mean objective",
            )
        return values

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown reduction: {self.reduction!r}.")

    def _reference_points_transformed(self, X_ref, *, ref: Tensor) -> Tensor | None:
        X_ref = _coerce_reference_tensor(X_ref, ref=ref)
        if X_ref is None or X_ref.numel() == 0:
            return None
        return self._apply_input_transform(X_ref).reshape(-1, ref.shape[-1]).to(ref)

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if self.pending_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-1])
        Xp = self._reference_points_transformed(getattr(self, "X_pending", None), ref=Xt)
        if Xp is None:
            return Xt.new_zeros(Xt.shape[:-1])
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xp).min(dim=-1).values
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))

    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if self.observed_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-1])
        Xobs = self._reference_points_transformed(self.X_observed, ref=Xt)
        if Xobs is None:
            return Xt.new_zeros(Xt.shape[:-1])
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xobs).min(dim=-1).values
        return self.observed_penalty_weight * torch.exp(-self.observed_penalty_beta * dist.reshape(Xt.shape[:-1]))

    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if self.same_batch_penalty_weight <= 0 or Xt.shape[-2] <= 1:
            return Xt.new_zeros(Xt.shape[:-2])
        batch_shape = Xt.shape[:-2]
        q = Xt.shape[-2]
        Xb = Xt.reshape(-1, q, Xt.shape[-1])
        d = torch.cdist(Xb, Xb)
        eye = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, float("inf"))
        penalty = 0.5 * self.same_batch_penalty_weight * torch.exp(-self.same_batch_penalty_beta * d).sum(dim=(-1, -2))
        return penalty.reshape(*batch_shape)

    def _pointwise_score_to_value(self, score: Tensor, raw_X: Tensor, Xt: Tensor) -> Tensor:
        score = _reduce_extra_leading_dims_to_raw_X(score, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.score")
        score = score - self._pending_penalty_per_point(Xt)
        score = score - self._observed_penalty_per_point(Xt)
        value = self._reduce_q(score)
        value = _reduce_extra_leading_dims_to_raw_X(value, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.value")
        value = value - self._same_batch_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)

    def _q_penalty(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        penalty = self._pending_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._observed_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._same_batch_penalty(Xt)
        return penalty


class qMulticlassProbabilityOfFeasibility(_MulticlassProbabilityBOBase):
    """Probability of target-class feasibility.

    If ``threshold`` is ``None``, this maximizes ``p(target_class | x)``. If a
    threshold is provided, it maximizes a smooth exceedance indicator.
    """

    def __init__(
        self,
        model: Model,
        *,
        target_class: int | Sequence[int] | None,
        threshold: float | None = None,
        tau: float = 0.02,
        q_feas_mode: QFeasMode | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.threshold = None if threshold is None else float(threshold)
        self.tau = float(tau)
        self.q_feas_mode = q_feas_mode

    def _reduce_q_feas(self, score: Tensor) -> Tensor:
        mode = self.q_feas_mode
        if mode is None:
            return self._reduce_q(score)
        if mode == "prod":
            return score.prod(dim=-1)
        if mode == "mean":
            return score.mean(dim=-1)
        if mode == "min":
            return score.min(dim=-1).values
        if mode == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown q_feas_mode: {mode!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob_mean(raw_X)
        score = p if self.threshold is None else torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score = _reduce_extra_leading_dims_to_raw_X(score, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.score")
        score = score - self._pending_penalty_per_point(Xt)
        score = score - self._observed_penalty_per_point(Xt)
        value = self._reduce_q_feas(score)
        value = _reduce_extra_leading_dims_to_raw_X(value, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.value")
        value = value - self._same_batch_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassExpectedImprovement(_MulticlassProbabilityBOBase):
    """Expected improvement for target-class probability."""

    def __init__(self, model: Model, *, target_class: int | Sequence[int] | None, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        target_samples = self._target_prob_samples(raw_X)
        best_q = target_samples.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        value = (best_q - best_f).clamp_min(0.0)
        value = _mean_over_sample_dims(value, self.sampler)
        value = _reduce_extra_leading_dims_to_raw_X(value, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.value")
        value = value - self._q_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassProbabilityOfImprovement(_MulticlassProbabilityBOBase):
    """Probability of improvement for target-class probability."""

    def __init__(
        self,
        model: Model,
        *,
        target_class: int | Sequence[int] | None,
        best_f: float | Tensor,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        target_samples = self._target_prob_samples(raw_X)
        best_q = target_samples.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        tau = self.tau.to(best_q).clamp_min(self.eps)
        value = torch.sigmoid((best_q - best_f) / tau)
        value = _mean_over_sample_dims(value, self.sampler)
        value = _reduce_extra_leading_dims_to_raw_X(value, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.value")
        value = value - self._q_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassUpperConfidenceBound(_MulticlassProbabilityBOBase):
    """Upper confidence bound for target-class probability."""

    def __init__(self, model: Model, *, target_class: int | Sequence[int] | None, beta: float | Tensor = 2.0, **kwargs) -> None:
        kwargs.setdefault("reduction", "max")
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        target_samples = self._target_prob_samples(raw_X)
        mean = _mean_over_sample_dims(target_samples, self.sampler)
        std = _std_over_sample_dims(target_samples, self.sampler, eps=self.eps)
        mean = _reduce_extra_leading_dims_to_raw_X(mean, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.mean")
        std = _reduce_extra_leading_dims_to_raw_X(std, raw_X, sample_ndim=0, name=f"{self.__class__.__name__}.std")
        beta = self.beta.to(mean)
        score = mean + beta.sqrt() * std
        return self._pointwise_score_to_value(score, raw_X, Xt)


__all__ = [
    "QFeasMode",
    "ensure_q_batch",
    "compute_multiclass_target_probability_values",
    "compute_multiclass_target_probability_best_f",
    "_MulticlassProbabilityBOBase",
    "qMulticlassProbabilityOfFeasibility",
    "qMulticlassExpectedImprovement",
    "qMulticlassProbabilityOfImprovement",
    "qMulticlassUpperConfidenceBound",
]
