from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

ReductionType = Literal["mean", "sum", "max"]
OutputReductionType = Literal["mean", "sum", "max", "min", "weighted_mean"]
OutputModeType = OutputReductionType
LargeQStrategy = Literal["per_point", "truncate", "raise"]


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _align_pointwise_to_reference(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
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
        f"{name}: cannot align value to reference. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


class _TensorProbabilityPosterior:
    def __init__(self, probs: Tensor, *, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self._probs = self._normalize(probs)

    def _normalize(self, probs: Tensor) -> Tensor:
        if probs.shape[-1] <= 1:
            raise RuntimeError(f"Multiclass probabilities must have class dim C >= 2. Got {tuple(probs.shape)}.")
        probs = probs.clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    @property
    def mean(self) -> Tensor:
        return self._probs

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        return self._probs.expand(*sample_shape, *self._probs.shape)


class _MaybeProbabilityPosterior:
    def __init__(self, posterior, *, eps: float = 1e-8, apply_softmax_if_needed: bool = True) -> None:
        self.posterior = posterior
        self.eps = float(eps)
        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)

    def _to_probs(self, values: Tensor) -> Tensor:
        if values.shape[-1] <= 1:
            raise RuntimeError(f"Multiclass posterior must include class dim C >= 2. Got {tuple(values.shape)}.")
        if self.apply_softmax_if_needed and (values.min() < -self.eps or values.max() > 1.0 + self.eps):
            return torch.softmax(values, dim=-1)
        values = values.clamp_min(self.eps)
        return values / values.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    @property
    def mean(self) -> Tensor:
        return self._to_probs(self.posterior.mean)

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        return self._to_probs(self.posterior.rsample(sample_shape))


class _SoftmaxPosterior:
    def __init__(self, posterior) -> None:
        self.posterior = posterior

    @property
    def mean(self) -> Tensor:
        return torch.softmax(self.posterior.mean, dim=-1)

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        return torch.softmax(self.posterior.rsample(sample_shape), dim=-1)


class _StackedMulticlassPosterior:
    """Stack single-output multiclass posteriors into one multi-output posterior.

    mean: ``batch_shape x q x m x C``
    rsample: ``sample_shape x batch_shape x q x m x C``
    """

    def __init__(self, posteriors: Sequence, *, eps: float = 1e-8) -> None:
        if len(posteriors) == 0:
            raise ValueError("At least one posterior is required.")
        self.posteriors = list(posteriors)
        self.eps = float(eps)

    @property
    def mean(self) -> Tensor:
        return torch.stack([p.mean for p in self.posteriors], dim=-2)

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        return torch.stack([p.rsample(sample_shape) for p in self.posteriors], dim=-2)


class _DirectMultiOutputMulticlassAcqBase(AcquisitionFunction):
    """Complete direct-tensor base for multi-output multiclass active learning.

    The base obtains multiclass probability tensors with shape
    ``batch_shape x q_like x m x C`` and computes pointwise scores before output
    aggregation. It mirrors the complete single-output implementation by
    supporting latent logits, probability posteriors, class_probs, input
    transforms, pending / observed / same-batch penalties, and optional score
    objectives.
    """

    def __init__(
        self,
        model,
        *,
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
        apply_softmax_if_needed: bool = True,
        eps: float = 1e-8,
        objective=None,
    ) -> None:
        super().__init__(model=model)
        self.reduction = reduction
        self.output_mode = output_reduction or output_mode
        self.output_weights = None if output_weights is None else torch.as_tensor(output_weights, dtype=torch.double)
        self.normalize_output_weights = bool(normalize_output_weights)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()
        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)
        self.eps = float(eps)
        self.objective = objective
        self.set_X_pending(None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            return X.view(1, 1, -1)
        if X.ndim == 2:
            return X.unsqueeze(0)
        return X

    def _set_eval_mode(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None:
            likelihood.eval()

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            return self._ensure_q_batch(Xt[0] if isinstance(Xt, tuple) else Xt)
        submodels = getattr(self.model, "models", None) or getattr(self.model, "submodels", None)
        if submodels is not None and len(submodels) > 0:
            it = getattr(submodels[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                return self._ensure_q_batch(Xt[0] if isinstance(Xt, tuple) else Xt)
        return X

    def _submodels(self) -> list:
        if isinstance(self.model, (ModelListGP, ModelListGPyTorchModel)):
            return list(self.model.models)
        submodels = getattr(self.model, "models", None)
        if submodels is not None:
            return list(submodels)
        submodels = getattr(self.model, "submodels", None)
        if submodels is not None:
            return list(submodels)
        return []

    def _posterior_for_submodel(self, submodel, X: Tensor):
        class_probs = getattr(submodel, "class_probs", None)
        if callable(class_probs):
            return _TensorProbabilityPosterior(class_probs(X), eps=self.eps)
        prob_post = getattr(submodel, "probability_posterior", None)
        if callable(prob_post):
            return _MaybeProbabilityPosterior(prob_post(X), eps=self.eps, apply_softmax_if_needed=False)
        latent_post = getattr(submodel, "latent_posterior", None)
        if callable(latent_post):
            return _SoftmaxPosterior(latent_post(X))
        return _MaybeProbabilityPosterior(
            submodel.posterior(X),
            eps=self.eps,
            apply_softmax_if_needed=self.apply_softmax_if_needed,
        )

    def _get_multiclass_probability_posterior(self, X: Tensor):
        X = self._ensure_q_batch(X)
        fn = getattr(self.model, "class_probs_list", None)
        if callable(fn):
            probs_list = fn(X)
            return _StackedMulticlassPosterior(
                [_TensorProbabilityPosterior(p, eps=self.eps) for p in probs_list],
                eps=self.eps,
            )

        submodels = self._submodels()
        if len(submodels) > 0:
            return _StackedMulticlassPosterior(
                [self._posterior_for_submodel(submodel, X) for submodel in submodels],
                eps=self.eps,
            )

        prob_post = getattr(self.model, "probability_posterior", None)
        if callable(prob_post):
            return _MaybeProbabilityPosterior(prob_post(X), eps=self.eps, apply_softmax_if_needed=False)
        latent_post = getattr(self.model, "latent_posterior", None)
        if callable(latent_post):
            return _SoftmaxPosterior(latent_post(X))
        return _MaybeProbabilityPosterior(
            self.model.posterior(X),
            eps=self.eps,
            apply_softmax_if_needed=self.apply_softmax_if_needed,
        )

    def _mean_probs(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        probs = self._get_multiclass_probability_posterior(X).mean
        if probs.ndim == X.ndim:
            raise RuntimeError(
                "Multi-output multiclass active learning requires class probabilities with shape "
                "batch_shape x q x m x C. The model posterior returned scalar output probabilities."
            )
        return probs

    def _sample_probs(self, X: Tensor, *, num_samples: int) -> Tensor:
        X = self._ensure_q_batch(X)
        samples = self._get_multiclass_probability_posterior(X).rsample(torch.Size([int(num_samples)]))
        if samples.ndim == X.ndim + 1:
            raise RuntimeError(
                "Multi-output multiclass BALD requires probability samples with shape "
                "S x batch_shape x q x m x C. Got scalar output samples."
            )
        return samples

    def _entropy(self, probs: Tensor) -> Tensor:
        probs = probs.clamp_min(self.eps)
        return -(probs * probs.log()).sum(dim=-1)

    def _class_probability_variance(self, probs: Tensor) -> Tensor:
        return (probs * (1.0 - probs)).sum(dim=-1)

    def _margin_uncertainty(self, probs: Tensor) -> Tensor:
        top2 = probs.topk(k=2, dim=-1).values
        return 1.0 - (top2[..., 0] - top2[..., 1])

    def _aggregate_outputs(self, score_per_output: Tensor) -> Tensor:
        if self.output_mode == "mean":
            return score_per_output.mean(dim=-1)
        if self.output_mode == "sum":
            return score_per_output.sum(dim=-1)
        if self.output_mode == "max":
            return score_per_output.max(dim=-1).values
        if self.output_mode == "min":
            return score_per_output.min(dim=-1).values
        if self.output_mode == "weighted_mean":
            if self.output_weights is None:
                raise ValueError("output_weights must be provided when output_mode='weighted_mean'.")
            weights = self.output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype)
            if weights.ndim != 1 or weights.numel() != score_per_output.shape[-1]:
                raise ValueError(f"output_weights must have shape ({score_per_output.shape[-1]},), got {tuple(weights.shape)}.")
            if self.normalize_output_weights:
                weights = weights / weights.abs().sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (weights.numel(),)
            return (score_per_output * weights.view(*view_shape)).sum(dim=-1)
        raise ValueError(f"Unknown output_mode/output_reduction: {self.output_mode!r}.")

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
        Xt = self._apply_input_transform(X_ref)
        return self._ensure_q_batch(Xt).reshape(-1, ref.shape[-1]).to(ref)

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.pending_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-1])
        Xp = self._reference_points_transformed(getattr(self, "X_pending", None), ref=Xt)
        if Xp is None:
            return Xt.new_zeros(Xt.shape[:-1])
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xp).min(dim=-1).values
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))

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

    def _apply_objective(self, score: Tensor, *, raw_X: Tensor, expanded_X: Tensor) -> Tensor:
        if self.objective is None:
            return score
        try:
            out = self.objective(score, X=raw_X)
        except TypeError:
            out = self.objective(score)
        if not torch.is_tensor(out):
            raise TypeError(f"objective must return a Tensor. Got {type(out)}.")
        if out.ndim == raw_X.ndim and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def _reduce_q(self, score: Tensor) -> Tensor:
        if score.shape == tuple(self._current_batch_shape):
            return score
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown reduction: {self.reduction!r}.")

    def _finalize(self, value: Tensor, X: Tensor, *, name: str) -> Tensor:
        target_shape = tuple(X.shape[:-2])
        if value.shape == target_shape:
            return value
        if len(target_shape) == 0:
            return value.mean() if value.ndim > 0 else value
        if value.ndim == 0:
            return value.expand(*target_shape)
        if value.numel() == _prod(target_shape):
            return value.reshape(target_shape)
        raise RuntimeError(f"{name}: could not align output to t-batch shape. value={tuple(value.shape)}, target={target_shape}.")

    def _pointwise_score_to_value(self, score_per_output: Tensor, raw_X: Tensor, Xt: Tensor) -> Tensor:
        score = self._aggregate_outputs(score_per_output)
        pending = _align_pointwise_to_reference(self._pending_penalty_per_point(Xt), score, name=f"{self.__class__.__name__}.pending")
        observed = _align_pointwise_to_reference(self._observed_penalty_per_point(Xt), score, name=f"{self.__class__.__name__}.observed")
        score = score - pending - observed
        score = self._apply_objective(score, raw_X=raw_X, expanded_X=Xt)
        value = score if score.shape == raw_X.shape[:-2] else self._reduce_q(score)
        value = value - self._same_batch_penalty(Xt)
        return value

    def _joint_penalty(self, raw_X: Tensor, Xt: Tensor) -> Tensor:
        penalty = self._pending_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._observed_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._same_batch_penalty(Xt)
        return penalty


class qMultiOutputMulticlassPredictiveEntropy(_DirectMultiOutputMulticlassAcqBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._entropy(self._mean_probs(raw_X))
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassProbabilityVariance(_DirectMultiOutputMulticlassAcqBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._class_probability_variance(self._mean_probs(raw_X))
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassMarginUncertainty(_DirectMultiOutputMulticlassAcqBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._margin_uncertainty(self._mean_probs(raw_X))
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassBALD(_DirectMultiOutputMulticlassAcqBase):
    def __init__(self, model, *, num_samples: int = 32, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.num_samples = int(num_samples)

    def _pointwise_bald_per_output(self, X: Tensor) -> Tensor:
        samples = self._sample_probs(X, num_samples=self.num_samples)
        mean_probs = samples.mean(dim=0)
        predictive_entropy = self._entropy(mean_probs)
        expected_entropy = self._entropy(samples).mean(dim=0)
        return predictive_entropy - expected_entropy

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._pointwise_bald_per_output(raw_X)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassJointBALD(qMultiOutputMulticlassBALD):
    def __init__(
        self,
        model,
        *,
        num_samples: int = 32,
        max_joint_q: int = 5,
        max_joint_states: int = 4096,
        large_q_strategy: LargeQStrategy = "per_point",
        **kwargs,
    ) -> None:
        super().__init__(model=model, num_samples=num_samples, reduction="sum", **kwargs)
        self.max_joint_q = int(max_joint_q)
        self.max_joint_states = int(max_joint_states)
        self.large_q_strategy = large_q_strategy

    def _joint_entropy_exact_per_output(self, samples: Tensor) -> Tensor:
        # samples: S x batch_shape x q x m x C
        q = int(samples.shape[-3])
        num_classes = int(samples.shape[-1])
        entropy = samples.new_zeros(samples.shape[1:-3] + samples.shape[-2:-1])
        for state in itertools.product(range(num_classes), repeat=q):
            p_state_per_sample = samples[..., 0, :, state[0]]
            for i in range(1, q):
                p_state_per_sample = p_state_per_sample * samples[..., i, :, state[i]]
            p_state = p_state_per_sample.mean(dim=0).clamp_min(self.eps)
            entropy = entropy - p_state * p_state.log()
        return entropy

    def _joint_bald_per_output(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        samples = self._sample_probs(X, num_samples=self.num_samples)
        q = int(samples.shape[-3])
        num_classes = int(samples.shape[-1])
        num_states = int(num_classes**q)
        if q <= self.max_joint_q and num_states <= self.max_joint_states:
            joint_entropy = self._joint_entropy_exact_per_output(samples)
            conditional_entropy = self._entropy(samples).sum(dim=-2).mean(dim=0)
            return joint_entropy - conditional_entropy
        if self.large_q_strategy == "raise":
            raise RuntimeError(f"Exact multiclass joint BALD is too large: q={q}, C={num_classes}, C**q={num_states}.")
        if self.large_q_strategy == "per_point":
            return self._pointwise_bald_per_output(X).sum(dim=-2)
        if self.large_q_strategy == "truncate":
            k = min(q, self.max_joint_q)
            first = X[..., :k, :]
            rest = X[..., k:, :]
            first_val = self._joint_bald_per_output(first)
            if rest.shape[-2] == 0:
                return first_val
            return first_val + self._pointwise_bald_per_output(rest).sum(dim=-2)
        raise ValueError(f"Unknown large_q_strategy: {self.large_q_strategy!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        value_per_output = self._joint_bald_per_output(raw_X)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._joint_penalty(raw_X, Xt)
        value = self._apply_objective(value, raw_X=raw_X, expanded_X=Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassGreedyJointBALD(qMultiOutputMulticlassJointBALD):
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

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None or X_pending.numel() == 0:
            value_per_output = self._joint_bald_per_output(raw_X)
        else:
            Xp = X_pending.to(device=raw_X.device, dtype=raw_X.dtype)
            Xp = self._expand_pending_to_batch(Xp, raw_X.shape[:-2])
            pending_value = self._joint_bald_per_output(Xp)
            all_value = self._joint_bald_per_output(torch.cat([Xp, raw_X], dim=-2))
            value_per_output = all_value - pending_value
        value = self._aggregate_outputs(value_per_output)
        value = value - self._observed_penalty_per_point(Xt).sum(dim=-1)
        value = value - self._same_batch_penalty(Xt)
        value = self._apply_objective(value, raw_X=raw_X, expanded_X=Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassIntegratedPosteriorVarianceProxy(_DirectMultiOutputMulticlassAcqBase):
    def __init__(
        self,
        model,
        *,
        mc_points: Tensor | Sequence[Tensor] | None = None,
        integration_beta: float = 25.0,
        local_weight: float | None = None,
        integrated_weight: float = 1.0,
        num_samples: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.mc_points = mc_points
        self.integration_beta = float(integration_beta)
        self.local_weight = 1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        self.integrated_weight = float(integrated_weight)
        self.num_samples = int(num_samples)

    def _common_mc_points(self, ref: Tensor) -> Tensor | None:
        if self.mc_points is None:
            return None
        if torch.is_tensor(self.mc_points):
            return self.mc_points.to(device=ref.device, dtype=ref.dtype)
        return None

    def _single_model_mean_probs(self, model, X: Tensor) -> Tensor:
        class_probs = getattr(model, "class_probs", None)
        if callable(class_probs):
            return _TensorProbabilityPosterior(class_probs(X), eps=self.eps).mean
        prob_post = getattr(model, "probability_posterior", None)
        if callable(prob_post):
            return _MaybeProbabilityPosterior(prob_post(X), eps=self.eps, apply_softmax_if_needed=False).mean
        latent_post = getattr(model, "latent_posterior", None)
        if callable(latent_post):
            return _SoftmaxPosterior(latent_post(X)).mean
        return _MaybeProbabilityPosterior(model.posterior(X), eps=self.eps, apply_softmax_if_needed=self.apply_softmax_if_needed).mean

    def _integrated_variance_per_output(self, raw_X: Tensor, Xt: Tensor | None = None) -> Tensor:
        raw_X = self._ensure_q_batch(raw_X)
        Xt = self._apply_input_transform(raw_X) if Xt is None else self._ensure_q_batch(Xt)
        common_mc = self._common_mc_points(raw_X)
        if common_mc is not None:
            mc_raw = common_mc.unsqueeze(0) if common_mc.ndim == 2 else self._ensure_q_batch(common_mc)
            mc_probs = self._mean_probs(mc_raw)
            mc_var = self._class_probability_variance(mc_probs).reshape(-1, mc_probs.shape[-2])
            mc_t = self._apply_input_transform(mc_raw).reshape(-1, Xt.shape[-1])
            d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), mc_t).pow(2)
            weights = torch.exp(-self.integration_beta * d2)
            score = (weights.unsqueeze(-1) * mc_var.unsqueeze(0)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            return score.reshape(*Xt.shape[:-1], mc_var.shape[-1])

        if isinstance(self.mc_points, Sequence):
            submodels = self._submodels()
            if len(self.mc_points) != len(submodels):
                raise ValueError("mc_points sequence length must match number of outputs.")
            scores = []
            for i, points in enumerate(self.mc_points):
                points_i = torch.as_tensor(points, device=raw_X.device, dtype=raw_X.dtype)
                if points_i.ndim != 2:
                    raise ValueError(f"mc_points[{i}] must have shape n_mc x d. Got {tuple(points_i.shape)}.")
                points_q = points_i.unsqueeze(0)
                probs_i = self._single_model_mean_probs(submodels[i], points_q)
                var_i = self._class_probability_variance(probs_i).reshape(-1)
                points_t = self._apply_input_transform(points_q).reshape(-1, Xt.shape[-1])
                d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), points_t).pow(2)
                weights = torch.exp(-self.integration_beta * d2)
                score_i = (weights * var_i.view(1, -1)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(self.eps)
                scores.append(score_i.reshape(*Xt.shape[:-1]).unsqueeze(-1))
            return torch.cat(scores, dim=-1)

        n_outputs = self._mean_probs(raw_X).shape[-2]
        return raw_X.new_zeros(*raw_X.shape[:-1], n_outputs)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_per_output(raw_X, Xt)
        score_per_output = self.local_weight * local_score + self.integrated_weight * integrated_score
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "ReductionType",
    "OutputReductionType",
    "OutputModeType",
    "LargeQStrategy",
    "_DirectMultiOutputMulticlassAcqBase",
    "qMultiOutputMulticlassPredictiveEntropy",
    "qMultiOutputMulticlassProbabilityVariance",
    "qMultiOutputMulticlassMarginUncertainty",
    "qMultiOutputMulticlassBALD",
    "qMultiOutputMulticlassJointBALD",
    "qMultiOutputMulticlassGreedyJointBALD",
    "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
