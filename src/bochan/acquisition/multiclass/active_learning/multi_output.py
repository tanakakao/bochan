from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import (
    qMulticlassBALD,
    qMulticlassGreedyJointBALD,
    qMulticlassIntegratedPosteriorVarianceProxy,
    qMulticlassJointBALD,
)

ReductionType = Literal["mean", "sum"]
OutputReductionType = Literal["mean", "sum", "max", "min", "weighted_mean"]
OutputModeType = OutputReductionType
LargeQStrategy = Literal["per_point", "truncate", "raise"]


class _StackedMulticlassPosterior:
    """Stack single-output multiclass posteriors into a multi-output posterior.

    mean:
        ``batch_shape x q x m x C``

    rsample(sample_shape):
        ``sample_shape x batch_shape x q x m x C``
    """

    def __init__(self, posteriors: Sequence, *, eps: float = 1e-8) -> None:
        if len(posteriors) == 0:
            raise ValueError("At least one posterior is required.")
        self.posteriors = list(posteriors)
        self.eps = float(eps)
        self._mean = torch.stack([self._normalize_probs(p.mean) for p in self.posteriors], dim=-2)

    def _normalize_probs(self, probs: Tensor) -> Tensor:
        if probs.ndim >= 1 and probs.shape[-1] <= 1:
            raise RuntimeError(
                "Multiclass posterior must include class dimension C >= 2. "
                f"Got shape={tuple(probs.shape)}."
            )
        probs = probs.clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    @property
    def mean(self) -> Tensor:
        return self._mean

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        samples = [self._normalize_probs(p.rsample(sample_shape)) for p in self.posteriors]
        return torch.stack(samples, dim=-2)


class _MultiOutputMulticlassAcqBase(AcquisitionFunction):
    """Per-output wrapper base kept for hetero multi-output variants.

    Normal multi-output multiclass acquisitions below use
    ``_DirectMultiOutputMulticlassAcqBase`` to match the binary implementation
    style. This wrapper remains because heteroscedastic variants reuse the
    single-output hetero acquisitions directly.
    """

    single_output_acqf_cls: type[AcquisitionFunction]

    def __init__(
        self,
        model,
        *,
        output_reduction: OutputReductionType = "mean",
        output_weights: Tensor | Sequence[float] | None = None,
        normalize_output_weights: bool = True,
        output_acqf_kwargs: Sequence[Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model)
        self.output_reduction = output_reduction
        self.normalize_output_weights = bool(normalize_output_weights)
        self.output_weights = None if output_weights is None else torch.as_tensor(output_weights, dtype=torch.double)
        self.common_acqf_kwargs = dict(kwargs)
        self.submodels = self._submodels()
        self.output_acqf_kwargs = self._normalize_output_acqf_kwargs(output_acqf_kwargs, len(self.submodels))
        self.sub_acqfs = [
            self.single_output_acqf_cls(submodel, **self._kwargs_for_output(i))
            for i, submodel in enumerate(self.submodels)
        ]
        self.set_X_pending(None)

    def _submodels(self) -> list:
        if isinstance(self.model, (ModelListGP, ModelListGPyTorchModel)):
            return list(self.model.models)
        submodels = getattr(self.model, "models", None)
        if submodels is not None:
            return list(submodels)
        submodels = getattr(self.model, "submodels", None)
        if submodels is not None:
            return list(submodels)
        raise RuntimeError(
            f"{self.__class__.__name__} requires a multi-output model with `.models` or `.submodels`. "
            f"Got {type(self.model).__name__}."
        )

    @staticmethod
    def _normalize_output_acqf_kwargs(
        output_acqf_kwargs: Sequence[Mapping[str, Any]] | None,
        n_outputs: int,
    ) -> list[dict[str, Any]]:
        if output_acqf_kwargs is None:
            return [{} for _ in range(n_outputs)]
        if len(output_acqf_kwargs) != n_outputs:
            raise ValueError(
                "output_acqf_kwargs length must match number of outputs. "
                f"Got {len(output_acqf_kwargs)} and {n_outputs}."
            )
        return [dict(item) for item in output_acqf_kwargs]

    def _kwargs_for_output(self, output_idx: int) -> dict[str, Any]:
        kwargs = dict(self.common_acqf_kwargs)
        kwargs.update(self.output_acqf_kwargs[output_idx])
        return kwargs

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = X_pending
        for acqf in getattr(self, "sub_acqfs", []):
            if hasattr(acqf, "set_X_pending"):
                acqf.set_X_pending(X_pending)

    def _reduce_outputs(self, values: Tensor) -> Tensor:
        if values.ndim == 1:
            return values.mean(dim=0)
        if self.output_reduction == "mean":
            return values.mean(dim=0)
        if self.output_reduction == "sum":
            return values.sum(dim=0)
        if self.output_reduction == "max":
            return values.max(dim=0).values
        if self.output_reduction == "min":
            return values.min(dim=0).values
        if self.output_reduction == "weighted_mean":
            if self.output_weights is None:
                raise ValueError("output_weights must be provided when output_reduction='weighted_mean'.")
            weights = self.output_weights.to(device=values.device, dtype=values.dtype)
            if weights.numel() != values.shape[0]:
                raise ValueError(
                    f"output_weights length must match number of outputs. Got {weights.numel()} and {values.shape[0]}."
                )
            if self.normalize_output_weights:
                weights = weights / weights.abs().sum().clamp_min(1e-12)
            return (values * weights.view(-1, *([1] * (values.ndim - 1)))).sum(dim=0)
        raise ValueError(f"Unknown output_reduction: {self.output_reduction!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        vals = [acqf(X) for acqf in self.sub_acqfs]
        return self._reduce_outputs(torch.stack(vals, dim=0))


class _DirectMultiOutputMulticlassAcqBase(AcquisitionFunction):
    """Binary-style base for multi-output multiclass acquisitions.

    This base obtains one multiclass probability tensor with shape
    ``batch_shape x q_like x m x C`` and computes pointwise scores before output
    aggregation. This is intentionally closer to
    ``_MultiOutputBinaryClassificationAcqBase`` than to a per-submodel wrapper.
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
        self.eps = float(eps)
        self.objective = objective
        self.set_X_pending(None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            return X.view(1, 1, -1)
        if X.ndim == 2:
            return X.unsqueeze(-2)
        return X

    def _set_eval_mode(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None:
            likelihood.eval()

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            return Xt[0] if isinstance(Xt, tuple) else Xt
        submodels = getattr(self.model, "models", None) or getattr(self.model, "submodels", None)
        if submodels is not None and len(submodels) > 0:
            it = getattr(submodels[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                return Xt[0] if isinstance(Xt, tuple) else Xt
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

    def _normalize_probs(self, probs: Tensor, *, name: str) -> Tensor:
        if probs.shape[-1] <= 1:
            raise RuntimeError(f"{name}: multiclass probabilities must have class dim C >= 2. Got {tuple(probs.shape)}.")
        probs = probs.clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _get_multiclass_probability_posterior(self, X: Tensor):
        fn = getattr(self.model, "class_probs_list", None)
        if callable(fn):
            probs_list = fn(X)
            return _StackedMulticlassPosterior([_TensorProbabilityPosterior(p) for p in probs_list], eps=self.eps)

        submodels = self._submodels()
        if len(submodels) > 0:
            posteriors = []
            for submodel in submodels:
                class_probs = getattr(submodel, "class_probs", None)
                if callable(class_probs):
                    posteriors.append(_TensorProbabilityPosterior(class_probs(X)))
                    continue
                prob_post = getattr(submodel, "probability_posterior", None)
                posteriors.append(prob_post(X) if callable(prob_post) else submodel.posterior(X))
            return _StackedMulticlassPosterior(posteriors, eps=self.eps)

        prob_post = getattr(self.model, "probability_posterior", None)
        if callable(prob_post):
            return prob_post(X)
        return self.model.posterior(X)

    def _mean_probs(self, X: Tensor) -> Tensor:
        post = self._get_multiclass_probability_posterior(X)
        probs = post.mean
        # Hybrid posterior in probability mode may return target-class scalar
        # probabilities [..., q, m]. In that case true multiclass active learning
        # scores are not well-defined.
        if probs.ndim == X.ndim:
            raise RuntimeError(
                "Multi-output multiclass active learning requires class probabilities with shape "
                "batch_shape x q x m x C. The model posterior returned scalar output probabilities. "
                "Use a HybridMultiOutputModel exposing class_probs_list() or submodels with class_probs()/posterior()."
            )
        return self._normalize_probs(probs, name="mean_probs")

    def _sample_probs(self, X: Tensor, *, num_samples: int) -> Tensor:
        post = self._get_multiclass_probability_posterior(X)
        samples = post.rsample(torch.Size([int(num_samples)]))
        if samples.ndim == X.ndim + 1:
            raise RuntimeError(
                "Multi-output multiclass BALD requires probability samples with shape "
                "S x batch_shape x q x m x C. Got scalar output samples."
            )
        return self._normalize_probs(samples, name="sample_probs")

    def _entropy(self, probs: Tensor) -> Tensor:
        probs = probs.clamp_min(self.eps)
        return -(probs * probs.log()).sum(dim=-1)

    def _class_probability_variance(self, probs: Tensor) -> Tensor:
        return (probs * (1.0 - probs)).sum(dim=-1)

    def _margin_uncertainty(self, probs: Tensor) -> Tensor:
        top2 = probs.topk(k=2, dim=-1).values
        return 1.0 - (top2[..., 0] - top2[..., 1])

    def _aggregate_outputs(self, score_per_output: Tensor) -> Tensor:
        # score_per_output: batch_shape x q_like x m
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
                raise ValueError(
                    f"output_weights must have shape ({score_per_output.shape[-1]},), got {tuple(weights.shape)}."
                )
            if self.normalize_output_weights:
                weights = weights / weights.abs().sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (weights.numel(),)
            return (score_per_output * weights.view(*view_shape)).sum(dim=-1)
        raise ValueError(f"Unknown output_mode/output_reduction: {self.output_mode!r}.")

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.pending_penalty_weight <= 0:
            return torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)
        Xp = getattr(self, "X_pending", None)
        if Xp is None or Xp.numel() == 0:
            return torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)
        Xp = Xp.to(device=Xt.device, dtype=Xt.dtype)
        Xp = self._apply_input_transform(Xp)
        Xp = self._ensure_q_batch(Xp)
        d = Xt.shape[-1]
        dist = torch.cdist(Xt.reshape(-1, d), Xp.reshape(-1, Xp.shape[-1])).min(dim=-1).values
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))

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
        raise ValueError(f"Unknown reduction: {self.reduction!r}.")

    def _finalize(self, value: Tensor, X: Tensor, *, name: str) -> Tensor:
        target_shape = tuple(X.shape[:-2])
        if value.shape == target_shape:
            return value
        if len(target_shape) == 0:
            return value.mean() if value.ndim > 0 else value
        if value.numel() == int(torch.tensor(target_shape).prod().item()):
            return value.reshape(target_shape)
        raise RuntimeError(
            f"{name}: could not align output to t-batch shape. value={tuple(value.shape)}, target={target_shape}."
        )

    def _pointwise_score_to_value(self, score_per_output: Tensor, raw_X: Tensor, Xt: Tensor) -> Tensor:
        score = self._aggregate_outputs(score_per_output)
        score = score - self._pending_penalty_per_point(Xt)
        score = self._apply_objective(score, raw_X=raw_X, expanded_X=Xt)
        return self._reduce_q(score)


class _TensorProbabilityPosterior:
    def __init__(self, probs: Tensor) -> None:
        self._probs = probs

    @property
    def mean(self) -> Tensor:
        return self._probs

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        return self._probs.expand(*sample_shape, *self._probs.shape)


class qMultiOutputMulticlassPredictiveEntropy(_DirectMultiOutputMulticlassAcqBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        score_per_output = self._entropy(probs)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassProbabilityVariance(_DirectMultiOutputMulticlassAcqBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        score_per_output = self._class_probability_variance(probs)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassMarginUncertainty(_DirectMultiOutputMulticlassAcqBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        score_per_output = self._margin_uncertainty(probs)
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
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
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
        samples = self._sample_probs(X, num_samples=self.num_samples)
        q = int(samples.shape[-3])
        num_classes = int(samples.shape[-1])
        num_states = int(num_classes**q)
        if q <= self.max_joint_q and num_states <= self.max_joint_states:
            joint_entropy = self._joint_entropy_exact_per_output(samples)
            conditional_entropy = self._entropy(samples).sum(dim=-2).mean(dim=0)
            return joint_entropy - conditional_entropy
        if self.large_q_strategy == "raise":
            raise RuntimeError(
                f"Exact multiclass joint BALD is too large: q={q}, C={num_classes}, C**q={num_states}."
            )
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
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        value_per_output = self._joint_bald_per_output(raw_X)
        value = self._aggregate_outputs(value_per_output)
        if self.pending_penalty_weight > 0:
            value = value - self._pending_penalty_per_point(Xt).sum(dim=-1)
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
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None or X_pending.numel() == 0:
            value_per_output = self._joint_bald_per_output(raw_X)
            value = self._aggregate_outputs(value_per_output)
            return self._finalize(value, raw_X, name=self.__class__.__name__)
        Xp = X_pending.to(device=raw_X.device, dtype=raw_X.dtype)
        Xp = self._expand_pending_to_batch(Xp, raw_X.shape[:-2])
        pending_value = self._joint_bald_per_output(Xp)
        all_value = self._joint_bald_per_output(torch.cat([Xp, raw_X], dim=-2))
        value = self._aggregate_outputs(all_value - pending_value)
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
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.mc_points = mc_points
        self.integration_beta = float(integration_beta)
        self.local_weight = 1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        self.integrated_weight = float(integrated_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.X_observed = X_observed

    def _observed_penalty_per_point(self, X: Tensor) -> Tensor:
        if self.observed_penalty_weight <= 0 or self.X_observed is None:
            return X.new_zeros(X.shape[:-1])
        X_obs = torch.as_tensor(self.X_observed, device=X.device, dtype=X.dtype)
        if X_obs.ndim == 1:
            X_obs = X_obs.view(1, -1)
        if X_obs.ndim > 2:
            X_obs = X_obs.reshape(-1, X_obs.shape[-1])
        dist = torch.cdist(X.reshape(-1, X.shape[-1]), X_obs).min(dim=-1).values.reshape(X.shape[:-1])
        return self.observed_penalty_weight * torch.exp(-self.observed_penalty_beta * dist)

    def _common_mc_points(self, ref: Tensor) -> Tensor | None:
        if self.mc_points is None:
            return None
        if torch.is_tensor(self.mc_points):
            return self.mc_points.to(device=ref.device, dtype=ref.dtype)
        return None

    def _integrated_variance_per_output(self, X: Tensor) -> Tensor:
        common_mc = self._common_mc_points(X)
        if common_mc is not None:
            mc_probs = self._mean_probs(common_mc.unsqueeze(0) if common_mc.ndim == 2 else common_mc)
            mc_var = self._class_probability_variance(mc_probs).reshape(-1, mc_probs.shape[-2])
            mc_points = common_mc.reshape(-1, common_mc.shape[-1])
            d2 = torch.cdist(X.reshape(-1, X.shape[-1]), mc_points).pow(2)
            weights = torch.exp(-self.integration_beta * d2)
            score = (weights.unsqueeze(-1) * mc_var.unsqueeze(0)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            return score.reshape(*X.shape[:-1], mc_var.shape[-1])

        if isinstance(self.mc_points, Sequence):
            submodels = self._submodels()
            if len(self.mc_points) != len(submodels):
                raise ValueError("mc_points sequence length must match number of outputs.")
            scores = []
            for i, points in enumerate(self.mc_points):
                points_i = torch.as_tensor(points, device=X.device, dtype=X.dtype)
                if points_i.ndim != 2:
                    raise ValueError(f"mc_points[{i}] must have shape n_mc x d. Got {tuple(points_i.shape)}.")
                post = submodels[i].posterior(points_i.unsqueeze(0))
                probs_i = self._normalize_probs(post.mean, name=f"mc_points[{i}]")
                var_i = self._class_probability_variance(probs_i).reshape(-1)
                d2 = torch.cdist(X.reshape(-1, X.shape[-1]), points_i).pow(2)
                weights = torch.exp(-self.integration_beta * d2)
                score_i = (weights * var_i.view(1, -1)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(self.eps)
                scores.append(score_i.reshape(*X.shape[:-1]).unsqueeze(-1))
            return torch.cat(scores, dim=-1)

        return X.new_zeros(*X.shape[:-1], len(self._submodels()))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_per_output(raw_X)
        score_per_output = self.local_weight * local_score + self.integrated_weight * integrated_score
        score = self._aggregate_outputs(score_per_output)
        score = score - self._observed_penalty_per_point(Xt)
        score = score - self._pending_penalty_per_point(Xt)
        score = self._apply_objective(score, raw_X=raw_X, expanded_X=Xt)
        value = self._reduce_q(score)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "ReductionType",
    "OutputReductionType",
    "OutputModeType",
    "LargeQStrategy",
    "qMultiOutputMulticlassPredictiveEntropy",
    "qMultiOutputMulticlassProbabilityVariance",
    "qMultiOutputMulticlassMarginUncertainty",
    "qMultiOutputMulticlassBALD",
    "qMultiOutputMulticlassJointBALD",
    "qMultiOutputMulticlassGreedyJointBALD",
    "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
