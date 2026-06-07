from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.multi_output import (
    OutputModeType,
    OutputReductionType,
    ReductionType,
    _DirectMultiOutputMulticlassAcqBase,
)
from bochan.acquisition.multiclass.base import ClassReductionType


class _MultiOutputMulticlassTargetClassBOBase(_DirectMultiOutputMulticlassAcqBase):
    """Binary-style base for multi-output multiclass target-class BO.

    This base operates on multiclass probabilities with shape
    ``batch_shape x q_like x m x C``. The target objective is target-class
    probability per output.
    """

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
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
        if target_class is None and output_target_classes is None:
            raise ValueError(
                "target_class or output_target_classes must be specified for "
                "multi-output multiclass Bayesian optimization acquisitions."
            )
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
            objective=objective,
        )
        self.target_class = target_class
        self.output_target_classes = None if output_target_classes is None else [int(i) for i in output_target_classes]
        self.class_reduction = class_reduction

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

        if isinstance(self.target_class, int):
            return probs[..., int(self.target_class)]
        indices = [int(i) for i in self.target_class]
        selected = probs[..., indices]
        return self._reduce_classes(selected)

    def _target_prob_samples_per_output(self, X: Tensor, *, num_samples: int) -> Tensor:
        samples = self._sample_probs(X, num_samples=num_samples)
        return self._target_prob_per_output(samples)

    def _target_prob_mean_per_output(self, X: Tensor) -> Tensor:
        probs = self._mean_probs(X)
        return self._target_prob_per_output(probs)

    def _align_output_param(self, value: float | Tensor, *, ref: Tensor, name: str) -> Tensor:
        value_t = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
        if value_t.ndim == 0:
            return value_t
        n_outputs = int(ref.shape[-1])
        if value_t.numel() == n_outputs:
            return value_t.reshape(*([1] * (ref.ndim - 1)), n_outputs)
        if value_t.numel() == ref.numel():
            return value_t.reshape_as(ref)
        raise ValueError(
            f"{name} must be scalar, length n_outputs, or broadcastable to output score. "
            f"Got {tuple(value_t.shape)}, expected n_outputs={n_outputs}."
        )

    def _pending_q_penalty(self, Xt: Tensor) -> Tensor:
        if self.pending_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-2])
        return self._pending_penalty_per_point(Xt).sum(dim=-1)


class _MultiOutputMulticlassHypervolumeBase(_MultiOutputMulticlassTargetClassBOBase):
    """Base for multiclass multi-objective BO over target-class probabilities."""

    def __init__(
        self,
        model,
        *,
        ref_point: Tensor | Sequence[float],
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        num_samples: int = 128,
        X_baseline: Tensor | None = None,
        baseline_Y: Tensor | None = None,
        objective_signs: Tensor | Sequence[float] | None = None,
        max_hv_q: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            target_class=target_class,
            output_target_classes=output_target_classes,
            reduction="sum",
            **kwargs,
        )
        self.num_samples = int(num_samples)
        self.max_hv_q = int(max_hv_q)
        self.register_buffer("ref_point", torch.as_tensor(ref_point, dtype=torch.double))
        self.register_buffer("objective_signs", None if objective_signs is None else torch.as_tensor(objective_signs, dtype=torch.double))
        self.X_baseline = None if X_baseline is None else torch.as_tensor(X_baseline).detach()
        self.baseline_Y = None if baseline_Y is None else torch.as_tensor(baseline_Y).detach()

    def _signed(self, Y: Tensor) -> Tensor:
        if self.objective_signs is None:
            return Y
        signs = self.objective_signs.to(device=Y.device, dtype=Y.dtype).reshape(-1)
        return Y * signs.view(*([1] * (Y.ndim - 1)), -1)

    def _ref(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        ref = self.ref_point.to(device=device, dtype=dtype).reshape(-1)
        if self.objective_signs is not None:
            ref = ref * self.objective_signs.to(device=device, dtype=dtype).reshape(-1)
        return ref

    def _hypervolume(self, Y: Tensor, ref: Tensor) -> Tensor:
        """Differentiable inclusion-exclusion hypervolume for small q.

        Args:
            Y: ``... x q x m`` objective values, all maximized.
            ref: ``m`` reference point.
        """

        if Y.shape[-2] == 0:
            return Y.new_zeros(Y.shape[:-2])
        q = int(Y.shape[-2])
        if q > self.max_hv_q:
            improv = (Y - ref.view(*([1] * (Y.ndim - 1)), -1)).clamp_min(0.0).sum(dim=-1)
            topk = improv.topk(k=self.max_hv_q, dim=-1).indices
            gather_idx = topk.unsqueeze(-1).expand(*topk.shape, Y.shape[-1])
            Y = torch.gather(Y, dim=-2, index=gather_idx)
            q = self.max_hv_q

        hv = Y.new_zeros(Y.shape[:-2])
        for mask in range(1, 1 << q):
            idx = [i for i in range(q) if mask & (1 << i)]
            corner = Y[..., idx, :].min(dim=-2).values
            volume = (corner - ref.view(*([1] * (corner.ndim - 1)), -1)).clamp_min(0.0).prod(dim=-1)
            hv = hv + volume if (len(idx) % 2 == 1) else hv - volume
        return hv.clamp_min(0.0)

    def _baseline_values(self, *, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        if self.baseline_Y is not None:
            Y = self.baseline_Y.to(device=device, dtype=dtype)
            if Y.ndim == 1:
                Y = Y.unsqueeze(0)
            return self._signed(Y)
        if self.X_baseline is None:
            return None
        Xb = self.X_baseline.to(device=device, dtype=dtype)
        if Xb.ndim == 2:
            Xb = Xb.unsqueeze(0)
        Yb = self._target_prob_mean_per_output(Xb)
        return self._signed(Yb.reshape(-1, Yb.shape[-1]))

    def _baseline_hv(self, *, ref: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        Yb = self._baseline_values(device=device, dtype=dtype)
        if Yb is None:
            return torch.zeros((), device=device, dtype=dtype)
        return self._hypervolume(Yb.unsqueeze(0), ref).squeeze(0)

    def _candidate_samples(self, raw_X: Tensor) -> Tensor:
        return self._signed(self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples))


class qMultiOutputMulticlassProbabilityOfFeasibility(_MultiOutputMulticlassTargetClassBOBase):
    """Multi-output probability of target-class feasibility."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        threshold: float | None = None,
        tau: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            target_class=target_class,
            output_target_classes=output_target_classes,
            **kwargs,
        )
        self.threshold = None if threshold is None else float(threshold)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        score_per_output = p if self.threshold is None else torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassExpectedHypervolumeImprovement(_MultiOutputMulticlassHypervolumeBase):
    """Monte Carlo EHVI over target-class probabilities."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._candidate_samples(raw_X)
        ref = self._ref(device=samples.device, dtype=samples.dtype)
        base_hv = self._baseline_hv(ref=ref, device=samples.device, dtype=samples.dtype)
        hv = self._hypervolume(samples, ref)
        value = (hv - base_hv).clamp_min(0.0).mean(dim=0)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(_MultiOutputMulticlassHypervolumeBase):
    """Monte Carlo NEHVI-style improvement using ``X_baseline`` or ``baseline_Y``."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._candidate_samples(raw_X)
        ref = self._ref(device=samples.device, dtype=samples.dtype)
        Yb = self._baseline_values(device=samples.device, dtype=samples.dtype)
        if Yb is None:
            base_hv = samples.new_zeros(())
            hv = self._hypervolume(samples, ref)
        else:
            batch_shape = samples.shape[1:-2]
            Yb_exp = Yb.view(*([1] * len(batch_shape)), *Yb.shape).expand(*batch_shape, *Yb.shape)
            Yb_exp = Yb_exp.unsqueeze(0).expand(samples.shape[0], *Yb_exp.shape)
            combined = torch.cat([Yb_exp, samples], dim=-2)
            base_hv = self._hypervolume(Yb_exp, ref)
            hv = self._hypervolume(combined, ref)
        value = (hv - base_hv).clamp_min(0.0).mean(dim=0)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassNParEGO(_MultiOutputMulticlassTargetClassBOBase):
    """NParEGO-style random scalarization for multiclass target probabilities."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        weights: Tensor | Sequence[float] | None = None,
        best_f: float | Tensor | None = None,
        X_baseline: Tensor | None = None,
        baseline_Y: Tensor | None = None,
        num_samples: int = 128,
        scalarization: str = "chebyshev",
        rho: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            target_class=target_class,
            output_target_classes=output_target_classes,
            reduction="sum",
            **kwargs,
        )
        self.num_samples = int(num_samples)
        self.scalarization = scalarization
        self.rho = float(rho)
        self.register_buffer("weights", None if weights is None else torch.as_tensor(weights, dtype=torch.double))
        self.register_buffer("best_f", None if best_f is None else torch.as_tensor(best_f, dtype=torch.double))
        self.X_baseline = None if X_baseline is None else torch.as_tensor(X_baseline).detach()
        self.baseline_Y = None if baseline_Y is None else torch.as_tensor(baseline_Y).detach()

    def _weights(self, *, m: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.weights is None:
            return torch.ones(m, device=device, dtype=dtype) / float(m)
        w = self.weights.to(device=device, dtype=dtype).reshape(-1)
        if w.numel() != m:
            raise ValueError(f"weights must have length {m}. Got {w.numel()}.")
        return w / w.abs().sum().clamp_min(self.eps)

    def _scalarize(self, Y: Tensor, weights: Tensor) -> Tensor:
        if self.scalarization == "linear":
            return (Y * weights.view(*([1] * (Y.ndim - 1)), -1)).sum(dim=-1)
        if self.scalarization == "chebyshev":
            weighted = Y * weights.view(*([1] * (Y.ndim - 1)), -1)
            return weighted.min(dim=-1).values + self.rho * weighted.sum(dim=-1)
        raise ValueError(f"Unknown scalarization: {self.scalarization!r}.")

    def _baseline_best_f(self, *, weights: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.best_f is not None:
            return self.best_f.to(device=device, dtype=dtype)
        if self.baseline_Y is not None:
            Yb = self.baseline_Y.to(device=device, dtype=dtype)
            if Yb.ndim == 1:
                Yb = Yb.unsqueeze(0)
            return self._scalarize(Yb, weights).max()
        if self.X_baseline is not None:
            Xb = self.X_baseline.to(device=device, dtype=dtype)
            if Xb.ndim == 2:
                Xb = Xb.unsqueeze(0)
            Yb = self._target_prob_mean_per_output(Xb)
            return self._scalarize(Yb.reshape(-1, Yb.shape[-1]), weights).max()
        return torch.zeros((), device=device, dtype=dtype)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        weights = self._weights(m=samples.shape[-1], device=samples.device, dtype=samples.dtype)
        scalar = self._scalarize(samples, weights)
        best_q = scalar.max(dim=-1).values
        best_f = self._baseline_best_f(weights=weights, device=samples.device, dtype=samples.dtype)
        value = (best_q - best_f).clamp_min(0.0).mean(dim=0)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassExpectedImprovement(_MultiOutputMulticlassTargetClassBOBase):
    """Legacy scalar EI for target-class probability.

    Kept for experimentation, but multi-output multiclass BO should generally use
    qMultiOutputMulticlassExpectedHypervolumeImprovement or NParEGO.
    """

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, output_target_classes: Sequence[int] | None = None, best_f: float | Tensor, num_samples: int = 128, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, output_target_classes=output_target_classes, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        best_q_per_output = samples.max(dim=-2).values
        best_f = self._align_output_param(self.best_f, ref=best_q_per_output, name="best_f")
        value_per_output = (best_q_per_output - best_f).clamp_min(0.0).mean(dim=0)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassProbabilityOfImprovement(_MultiOutputMulticlassTargetClassBOBase):
    """Legacy scalar PI for target-class probability."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, output_target_classes: Sequence[int] | None = None, best_f: float | Tensor, num_samples: int = 128, tau: float = 1e-3, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, output_target_classes=output_target_classes, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        best_q_per_output = samples.max(dim=-2).values
        best_f = self._align_output_param(self.best_f, ref=best_q_per_output, name="best_f")
        tau = self.tau.to(best_q_per_output).clamp_min(self.eps)
        value_per_output = torch.sigmoid((best_q_per_output - best_f) / tau).mean(dim=0)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassUpperConfidenceBound(_MultiOutputMulticlassTargetClassBOBase):
    """Legacy scalar UCB for target-class probability."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, output_target_classes: Sequence[int] | None = None, beta: float | Tensor = 2.0, num_samples: int = 128, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, output_target_classes=output_target_classes, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=False).clamp_min(self.eps)
        beta = self._align_output_param(self.beta, ref=mean, name="beta")
        score_per_output = mean + beta.sqrt() * std
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "OutputReductionType",
    "OutputModeType",
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassNParEGO",
    "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
]
