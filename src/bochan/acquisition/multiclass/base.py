from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from torch import Tensor

ReductionType = Literal["mean", "sum", "max"]
ClassReductionType = Literal["mean", "sum", "max", "min", "prod"]


class _MulticlassAcquisitionBase(AcquisitionFunction):
    """Common utilities for multiclass probability-based acquisitions.

    Assumptions:
        - ``model.posterior(X).mean`` returns class probabilities with shape
          ``batch_shape x q x C``.
        - ``model.posterior(X).rsample(sample_shape)`` returns probability samples
          with shape ``sample_shape x batch_shape x q x C``.

    This matches ``MulticlassProbsPosterior`` used by bochan multiclass models.
    """

    def __init__(
        self,
        model,
        *,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-8,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    ) -> None:
        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
            model = model.models[0]
        super().__init__(model=model)
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.eps = float(eps)
        self.objective = objective
        self.set_X_pending(None)

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None:
            likelihood.eval()

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        if X.ndim == 1:
            return X.view(1, 1, -1)
        if X.ndim == 2:
            return X.unsqueeze(-2)
        return X

    def _posterior(self, X: Tensor):
        return self.model.posterior(self._ensure_q_batch(X))

    def _normalize_probs(self, probs: Tensor, X: Tensor, *, name: str) -> Tensor:
        X = self._ensure_q_batch(X)
        if probs.ndim < X.ndim:
            raise RuntimeError(
                f"{name}: posterior probability tensor must include class dimension. "
                f"X.shape={tuple(X.shape)}, probs.shape={tuple(probs.shape)}."
            )
        if probs.shape[-1] <= 1:
            raise RuntimeError(
                f"{name}: multiclass posterior must have class dimension C >= 2. "
                f"Got probs.shape={tuple(probs.shape)}."
            )
        probs = probs.clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _mean_probs(self, X: Tensor) -> Tensor:
        posterior = self._posterior(X)
        return self._normalize_probs(posterior.mean, X, name="mean_probs")

    def _sample_probs(self, X: Tensor, *, num_samples: int) -> Tensor:
        posterior = self._posterior(X)
        samples = posterior.rsample(torch.Size([int(num_samples)]))
        return self._normalize_probs(samples, X, name="sample_probs")

    def _entropy(self, probs: Tensor) -> Tensor:
        probs = probs.clamp_min(self.eps)
        return -(probs * probs.log()).sum(dim=-1)

    def _class_probability_variance(self, probs: Tensor) -> Tensor:
        return (probs * (1.0 - probs)).sum(dim=-1)

    def _margin_uncertainty(self, probs: Tensor) -> Tensor:
        top2 = probs.topk(k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        return 1.0 - margin

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown reduction: {self.reduction!r}.")

    def _reduce_classes(self, probs: Tensor, reduction: ClassReductionType) -> Tensor:
        if reduction == "mean":
            return probs.mean(dim=-1)
        if reduction == "sum":
            return probs.sum(dim=-1)
        if reduction == "max":
            return probs.max(dim=-1).values
        if reduction == "min":
            return probs.min(dim=-1).values
        if reduction == "prod":
            return probs.prod(dim=-1)
        raise ValueError(f"Unknown class reduction: {reduction!r}.")

    def _select_class_probs(
        self,
        probs: Tensor,
        *,
        target_class: int | Sequence[int] | None,
        class_reduction: ClassReductionType = "mean",
    ) -> Tensor:
        """Select or aggregate target-class probabilities.

        If ``target_class`` is ``None``, the maximum class probability is used.
        This is useful for generic feasibility-style scoring. For target-class BO,
        passing an explicit integer is recommended.
        """

        if target_class is None:
            return probs.max(dim=-1).values
        if isinstance(target_class, int):
            return probs[..., int(target_class)]
        indices = [int(i) for i in target_class]
        selected = probs[..., indices]
        return self._reduce_classes(selected, class_reduction)

    def _pending_penalty_per_point(self, X: Tensor) -> Tensor:
        if self.pending_penalty_weight <= 0:
            return torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None:
            return torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        X_pending = torch.as_tensor(X_pending, dtype=X.dtype, device=X.device)
        if X_pending.numel() == 0:
            return torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        if X_pending.ndim == 1:
            X_pending = X_pending.view(1, -1)
        if X_pending.ndim > 2:
            X_pending = X_pending.reshape(-1, X_pending.shape[-1])
        dist = torch.cdist(X.reshape(-1, 1, X.shape[-1]), X_pending.unsqueeze(0)).min(dim=-1).values
        dist = dist.reshape(X.shape[:-1])
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist)

    def _apply_common_pointwise_adjustments(self, score: Tensor, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        penalty = self._pending_penalty_per_point(X)
        if penalty.shape == score.shape:
            score = score - penalty
        elif penalty.numel() == score.numel():
            score = score - penalty.reshape_as(score)
        elif self.pending_penalty_weight > 0:
            raise RuntimeError(
                "Pending penalty shape mismatch: "
                f"score.shape={tuple(score.shape)}, penalty.shape={tuple(penalty.shape)}."
            )
        if self.objective is not None:
            score = self.objective(score, X=X)
        return score

    def _finalize(self, value: Tensor, X: Tensor, *, name: str) -> Tensor:
        X = self._ensure_q_batch(X)
        target_shape = tuple(X.shape[:-2])
        out = value
        if out.shape == target_shape:
            return out
        if len(target_shape) == 0:
            return out.mean() if out.ndim > 0 else out
        if out.ndim == 0:
            return out.expand(*target_shape)
        while out.ndim > len(target_shape):
            out = out.mean(dim=0)
            if out.shape == target_shape:
                return out
        if out.shape == target_shape:
            return out
        target_numel = 1
        for size in target_shape:
            target_numel *= int(size)
        if out.numel() == target_numel:
            return out.reshape(target_shape)
        if out.ndim == 1 and len(target_shape) == 1:
            return out.mean().expand(*target_shape)
        raise RuntimeError(
            f"{name}: could not align output to t-batch shape. "
            f"value.shape={tuple(value.shape)}, target_shape={target_shape}, X.shape={tuple(X.shape)}."
        )


__all__ = [
    "ClassReductionType",
    "ReductionType",
    "_MulticlassAcquisitionBase",
]
