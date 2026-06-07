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


class _MultiOutputMulticlassTargetProbabilityBase(_DirectMultiOutputMulticlassAcqBase):
    """Binary-style base for multi-output multiclass level-set acquisitions.

    The base operates on multiclass probabilities with shape
    ``batch_shape x q_like x m x C`` and returns one target-probability score per
    output before applying output aggregation.
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
        eps: float = 1e-8,
        objective=None,
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
            objective=objective,
        )
        self.target_class = target_class
        self.output_target_classes = None if output_target_classes is None else [int(i) for i in output_target_classes]
        self.threshold = float(threshold)
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

        if self.target_class is None:
            return probs.max(dim=-1).values
        if isinstance(self.target_class, int):
            return probs[..., int(self.target_class)]
        indices = [int(i) for i in self.target_class]
        selected = probs[..., indices]
        return self._reduce_classes(selected)

    def _apply_pointwise_pipeline(self, score_per_output: Tensor, raw_X: Tensor, Xt: Tensor) -> Tensor:
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassLatentStraddleAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Multi-output target-class probability straddle acquisition."""

    def __init__(self, model, *, beta: float = 1.0, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        std = (p * (1.0 - p)).clamp_min(self.eps).sqrt()
        score_per_output = self.beta * std - (p - self.threshold).abs()
        return self._apply_pointwise_pipeline(score_per_output, raw_X, Xt)


class qMultiOutputMulticlassJointLatentStraddleAcquisition(qMultiOutputMulticlassLatentStraddleAcquisition):
    """q-batch straddle variant using sum reduction by default."""

    def __init__(self, model, **kwargs) -> None:
        kwargs.setdefault("reduction", "sum")
        super().__init__(model=model, **kwargs)


class qMultiOutputMulticlassICUAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Integrated contour uncertainty style acquisition for multi-output multiclass."""

    def __init__(self, model, *, bandwidth: float = 0.10, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        uncertainty = (p * (1.0 - p)).clamp_min(self.eps)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score_per_output = uncertainty * contour_weight
        return self._apply_pointwise_pipeline(score_per_output, raw_X, Xt)


class qMultiOutputMulticlassBoundaryVarianceAcquisition(_MultiOutputMulticlassTargetProbabilityBase):
    """Boundary-weighted target-class variance acquisition."""

    def __init__(self, model, *, bandwidth: float = 0.15, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        variance = p * (1.0 - p)
        boundary_weight = torch.exp(-((p - self.threshold).abs() / max(self.bandwidth, self.eps)))
        score_per_output = variance * boundary_weight
        return self._apply_pointwise_pipeline(score_per_output, raw_X, Xt)


class qMultiOutputMulticlassClassEntropyAcquisition(_DirectMultiOutputMulticlassAcqBase):
    """Per-output class entropy aggregated over outputs."""

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


class qMultiOutputMulticlassProbabilityOfExceedance(_MultiOutputMulticlassTargetProbabilityBase):
    """Smooth probability-space exceedance score for target-class probability."""

    def __init__(self, model, *, tau: float = 0.02, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        score_per_output = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        return self._apply_pointwise_pipeline(score_per_output, raw_X, Xt)


class qMultiOutputMulticlassLevelSetUncertainty(qMultiOutputMulticlassICUAcquisition):
    """Alias for multi-output level-set uncertainty around target-class threshold."""

    pass


__all__ = [
    "OutputReductionType",
    "OutputModeType",
    "qMultiOutputMulticlassLatentStraddleAcquisition",
    "qMultiOutputMulticlassJointLatentStraddleAcquisition",
    "qMultiOutputMulticlassICUAcquisition",
    "qMultiOutputMulticlassBoundaryVarianceAcquisition",
    "qMultiOutputMulticlassClassEntropyAcquisition",
    "qMultiOutputMulticlassProbabilityOfExceedance",
    "qMultiOutputMulticlassLevelSetUncertainty",
]
