"""Multi-output non-Gaussian active-learning acquisitions."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from . import single_output as _single


class _MultiOutputMixin:
    """Add weighted output reduction and unit scaling to pointwise scores."""

    def __init__(
        self,
        model,
        *,
        output_reduction: str = "mean",
        output_weights: Tensor | Sequence[float] | None = None,
        output_scales: Tensor | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        valid = {"mean", "sum", "max", "min", "weighted_mean"}
        if output_reduction not in valid:
            raise ValueError(
                f"output_reduction must be one of {sorted(valid)}."
            )
        if output_reduction == "weighted_mean" and output_weights is None:
            raise ValueError(
                "output_weights is required for weighted_mean."
            )
        base_reduction = (
            "mean" if output_reduction == "weighted_mean" else output_reduction
        )
        super().__init__(model, output_reduction=base_reduction, **kwargs)
        self.multi_output_reduction = output_reduction
        self.register_buffer(
            "output_weights",
            None
            if output_weights is None
            else torch.as_tensor(output_weights),
        )
        self.register_buffer(
            "output_scales",
            None if output_scales is None else torch.as_tensor(output_scales),
        )

    def _finish(self, score: Tensor, X: Tensor) -> Tensor:
        """Reduce outputs before applying the common LSE finalization pipeline."""
        Xq = _single.ensure_q_batch(X)
        Xt = self._apply_input_transform_for_distance(Xq)
        if self.multi_output_reduction == "weighted_mean":
            weights = self.output_weights.to(score)
            if (
                weights.numel() != score.shape[-1]
                or torch.any(weights < 0)
                or weights.sum() <= 0
            ):
                raise ValueError(
                    "output_weights must be non-negative, non-zero, and match "
                    "num_outputs."
                )
            score = (score * (weights / weights.sum())).sum(dim=-1)
        else:
            score = self._reduce_outputs_if_needed(
                score,
                Xt,
                name=type(self).__name__,
            )
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name=type(self).__name__,
        )

    def _pointwise(self, X: Tensor, field: str) -> Tensor:
        """Reduce output-wise point scores before q reduction."""
        Xq = _single.ensure_q_batch(X)
        value = getattr(self._stats(Xq), field)
        if self.output_scales is not None:
            scales = self.output_scales.to(value)
            if scales.numel() != value.shape[-1]:
                raise ValueError(
                    "output_scales length must equal num_outputs."
                )
            if torch.any(scales <= 0):
                raise ValueError("output_scales must be positive.")
            value = value / scales
        if self.multi_output_reduction == "weighted_mean":
            weights = self.output_weights.to(value)
            if (
                weights.numel() != value.shape[-1]
                or torch.any(weights < 0)
                or weights.sum() <= 0
            ):
                raise ValueError(
                    "output_weights must be non-negative, non-zero, and match "
                    "num_outputs."
                )
            value = (value * (weights / weights.sum())).sum(dim=-1)
        else:
            Xt = self._apply_input_transform_for_distance(Xq)
            value = self._reduce_outputs_if_needed(
                value,
                Xt,
                name=type(self).__name__,
            )
        Xt = self._apply_input_transform_for_distance(Xq)
        return self._finalize_pointwise_score(
            value,
            X,
            Xt,
            name=type(self).__name__,
        )


def _multi(name: str, parent: type) -> type:
    """Construct a named public multi-output specialization."""
    return type(
        name,
        (_MultiOutputMixin, parent),
        {
            "__doc__": (
                f"Multi-output specialization of ``{parent.__name__}``."
            )
        },
    )


for _suffix in [
    "ResponseMeanVariance",
    "ExpectedObservationVariance",
    "TotalObservationVariance",
    "ExpectedObservationEntropy",
    "PredictiveEntropyProxy",
    "BALDProxy",
    "IntegratedResponseMeanVarianceProxy",
    "JointBALDProxy",
    "GreedyJointBALDProxy",
]:
    globals()["qMultiOutputNonGaussian" + _suffix] = _multi(
        "qMultiOutputNonGaussian" + _suffix,
        getattr(_single, "qNonGaussian" + _suffix),
    )


class qMultiOutputNonGaussianNegIntegratedResponseMeanVariance(
    AcquisitionFunction
):
    """Multi-output non-Gaussian integrated variance-reduction proxy.

    Correlated multitask and model-list non-Gaussian models do not currently
    expose a validated joint response-aware ``fantasize`` contract. This class
    therefore uses the candidate-dependent covariance-reduction proxy and
    applies the same output reduction controls as the other multi-output
    acquisitions.
    """

    def __init__(
        self,
        model,
        mc_points: Tensor,
        *,
        output_reduction: str = "mean",
        output_weights: Tensor | Sequence[float] | None = None,
        output_scales: Tensor | Sequence[float] | None = None,
        X_pending: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        valid = {"mean", "sum", "max", "min", "weighted_mean"}
        if output_reduction not in valid:
            raise ValueError(
                f"output_reduction must be one of {sorted(valid)}."
            )
        if output_reduction == "weighted_mean" and output_weights is None:
            raise ValueError(
                "output_weights is required for weighted_mean."
            )
        base_reduction = (
            "mean" if output_reduction == "weighted_mean" else output_reduction
        )
        if X_pending is not None:
            kwargs.setdefault("X_pending", X_pending)
        self.acqf = _single.qNonGaussianIntegratedResponseMeanVarianceProxy(
            model=model,
            mc_points=mc_points,
            output_reduction=base_reduction,
            **kwargs,
        )
        self.acqf.multi_output_reduction = output_reduction
        self.acqf.register_buffer(
            "output_weights",
            None
            if output_weights is None
            else torch.as_tensor(output_weights),
        )
        self.acqf.register_buffer(
            "output_scales",
            None if output_scales is None else torch.as_tensor(output_scales),
        )

    @property
    def uses_proxy(self) -> bool:
        """Multi-output non-Gaussian NIPV currently uses the proxy."""
        return True

    @property
    def X_pending(self) -> Tensor | None:
        """Return pending points from the delegated acquisition."""
        return getattr(self.acqf, "X_pending", None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """Delegate pending-point updates."""
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        else:
            self.acqf.X_pending = X_pending

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate integrated multi-output response-mean variance reduction."""
        return self.acqf(X)


qMultiOutputNonGaussianNegIntegratedPosteriorVariance = (
    qMultiOutputNonGaussianNegIntegratedResponseMeanVariance
)
qMultiOutputNonGaussianNIPV = (
    qMultiOutputNonGaussianNegIntegratedResponseMeanVariance
)

__all__ = [name for name in globals() if name.startswith("qMultiOutput")]
