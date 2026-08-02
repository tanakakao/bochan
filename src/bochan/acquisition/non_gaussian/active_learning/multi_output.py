"""Multi-output non-Gaussian active-learning acquisitions."""
from __future__ import annotations

from typing import Any, Sequence

import torch
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
            value = value / scales.square()
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
    "NegIntegratedResponseMeanVariance",
    "JointBALDProxy",
    "GreedyJointBALDProxy",
]:
    globals()["qMultiOutputNonGaussian" + _suffix] = _multi(
        "qMultiOutputNonGaussian" + _suffix,
        getattr(_single, "qNonGaussian" + _suffix),
    )

qMultiOutputNonGaussianNegIntegratedPosteriorVariance = (
    qMultiOutputNonGaussianNegIntegratedResponseMeanVariance
)
qMultiOutputNonGaussianNIPV = (
    qMultiOutputNonGaussianNegIntegratedResponseMeanVariance
)

__all__ = [name for name in globals() if name.startswith("qMultiOutput")]
