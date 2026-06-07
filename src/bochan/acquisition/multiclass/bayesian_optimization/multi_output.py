from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import (
    qMulticlassExpectedImprovement,
    qMulticlassProbabilityOfFeasibility,
    qMulticlassProbabilityOfImprovement,
    qMulticlassUpperConfidenceBound,
)

OutputReductionType = Literal["mean", "sum", "max", "min", "weighted_mean"]


class _MultiOutputMulticlassBOBase(AcquisitionFunction):
    """Base wrapper for multiple multiclass target-class BO outputs."""

    single_output_acqf_cls: type[AcquisitionFunction]

    def __init__(
        self,
        model,
        *,
        output_reduction: OutputReductionType = "mean",
        output_weights: Tensor | Sequence[float] | None = None,
        normalize_output_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model=model)
        self.output_reduction = output_reduction
        self.normalize_output_weights = bool(normalize_output_weights)
        self.output_weights = None if output_weights is None else torch.as_tensor(output_weights, dtype=torch.double)
        self.acqf_kwargs = dict(kwargs)
        self.sub_acqfs = [self.single_output_acqf_cls(submodel, **self.acqf_kwargs) for submodel in self._submodels()]
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


class qMultiOutputMulticlassProbabilityOfFeasibility(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qMulticlassProbabilityOfFeasibility


class qMultiOutputMulticlassExpectedImprovement(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qMulticlassExpectedImprovement


class qMultiOutputMulticlassProbabilityOfImprovement(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qMulticlassProbabilityOfImprovement


class qMultiOutputMulticlassUpperConfidenceBound(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qMulticlassUpperConfidenceBound


__all__ = [
    "OutputReductionType",
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
]
