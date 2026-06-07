from __future__ import annotations

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
    qMulticlassMarginUncertainty,
    qMulticlassPredictiveEntropy,
    qMulticlassProbabilityVariance,
)

OutputReductionType = Literal["mean", "sum", "max", "min", "weighted_mean"]


class _MultiOutputMulticlassAcqBase(AcquisitionFunction):
    """Base wrapper for multiple multiclass outputs.

    The wrapper evaluates a single-output multiclass acquisition on each submodel
    and reduces output-wise acquisition values.

    Args:
        model: Multi-output model. Must expose ``models`` or ``submodels``.
        output_reduction: How to aggregate output-wise acquisition values.
        output_weights: Weights for ``output_reduction='weighted_mean'``.
        normalize_output_weights: Whether to normalize output weights by L1 norm.
        output_acqf_kwargs: Optional per-output acquisition kwargs. This can be
            useful when different outputs need different ``mc_points`` or other
            acquisition-specific options. Each dict is merged with ``kwargs``.
        **kwargs: Common kwargs passed to every single-output acquisition.
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
        self.sub_acqfs: list[AcquisitionFunction] = [
            self.single_output_acqf_cls(
                submodel,
                **self._kwargs_for_output(i),
            )
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

    def set_X_observed(self, X_observed: Tensor | Sequence[Tensor] | None = None) -> None:
        """Propagate observed-reference points to sub acquisitions when supported."""

        self.X_observed = X_observed
        for i, acqf in enumerate(getattr(self, "sub_acqfs", [])):
            if not hasattr(acqf, "set_X_observed"):
                continue
            if isinstance(X_observed, Sequence) and not torch.is_tensor(X_observed):
                if len(X_observed) != len(self.sub_acqfs):
                    raise ValueError(
                        "X_observed sequence length must match number of outputs. "
                        f"Got {len(X_observed)} and {len(self.sub_acqfs)}."
                    )
                acqf.set_X_observed(X_observed[i])
            else:
                acqf.set_X_observed(X_observed)

    def _reduce_outputs(self, values: Tensor) -> Tensor:
        # values: m x batch_shape
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
                    f"output_weights length must match number of outputs. "
                    f"Got {weights.numel()} and {values.shape[0]}."
                )
            if self.normalize_output_weights:
                weights = weights / weights.abs().sum().clamp_min(1e-12)
            return (values * weights.view(-1, *([1] * (values.ndim - 1)))).sum(dim=0)
        raise ValueError(f"Unknown output_reduction: {self.output_reduction!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        vals = [acqf(X) for acqf in self.sub_acqfs]
        stacked = torch.stack(vals, dim=0)
        return self._reduce_outputs(stacked)


class qMultiOutputMulticlassPredictiveEntropy(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qMulticlassPredictiveEntropy


class qMultiOutputMulticlassProbabilityVariance(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qMulticlassProbabilityVariance


class qMultiOutputMulticlassMarginUncertainty(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qMulticlassMarginUncertainty


class qMultiOutputMulticlassBALD(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qMulticlassBALD


class qMultiOutputMulticlassJointBALD(_MultiOutputMulticlassAcqBase):
    """Multi-output wrapper for exact / fallback multiclass joint BALD."""

    single_output_acqf_cls = qMulticlassJointBALD


class qMultiOutputMulticlassGreedyJointBALD(_MultiOutputMulticlassAcqBase):
    """Multi-output wrapper for greedy multiclass joint BALD."""

    single_output_acqf_cls = qMulticlassGreedyJointBALD


class qMultiOutputMulticlassIntegratedPosteriorVarianceProxy(_MultiOutputMulticlassAcqBase):
    """Multi-output wrapper for multiclass IPV proxy.

    To use different integration grids per output, pass ``output_acqf_kwargs``:

    ``output_acqf_kwargs=[{"mc_points": mc0}, {"mc_points": mc1}]``.
    """

    single_output_acqf_cls = qMulticlassIntegratedPosteriorVarianceProxy


__all__ = [
    "OutputReductionType",
    "qMultiOutputMulticlassPredictiveEntropy",
    "qMultiOutputMulticlassProbabilityVariance",
    "qMultiOutputMulticlassMarginUncertainty",
    "qMultiOutputMulticlassBALD",
    "qMultiOutputMulticlassJointBALD",
    "qMultiOutputMulticlassGreedyJointBALD",
    "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
