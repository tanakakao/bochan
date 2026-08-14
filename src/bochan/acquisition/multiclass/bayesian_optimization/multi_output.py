"""Multiclass Bayesian-optimization multi-output acquisitions.

The implementation core is kept in ``_multi_output_core``. This public facade
owns baseline partitioning for qNEHVI so InputPerturbation-aware objectives
always receive the raw ``X_baseline`` used to produce posterior values.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from torch import Tensor

from bochan.acquisition.multiclass.base import ClassReductionType

from . import _multi_output_core as _core

MulticlassTargetProbabilityObjective = _core.MulticlassTargetProbabilityObjective
compute_observed_multiclass_utility = _core.compute_observed_multiclass_utility
compute_observed_multiclass_target_probability_values = (
    _core.compute_observed_multiclass_target_probability_values
)
OutputReductionType = _core.OutputReductionType
OutputModeType = _core.OutputModeType
ReductionType = _core.ReductionType
ensure_q_batch = _core.ensure_q_batch
_reduce_sample_and_q_to_tbatch = _core._reduce_sample_and_q_to_tbatch

qMultiOutputMulticlassProbabilityOfFeasibility = (
    _core.qMultiOutputMulticlassProbabilityOfFeasibility
)
qMultiOutputMulticlassExpectedHypervolumeImprovement = (
    _core.qMultiOutputMulticlassExpectedHypervolumeImprovement
)
qMultiOutputMulticlassNParEGO = _core.qMultiOutputMulticlassNParEGO
qMultiOutputMulticlassExpectedImprovement = _core.qMultiOutputMulticlassExpectedImprovement
qMultiOutputMulticlassProbabilityOfImprovement = (
    _core.qMultiOutputMulticlassProbabilityOfImprovement
)
qMultiOutputMulticlassUpperConfidenceBound = (
    _core.qMultiOutputMulticlassUpperConfidenceBound
)


def _baseline_partitioning_from_model(
    *,
    model: Model,
    X_baseline: Tensor,
    ref_point: Tensor,
    objective: MCMultiOutputObjective,
) -> FastNondominatedPartitioning:
    """Build qNEHVI baseline partitioning with the matching raw baseline X."""

    with torch.no_grad():
        Xb = ensure_q_batch(X_baseline)
        posterior = model.posterior(Xb)
        values = objective(posterior.mean.unsqueeze(0), X=Xb)
        Y_baseline = _core._collapse_objective_values_to_2d(
            values,
            num_outputs=int(ref_point.numel()),
        )
    return FastNondominatedPartitioning(ref_point=ref_point, Y=Y_baseline)


class qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(
    _core.qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
):
    """Shape-safe qNEHVI with raw-X-aware baseline objective evaluation."""

    def __init__(
        self,
        model: Model,
        ref_point: Tensor | Sequence[float],
        X_baseline: Tensor,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list | None = None,
        X_pending: Tensor | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = False,
        marginalize_dim: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        ref_tensor = torch.as_tensor(
            ref_point,
            device=X_baseline.device,
            dtype=X_baseline.dtype,
        ).reshape(-1)
        objective = _core._disable_objective_shape_check(
            objective
            or _core._make_target_objective(
                ref_point=ref_tensor,
                target_class=target_class,
                output_target_classes=output_target_classes,
                class_reduction=class_reduction,
                utility_values=utility_values,
                objective_signs=objective_signs,
                eps=eps,
            )
        )
        partitioning = _baseline_partitioning_from_model(
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_tensor,
            objective=objective,
        )
        _core.qMultiOutputMulticlassExpectedHypervolumeImprovement.__init__(
            self,
            model=model,
            ref_point=ref_tensor,
            partitioning=partitioning,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            utility_values=utility_values,
            objective_signs=objective_signs,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            eps=eps,
        )
        self.X_baseline = X_baseline
        self.prune_baseline = prune_baseline
        self.alpha = alpha
        self.cache_pending = cache_pending
        self.max_iep = max_iep
        self.incremental_nehvi = incremental_nehvi
        self.cache_root = cache_root
        self.marginalize_dim = marginalize_dim


def __getattr__(name: str):
    """Delegate private implementation helpers to the unchanged core module."""

    return getattr(_core, name)


__all__ = [
    "MulticlassTargetProbabilityObjective",
    "compute_observed_multiclass_utility",
    "compute_observed_multiclass_target_probability_values",
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
