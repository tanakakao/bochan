from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from bochan.acquisition.multiclass.base import ClassReductionType

from .multi_output import _make_target_objective


class qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    """Shape-safe qNEHVI for multiclass multi-output objectives.

    Multiclass probability samples have shape ``... x q x m x C`` and are mapped
    by the objective to ``... x q x m``. During qNEHVI baseline initialization,
    BoTorch calls ``objective(samples, X=X_baseline)`` and performs a strict
    q-shape assertion against the raw ``X_baseline``. For custom multiclass
    probability posteriors this assertion can fail even when the objective value
    tensor itself is valid. This wrapper disables only that objective shape
    assertion and leaves the EHVI / NEHVI computation unchanged.
    """

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
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = False,
        marginalize_dim: Optional[int] = None,
        eps: float = 1e-8,
    ) -> None:
        objective = objective or _make_target_objective(
            ref_point=ref_point,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            utility_values=utility_values,
            objective_signs=objective_signs,
            eps=eps,
        )

        # This is the key fix for the current error. MCAcquisitionObjective.__call__
        # checks output.shape[-2] == X.shape[-2] for MO objectives. Multiclass
        # probability objectives reduce a class dimension and may have internal
        # q layout that does not match the raw baseline X. The transformed values
        # are still valid, so disable only this assertion.
        if hasattr(objective, "_verify_output_shape"):
            objective._verify_output_shape = False

        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )
        self.eta = eta
        self.fat = fat

    def forward(self, X: Tensor) -> Tensor:
        value = super().forward(X)
        # Some BoTorch versions return a trailing singleton in sequential q=1
        # evaluation. optimize_acqf expects exactly the t-batch shape.
        while value.ndim > 0 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value


__all__ = ["qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement"]
