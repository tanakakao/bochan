"""Multiclass Bayesian optimization acquisitions.

Importing this module is side-effect free. Input-perturbation, baseline, and
constraint behavior must be expressed through models, objectives, or explicit
acquisition constructor arguments rather than runtime class patching.
"""

from __future__ import annotations

from collections.abc import Sequence

from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.objective.outcome_constraints import (
    OutcomeConstraint,
    split_outcome_constraints,
)

from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassNParEGO,
)
from .hetero_single_output import NoiseCombineType, NoiseWeightMode
from .multi_output import (
    MulticlassTargetProbabilityObjective,
    OutputReductionType,
    compute_observed_multiclass_target_probability_values,
    compute_observed_multiclass_utility,
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
)
from .multi_output import (
    qMultiOutputMulticlassNParEGO as _BaseMultiOutputMulticlassNParEGO,
)
from .nominal_duplicate_safe import (
    qHeteroMulticlassExpectedImprovement,
    qHeteroMulticlassProbabilityOfFeasibility,
    qHeteroMulticlassProbabilityOfImprovement,
    qHeteroMulticlassUpperConfidenceBound,
    qMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
)
from .single_output import (
    compute_multiclass_target_probability_best_f,
    compute_multiclass_target_probability_values,
)
from .standard import (
    MulticlassProbabilityObjective,
    qMulticlassExpectedImprovement,
    qMulticlassProbabilityOfImprovement,
    qMulticlassUpperConfidenceBound,
)


class qMultiOutputMulticlassNParEGO(_BaseMultiOutputMulticlassNParEGO):
    """Multiclass NParEGO with explicit BoTorch-style outcome constraints."""

    def __init__(
        self,
        *args,
        constraints: Sequence[OutcomeConstraint] | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        raw_constraints, objective_constraints = split_outcome_constraints(constraints)
        self.constraints = list(constraints or [])
        self.raw_constraints = raw_constraints
        self.objective_constraints = objective_constraints
        self.eta = eta
        self.fat = bool(fat)

    def _feasibility_factor(
        self,
        *,
        raw_samples: Tensor,
        objective_values: Tensor,
    ) -> Tensor | None:
        factor: Tensor | None = None
        if self.raw_constraints:
            factor = compute_smoothed_feasibility_indicator(
                constraints=self.raw_constraints,
                samples=raw_samples,
                eta=self.eta,
                fat=self.fat,
            )
        if self.objective_constraints:
            objective_factor = compute_smoothed_feasibility_indicator(
                constraints=self.objective_constraints,
                samples=objective_values,
                eta=self.eta,
                fat=self.fat,
            )
            factor = objective_factor if factor is None else factor * objective_factor
        return factor

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = _multi_output.ensure_q_batch(X)
        posterior = self.model.posterior(Xq)
        samples = self.get_posterior_samples(posterior)
        objective_values = self.base_objective(samples, X=Xq)
        scalarized = self._scalarize(objective_values)
        improvement = (
            scalarized - self.best_value.to(scalarized)
        ).clamp_min(0.0)
        feasibility = self._feasibility_factor(
            raw_samples=samples,
            objective_values=objective_values,
        )
        if feasibility is not None:
            improvement = improvement * feasibility
        return _multi_output._reduce_sample_and_q_to_tbatch(improvement, Xq)


class qMulticlassExpectedHypervolumeImprovement(
    qMultiOutputMulticlassExpectedHypervolumeImprovement
):
    """Multiclass multi-objective qEHVI with domain-first naming."""


class qMulticlassNoisyExpectedHypervolumeImprovement(
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
):
    """Multiclass multi-objective qNEHVI with domain-first naming."""


class qMulticlassNParEGO(qMultiOutputMulticlassNParEGO):
    """Multiclass multi-objective NParEGO with domain-first naming."""


class qHeteroMulticlassExpectedHypervolumeImprovement(
    qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement
):
    """Heteroscedastic multiclass qEHVI with domain-first naming."""


class qHeteroMulticlassNoisyExpectedHypervolumeImprovement(
    qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic multiclass qNEHVI with domain-first naming."""


class qHeteroMulticlassNParEGO(qHeteroMultiOutputMulticlassNParEGO):
    """Heteroscedastic multiclass NParEGO with domain-first naming."""


__all__ = [
    "MulticlassProbabilityObjective",
    "MulticlassTargetProbabilityObjective",
    "NoiseCombineType",
    "NoiseWeightMode",
    "OutputReductionType",
    "compute_multiclass_target_probability_best_f",
    "compute_multiclass_target_probability_values",
    "compute_observed_multiclass_target_probability_values",
    "compute_observed_multiclass_utility",
    "qHeteroMulticlassExpectedHypervolumeImprovement",
    "qHeteroMulticlassExpectedImprovement",
    "qHeteroMulticlassNParEGO",
    "qHeteroMulticlassNoisyExpectedHypervolumeImprovement",
    "qHeteroMulticlassProbabilityOfFeasibility",
    "qHeteroMulticlassProbabilityOfImprovement",
    "qHeteroMulticlassUpperConfidenceBound",
    "qMulticlassExpectedHypervolumeImprovement",
    "qMulticlassExpectedImprovement",
    "qMulticlassNParEGO",
    "qMulticlassNoisyExpectedHypervolumeImprovement",
    "qMulticlassProbabilityOfFeasibility",
    "qMulticlassProbabilityOfImprovement",
    "qMulticlassUpperConfidenceBound",
    "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassNParEGO",
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
]
