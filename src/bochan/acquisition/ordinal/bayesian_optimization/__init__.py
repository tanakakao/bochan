"""Ordinal Bayesian optimization acquisitions.

Importing this module is side-effect free. Ordinal utility conversion is an
explicit BoTorch objective; acquisition classes are not patched at runtime.
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
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalExpectedImprovement,
    qHeteroMultiOutputOrdinalExpectedUtility,
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalNormalScoreObjective,
    qHeteroMultiOutputOrdinalNParEGO,
    qHeteroMultiOutputOrdinalProbabilityOfImprovement,
)
from .hetero_single_output import (
    qHeteroOrdinalExpectedImprovement,
    qHeteroOrdinalExpectedUtility,
    qHeteroOrdinalExpectedUtilityUpperConfidenceBound,
    qHeteroOrdinalProbabilityOfImprovement,
)
from .knowledge_gradient import qOrdinalKnowledgeGradient
from .multi_output import (
    compute_observed_ordinal_utility,
    qMultiOutputOrdinalExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
)
from .multi_output import qMultiOutputOrdinalNParEGO as _BaseMultiOutputOrdinalNParEGO
from .multi_output import (
    qMultiOutputOrdinalUtilityObjective as _BaseMultiOutputOrdinalUtilityObjective,
)
from .single_output import (
    compute_ordinal_expected_utility_best_f,
    qOrdinalProbabilityOfFeasibility,
)
from .standard import (
    qOrdinalExpectedImprovement,
    qOrdinalExpectedUtility,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)


class qMultiOutputOrdinalUtilityObjective(_BaseMultiOutputOrdinalUtilityObjective):
    """Ordinal utility objective with explicit wide-multitask likelihood mapping."""

    def __init__(
        self,
        model,
        utility_values,
        *,
        ordinal_likelihoods=None,
        **kwargs,
    ) -> None:
        if ordinal_likelihoods is None:
            num_outputs = int(getattr(model, "num_outputs", 1))
            likelihood = getattr(model, "ordinal_likelihood", None)
            if likelihood is None:
                likelihood = getattr(model, "likelihood", None)
            if likelihood is not None and num_outputs > 1:
                ordinal_likelihoods = [likelihood] * num_outputs
        super().__init__(
            model=model,
            utility_values=utility_values,
            ordinal_likelihoods=ordinal_likelihoods,
            **kwargs,
        )


class qMultiOutputOrdinalNParEGO(_BaseMultiOutputOrdinalNParEGO):
    """Ordinal NParEGO with explicit BoTorch-style outcome constraints."""

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


class qOrdinalExpectedHypervolumeImprovement(
    qMultiOutputOrdinalExpectedHypervolumeImprovement
):
    """Ordinal multi-objective qEHVI with domain-first naming."""


class qOrdinalNoisyExpectedHypervolumeImprovement(
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
):
    """Ordinal multi-objective qNEHVI with domain-first naming."""


class qOrdinalNParEGO(qMultiOutputOrdinalNParEGO):
    """Ordinal multi-objective NParEGO with domain-first naming."""


class qHeteroOrdinalExpectedHypervolumeImprovement(
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement
):
    """Heteroscedastic ordinal qEHVI with domain-first naming."""


class qHeteroOrdinalNoisyExpectedHypervolumeImprovement(
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic ordinal qNEHVI with domain-first naming."""


class qHeteroOrdinalNParEGO(qHeteroMultiOutputOrdinalNParEGO):
    """Heteroscedastic ordinal NParEGO with domain-first naming."""


__all__ = [
    "compute_observed_ordinal_utility",
    "compute_ordinal_expected_utility_best_f",
    "qHeteroMultiOutputOrdinalExpectedImprovement",
    "qHeteroMultiOutputOrdinalExpectedUtility",
    "qHeteroMultiOutputOrdinalNormalScoreObjective",
    "qHeteroMultiOutputOrdinalProbabilityOfImprovement",
    "qHeteroOrdinalExpectedHypervolumeImprovement",
    "qHeteroOrdinalExpectedImprovement",
    "qHeteroOrdinalExpectedUtility",
    "qHeteroOrdinalExpectedUtilityUpperConfidenceBound",
    "qHeteroOrdinalNParEGO",
    "qHeteroOrdinalNoisyExpectedHypervolumeImprovement",
    "qHeteroOrdinalProbabilityOfImprovement",
    "qMultiOutputOrdinalNParEGO",
    "qMultiOutputOrdinalUtilityObjective",
    "qOrdinalExpectedHypervolumeImprovement",
    "qOrdinalExpectedImprovement",
    "qOrdinalExpectedUtility",
    "qOrdinalKnowledgeGradient",
    "qOrdinalNParEGO",
    "qOrdinalNoisyExpectedHypervolumeImprovement",
    "qOrdinalProbabilityOfFeasibility",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
]
