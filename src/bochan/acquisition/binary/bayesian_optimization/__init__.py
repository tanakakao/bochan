"""Binary-classification Bayesian optimization acquisitions.

The package is intentionally declarative: importing it does not modify classes,
functions, or other modules at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.objective.outcome_constraints import (
    OutcomeConstraint,
    split_outcome_constraints,
)

from . import multi_output as _multi_output
from ._utils import compute_binary_best_f, compute_hetero_binary_classification_best_f
from .hetero_multi_output import (
    qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNParEGO,
)
from .hetero_multi_output_stable import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
)
from .hetero_single_output import (
    qHeteroBinaryExpectedImprovement,
    qHeteroBinaryProbabilityOfImprovement,
    qHeteroBinaryUpperConfidenceBound,
)
from .knowledge_gradient import qBinaryKnowledgeGradient
from .multi_output import (
    qMultiOutputBinaryExpectedHypervolumeImprovement,
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
)
from .multi_output import qMultiOutputBinaryNParEGO as _BaseMultiOutputBinaryNParEGO
from .nominal_duplicate_safe import (
    qBinaryProbabilityOfFeasibility,
    qMultiOutputBinaryProbabilityOfFeasibility,
)
from .standard import (
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
)


class qMultiOutputBinaryNParEGO(_BaseMultiOutputBinaryNParEGO):
    """Internal binary NParEGO implementation with explicit outcome constraints."""

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        weights: Tensor | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        rho: float = 0.05,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        constraints: Sequence[OutcomeConstraint] | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_point,
            weights=weights,
            sampler=sampler,
            objective=objective,
            rho=rho,
            samples_are_probs=samples_are_probs,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
        )
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
        posterior = _multi_output._get_binary_mc_posterior_for_probability_samples(
            self.model,
            Xq,
            samples_are_probs=self.samples_are_probs,
            prefer_latent=(not self.samples_are_probs)
            and self.apply_sigmoid_if_needed,
        )
        samples = self.get_posterior_samples(posterior)
        probability_values = _multi_output.to_probability(
            samples,
            apply_sigmoid_if_needed=(
                not self.samples_are_probs or self.apply_sigmoid_if_needed
            ),
            eps=self.eps,
            name="NParEGO posterior samples",
            model=self.model,
            values_are_probs=self.samples_are_probs,
        )
        objective_values = self.base_objective(probability_values, X=Xq)
        scalarized = self._scalarize(objective_values)
        improvement = (
            scalarized - self.best_value.to(scalarized)
        ).clamp_min(0.0)
        feasibility = self._feasibility_factor(
            raw_samples=probability_values,
            objective_values=objective_values,
        )
        if feasibility is not None:
            improvement = improvement * feasibility
        return _multi_output._reduce_sample_and_q_to_tbatch(improvement, Xq)


class qBinaryExpectedHypervolumeImprovement(
    qMultiOutputBinaryExpectedHypervolumeImprovement
):
    """Binary multi-objective qEHVI with domain-first naming."""


class qBinaryNoisyExpectedHypervolumeImprovement(
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement
):
    """Binary multi-objective qNEHVI with domain-first naming."""


class qBinaryNParEGO(qMultiOutputBinaryNParEGO):
    """Binary multi-objective NParEGO with domain-first naming."""


class qHeteroBinaryExpectedHypervolumeImprovement(
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement
):
    """Heteroscedastic binary qEHVI with domain-first naming."""


class qHeteroBinaryNoisyExpectedHypervolumeImprovement(
    qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic binary qNEHVI with domain-first naming."""


class qHeteroBinaryNParEGO(qHeteroMultiOutputBinaryNParEGO):
    """Heteroscedastic binary NParEGO with domain-first naming."""


__all__ = [
    "compute_binary_best_f",
    "compute_hetero_binary_classification_best_f",
    "qBinaryExpectedHypervolumeImprovement",
    "qBinaryExpectedImprovement",
    "qBinaryKnowledgeGradient",
    "qBinaryNParEGO",
    "qBinaryNoisyExpectedHypervolumeImprovement",
    "qBinaryProbabilityOfFeasibility",
    "qBinaryProbabilityOfImprovement",
    "qBinaryUpperConfidenceBound",
    "qHeteroBinaryExpectedHypervolumeImprovement",
    "qHeteroBinaryExpectedImprovement",
    "qHeteroBinaryNParEGO",
    "qHeteroBinaryNoisyExpectedHypervolumeImprovement",
    "qHeteroBinaryProbabilityOfImprovement",
    "qHeteroBinaryUpperConfidenceBound",
    "qMultiOutputBinaryProbabilityOfFeasibility",
]
