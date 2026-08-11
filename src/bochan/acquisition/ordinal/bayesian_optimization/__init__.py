"""Ordinal Bayesian optimization acquisitions.

Importing this module is side-effect free.  Ordinal utility conversion is an
explicit BoTorch objective; acquisition classes are not patched at runtime.
"""

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
    qMultiOutputOrdinalNParEGO,
    qMultiOutputOrdinalUtilityObjective,
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
    "compute_ordinal_expected_utility_best_f",
]
