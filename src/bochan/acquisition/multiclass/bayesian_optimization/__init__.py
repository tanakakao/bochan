"""Multiclass Bayesian optimization acquisitions.

Importing this module is side-effect free.  Input-perturbation, baseline, and
constraint behavior must be expressed through models, objectives, or explicit
acquisition constructor arguments rather than runtime class patching.
"""

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
    qMultiOutputMulticlassNParEGO,
)
from .nominal_duplicate_safe import (
    qHeteroMultiOutputMulticlassExpectedImprovement,
    qHeteroMultiOutputMulticlassProbabilityOfFeasibility,
    qHeteroMultiOutputMulticlassProbabilityOfImprovement,
    qHeteroMultiOutputMulticlassUpperConfidenceBound,
    qHeteroMulticlassExpectedImprovement,
    qHeteroMulticlassProbabilityOfFeasibility,
    qHeteroMulticlassProbabilityOfImprovement,
    qHeteroMulticlassUpperConfidenceBound,
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
    qMulticlassProbabilityOfFeasibility,
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
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
]
