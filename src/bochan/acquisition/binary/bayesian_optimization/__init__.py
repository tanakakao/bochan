"""Binary-classification Bayesian optimization acquisitions.

The package is intentionally declarative: importing it does not modify classes,
functions, or other modules at runtime.
"""

from __future__ import annotations

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
    qMultiOutputBinaryNParEGO,
)
from .nominal_duplicate_safe import (
    qBinaryProbabilityOfFeasibility,
    qMultiOutputBinaryProbabilityOfFeasibility,
)
from .standard import (
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
)


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
