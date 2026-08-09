"""Multiclass Bayesian-optimization acquisitions with nominal duplicate semantics."""

from bochan.acquisition._nominal_duplicate_penalties import (
    NominalDuplicatePenaltyMixin,
)

from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement as _qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassExpectedImprovement as _qHeteroMultiOutputMulticlassExpectedImprovement,
    qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement as _qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassNParEGO as _qHeteroMultiOutputMulticlassNParEGO,
    qHeteroMultiOutputMulticlassProbabilityOfFeasibility as _qHeteroMultiOutputMulticlassProbabilityOfFeasibility,
    qHeteroMultiOutputMulticlassProbabilityOfImprovement as _qHeteroMultiOutputMulticlassProbabilityOfImprovement,
    qHeteroMultiOutputMulticlassUpperConfidenceBound as _qHeteroMultiOutputMulticlassUpperConfidenceBound,
)
from .hetero_single_output import (
    qHeteroMulticlassExpectedImprovement as _qHeteroMulticlassExpectedImprovement,
    qHeteroMulticlassProbabilityOfFeasibility as _qHeteroMulticlassProbabilityOfFeasibility,
    qHeteroMulticlassProbabilityOfImprovement as _qHeteroMulticlassProbabilityOfImprovement,
    qHeteroMulticlassUpperConfidenceBound as _qHeteroMulticlassUpperConfidenceBound,
)
from .multi_output import (
    qMultiOutputMulticlassExpectedHypervolumeImprovement as _qMultiOutputMulticlassExpectedHypervolumeImprovement,
    qMultiOutputMulticlassExpectedImprovement as _qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement as _qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qMultiOutputMulticlassNParEGO as _qMultiOutputMulticlassNParEGO,
    qMultiOutputMulticlassProbabilityOfFeasibility as _qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement as _qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound as _qMultiOutputMulticlassUpperConfidenceBound,
)
from .single_output import (
    qMulticlassExpectedImprovement as _qMulticlassExpectedImprovement,
    qMulticlassProbabilityOfFeasibility as _qMulticlassProbabilityOfFeasibility,
    qMulticlassProbabilityOfImprovement as _qMulticlassProbabilityOfImprovement,
    qMulticlassUpperConfidenceBound as _qMulticlassUpperConfidenceBound,
)


def _safe(name: str, base: type) -> type:
    return type(name, (NominalDuplicatePenaltyMixin, base), {"__module__": __name__})


qMulticlassProbabilityOfFeasibility = _safe(
    "qMulticlassProbabilityOfFeasibility",
    _qMulticlassProbabilityOfFeasibility,
)
qMulticlassExpectedImprovement = _safe(
    "qMulticlassExpectedImprovement",
    _qMulticlassExpectedImprovement,
)
qMulticlassProbabilityOfImprovement = _safe(
    "qMulticlassProbabilityOfImprovement",
    _qMulticlassProbabilityOfImprovement,
)
qMulticlassUpperConfidenceBound = _safe(
    "qMulticlassUpperConfidenceBound",
    _qMulticlassUpperConfidenceBound,
)

qMultiOutputMulticlassProbabilityOfFeasibility = _safe(
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    _qMultiOutputMulticlassProbabilityOfFeasibility,
)
qMultiOutputMulticlassExpectedImprovement = _safe(
    "qMultiOutputMulticlassExpectedImprovement",
    _qMultiOutputMulticlassExpectedImprovement,
)
qMultiOutputMulticlassProbabilityOfImprovement = _safe(
    "qMultiOutputMulticlassProbabilityOfImprovement",
    _qMultiOutputMulticlassProbabilityOfImprovement,
)
qMultiOutputMulticlassUpperConfidenceBound = _safe(
    "qMultiOutputMulticlassUpperConfidenceBound",
    _qMultiOutputMulticlassUpperConfidenceBound,
)
qMultiOutputMulticlassExpectedHypervolumeImprovement = _safe(
    "qMultiOutputMulticlassExpectedHypervolumeImprovement",
    _qMultiOutputMulticlassExpectedHypervolumeImprovement,
)
qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement = _safe(
    "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    _qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
)
qMultiOutputMulticlassNParEGO = _safe(
    "qMultiOutputMulticlassNParEGO",
    _qMultiOutputMulticlassNParEGO,
)

qHeteroMulticlassProbabilityOfFeasibility = _safe(
    "qHeteroMulticlassProbabilityOfFeasibility",
    _qHeteroMulticlassProbabilityOfFeasibility,
)
qHeteroMulticlassExpectedImprovement = _safe(
    "qHeteroMulticlassExpectedImprovement",
    _qHeteroMulticlassExpectedImprovement,
)
qHeteroMulticlassProbabilityOfImprovement = _safe(
    "qHeteroMulticlassProbabilityOfImprovement",
    _qHeteroMulticlassProbabilityOfImprovement,
)
qHeteroMulticlassUpperConfidenceBound = _safe(
    "qHeteroMulticlassUpperConfidenceBound",
    _qHeteroMulticlassUpperConfidenceBound,
)

qHeteroMultiOutputMulticlassProbabilityOfFeasibility = _safe(
    "qHeteroMultiOutputMulticlassProbabilityOfFeasibility",
    _qHeteroMultiOutputMulticlassProbabilityOfFeasibility,
)
qHeteroMultiOutputMulticlassExpectedImprovement = _safe(
    "qHeteroMultiOutputMulticlassExpectedImprovement",
    _qHeteroMultiOutputMulticlassExpectedImprovement,
)
qHeteroMultiOutputMulticlassProbabilityOfImprovement = _safe(
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
    _qHeteroMultiOutputMulticlassProbabilityOfImprovement,
)
qHeteroMultiOutputMulticlassUpperConfidenceBound = _safe(
    "qHeteroMultiOutputMulticlassUpperConfidenceBound",
    _qHeteroMultiOutputMulticlassUpperConfidenceBound,
)
qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement = _safe(
    "qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement",
    _qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement,
)
qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement = _safe(
    "qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    _qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
)
qHeteroMultiOutputMulticlassNParEGO = _safe(
    "qHeteroMultiOutputMulticlassNParEGO",
    _qHeteroMultiOutputMulticlassNParEGO,
)


__all__ = [name for name in globals() if name.startswith("q")]
