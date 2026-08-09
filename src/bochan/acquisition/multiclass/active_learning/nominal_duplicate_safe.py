"""Multiclass active-learning acquisitions with nominal duplicate semantics."""

from bochan.acquisition._nominal_duplicate_penalties import (
    NominalDuplicatePenaltyMixin,
)

from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassBALD as _qHeteroMultiOutputMulticlassBALD,
    qHeteroMultiOutputMulticlassGreedyJointBALD as _qHeteroMultiOutputMulticlassGreedyJointBALD,
    qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy as _qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
    qHeteroMultiOutputMulticlassJointBALD as _qHeteroMultiOutputMulticlassJointBALD,
    qHeteroMultiOutputMulticlassMarginUncertainty as _qHeteroMultiOutputMulticlassMarginUncertainty,
    qHeteroMultiOutputMulticlassPredictiveEntropy as _qHeteroMultiOutputMulticlassPredictiveEntropy,
    qHeteroMultiOutputMulticlassProbabilityVariance as _qHeteroMultiOutputMulticlassProbabilityVariance,
)
from .hetero_single_output import (
    qHeteroMulticlassBALD as _qHeteroMulticlassBALD,
    qHeteroMulticlassGreedyJointBALD as _qHeteroMulticlassGreedyJointBALD,
    qHeteroMulticlassIntegratedPosteriorVarianceProxy as _qHeteroMulticlassIntegratedPosteriorVarianceProxy,
    qHeteroMulticlassJointBALD as _qHeteroMulticlassJointBALD,
    qHeteroMulticlassMarginUncertainty as _qHeteroMulticlassMarginUncertainty,
    qHeteroMulticlassPredictiveEntropy as _qHeteroMulticlassPredictiveEntropy,
    qHeteroMulticlassProbabilityVariance as _qHeteroMulticlassProbabilityVariance,
)
from .multi_output import (
    qMultiOutputMulticlassBALD as _qMultiOutputMulticlassBALD,
    qMultiOutputMulticlassGreedyJointBALD as _qMultiOutputMulticlassGreedyJointBALD,
    qMultiOutputMulticlassIntegratedPosteriorVarianceProxy as _qMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
    qMultiOutputMulticlassJointBALD as _qMultiOutputMulticlassJointBALD,
    qMultiOutputMulticlassMarginUncertainty as _qMultiOutputMulticlassMarginUncertainty,
    qMultiOutputMulticlassPredictiveEntropy as _qMultiOutputMulticlassPredictiveEntropy,
    qMultiOutputMulticlassProbabilityVariance as _qMultiOutputMulticlassProbabilityVariance,
)
from .single_output import (
    qMulticlassBALD as _qMulticlassBALD,
    qMulticlassGreedyJointBALD as _qMulticlassGreedyJointBALD,
    qMulticlassIntegratedPosteriorVarianceProxy as _qMulticlassIntegratedPosteriorVarianceProxy,
    qMulticlassJointBALD as _qMulticlassJointBALD,
    qMulticlassMarginUncertainty as _qMulticlassMarginUncertainty,
    qMulticlassPredictiveEntropy as _qMulticlassPredictiveEntropy,
    qMulticlassProbabilityVariance as _qMulticlassProbabilityVariance,
)


def _safe(name: str, base: type) -> type:
    return type(
        name,
        (NominalDuplicatePenaltyMixin, base),
        {"__module__": __name__},
    )


qMulticlassPredictiveEntropy = _safe("qMulticlassPredictiveEntropy", _qMulticlassPredictiveEntropy)
qMulticlassProbabilityVariance = _safe("qMulticlassProbabilityVariance", _qMulticlassProbabilityVariance)
qMulticlassMarginUncertainty = _safe("qMulticlassMarginUncertainty", _qMulticlassMarginUncertainty)
qMulticlassBALD = _safe("qMulticlassBALD", _qMulticlassBALD)
qMulticlassJointBALD = _safe("qMulticlassJointBALD", _qMulticlassJointBALD)
qMulticlassGreedyJointBALD = _safe("qMulticlassGreedyJointBALD", _qMulticlassGreedyJointBALD)
qMulticlassIntegratedPosteriorVarianceProxy = _safe(
    "qMulticlassIntegratedPosteriorVarianceProxy",
    _qMulticlassIntegratedPosteriorVarianceProxy,
)

qMultiOutputMulticlassPredictiveEntropy = _safe(
    "qMultiOutputMulticlassPredictiveEntropy",
    _qMultiOutputMulticlassPredictiveEntropy,
)
qMultiOutputMulticlassProbabilityVariance = _safe(
    "qMultiOutputMulticlassProbabilityVariance",
    _qMultiOutputMulticlassProbabilityVariance,
)
qMultiOutputMulticlassMarginUncertainty = _safe(
    "qMultiOutputMulticlassMarginUncertainty",
    _qMultiOutputMulticlassMarginUncertainty,
)
qMultiOutputMulticlassBALD = _safe("qMultiOutputMulticlassBALD", _qMultiOutputMulticlassBALD)
qMultiOutputMulticlassJointBALD = _safe(
    "qMultiOutputMulticlassJointBALD",
    _qMultiOutputMulticlassJointBALD,
)
qMultiOutputMulticlassGreedyJointBALD = _safe(
    "qMultiOutputMulticlassGreedyJointBALD",
    _qMultiOutputMulticlassGreedyJointBALD,
)
qMultiOutputMulticlassIntegratedPosteriorVarianceProxy = _safe(
    "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
    _qMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
)

qHeteroMulticlassPredictiveEntropy = _safe(
    "qHeteroMulticlassPredictiveEntropy",
    _qHeteroMulticlassPredictiveEntropy,
)
qHeteroMulticlassProbabilityVariance = _safe(
    "qHeteroMulticlassProbabilityVariance",
    _qHeteroMulticlassProbabilityVariance,
)
qHeteroMulticlassMarginUncertainty = _safe(
    "qHeteroMulticlassMarginUncertainty",
    _qHeteroMulticlassMarginUncertainty,
)
qHeteroMulticlassBALD = _safe("qHeteroMulticlassBALD", _qHeteroMulticlassBALD)
qHeteroMulticlassJointBALD = _safe("qHeteroMulticlassJointBALD", _qHeteroMulticlassJointBALD)
qHeteroMulticlassGreedyJointBALD = _safe(
    "qHeteroMulticlassGreedyJointBALD",
    _qHeteroMulticlassGreedyJointBALD,
)
qHeteroMulticlassIntegratedPosteriorVarianceProxy = _safe(
    "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
    _qHeteroMulticlassIntegratedPosteriorVarianceProxy,
)

qHeteroMultiOutputMulticlassPredictiveEntropy = _safe(
    "qHeteroMultiOutputMulticlassPredictiveEntropy",
    _qHeteroMultiOutputMulticlassPredictiveEntropy,
)
qHeteroMultiOutputMulticlassProbabilityVariance = _safe(
    "qHeteroMultiOutputMulticlassProbabilityVariance",
    _qHeteroMultiOutputMulticlassProbabilityVariance,
)
qHeteroMultiOutputMulticlassMarginUncertainty = _safe(
    "qHeteroMultiOutputMulticlassMarginUncertainty",
    _qHeteroMultiOutputMulticlassMarginUncertainty,
)
qHeteroMultiOutputMulticlassBALD = _safe(
    "qHeteroMultiOutputMulticlassBALD",
    _qHeteroMultiOutputMulticlassBALD,
)
qHeteroMultiOutputMulticlassJointBALD = _safe(
    "qHeteroMultiOutputMulticlassJointBALD",
    _qHeteroMultiOutputMulticlassJointBALD,
)
qHeteroMultiOutputMulticlassGreedyJointBALD = _safe(
    "qHeteroMultiOutputMulticlassGreedyJointBALD",
    _qHeteroMultiOutputMulticlassGreedyJointBALD,
)
qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy = _safe(
    "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
    _qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
)


__all__ = [name for name in globals() if name.startswith("q")]
