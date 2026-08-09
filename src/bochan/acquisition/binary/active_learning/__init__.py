from .nominal_duplicate_safe import (
    qBinaryBALD as _SingleOutputBinaryBALD,
    qBinaryGreedyJointBALD,
    qBinaryIntegratedPosteriorVarianceProxy,
    qBinaryJointBALD,
    qBinaryMarginUncertainty,
    qBinaryPredictiveEntropy as _SingleOutputBinaryPredictiveEntropy,
    qBinaryProbabilityVariance,
    qHeteroBinaryBALD,
    qHeteroBinaryIntegratedPosteriorVariance,
    qHeteroBinaryMarginUncertainty,
    qHeteroBinaryPredictiveEntropy,
    qHeteroBinaryProbabilityVariance,
    qHeteroMultiOutputBinaryBALD,
    qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy,
    qHeteroMultiOutputBinaryMarginUncertainty,
    qHeteroMultiOutputBinaryPredictiveEntropy,
    qHeteroMultiOutputBinaryProbabilityVariance,
    qMultiOutputBinaryBALD,
    qMultiOutputBinaryIntegratedPosteriorVarianceProxy,
    qMultiOutputBinaryMarginUncertainty,
    qMultiOutputBinaryPredictiveEntropy,
    qMultiOutputBinaryProbabilityVariance,
)
from .single_output import (
    qBinaryFantasyNegIntegratedPosteriorVariance as qBinaryFantasyNegIntegratedPosteriorVarianceEvo,
)


def qBinaryPredictiveEntropy(model, *args, **kwargs):
    """Construct single- or multi-output binary predictive entropy."""
    if int(getattr(model, "num_outputs", 1)) > 1:
        return qMultiOutputBinaryPredictiveEntropy(model, *args, **kwargs)
    return _SingleOutputBinaryPredictiveEntropy(model, *args, **kwargs)


def qBinaryBALD(model, *args, **kwargs):
    """Construct single- or multi-output binary BALD from the model shape."""
    if int(getattr(model, "num_outputs", 1)) > 1:
        return qMultiOutputBinaryBALD(model, *args, **kwargs)
    return _SingleOutputBinaryBALD(model, *args, **kwargs)


# The contextual short name ``nipv`` is currently registered against
# qBinaryFantasyNegIntegratedPosteriorVariance. For standard optimize_acqf,
# expose the differentiable proxy under that package-level name. The original
# refit/fantasy implementation remains available explicitly as the ``Evo`` name.
qBinaryFantasyNegIntegratedPosteriorVariance = qBinaryIntegratedPosteriorVarianceProxy

__all__ = [
    "qHeteroMultiOutputBinaryPredictiveEntropy",
    "qHeteroMultiOutputBinaryProbabilityVariance",
    "qHeteroMultiOutputBinaryMarginUncertainty",
    "qHeteroMultiOutputBinaryBALD",
    "qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy",
    "qHeteroBinaryPredictiveEntropy",
    "qHeteroBinaryBALD",
    "qHeteroBinaryProbabilityVariance",
    "qHeteroBinaryMarginUncertainty",
    "qHeteroBinaryIntegratedPosteriorVariance",
    "qMultiOutputBinaryPredictiveEntropy",
    "qMultiOutputBinaryProbabilityVariance",
    "qMultiOutputBinaryMarginUncertainty",
    "qMultiOutputBinaryBALD",
    "qMultiOutputBinaryIntegratedPosteriorVarianceProxy",
    "qBinaryPredictiveEntropy",
    "qBinaryBALD",
    "qBinaryJointBALD",
    "qBinaryGreedyJointBALD",
    "qBinaryProbabilityVariance",
    "qBinaryMarginUncertainty",
    "qBinaryIntegratedPosteriorVarianceProxy",
    "qBinaryFantasyNegIntegratedPosteriorVariance",
    "qBinaryFantasyNegIntegratedPosteriorVarianceEvo",
]
