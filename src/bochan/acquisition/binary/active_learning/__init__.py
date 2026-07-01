from .hetero_multi_output import (
    qHeteroMultiOutputBinaryPredictiveEntropy,
    qHeteroMultiOutputBinaryProbabilityVariance,
    qHeteroMultiOutputBinaryMarginUncertainty,
    qHeteroMultiOutputBinaryBALD,
    qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy,
)

from .hetero_single_output import (
    qHeteroBinaryPredictiveEntropy,
    qHeteroBinaryBALD,
    qHeteroBinaryProbabilityVariance,
    qHeteroBinaryMarginUncertainty,
    qHeteroBinaryIntegratedPosteriorVariance,
)

from .multi_output import (
    qMultiOutputBinaryPredictiveEntropy,
    qMultiOutputBinaryProbabilityVariance,
    qMultiOutputBinaryMarginUncertainty,
    qMultiOutputBinaryBALD,
    qMultiOutputBinaryIntegratedPosteriorVarianceProxy,
)

from .single_output import (
    qBinaryPredictiveEntropy,
    qBinaryBALD as _SingleOutputBinaryBALD,
    qBinaryJointBALD,
    qBinaryGreedyJointBALD,
    qBinaryProbabilityVariance,
    qBinaryMarginUncertainty,
    qBinaryFantasyNegIntegratedPosteriorVariance as qBinaryFantasyNegIntegratedPosteriorVarianceEvo,
)
from .integrated_posterior_variance import (
    qBinaryIntegratedPosteriorVarianceProxy,
)


def qBinaryBALD(model, *args, **kwargs):
    """Construct single- or multi-output binary BALD from the model shape.

    Correlated Kronecker models are built as one model rather than a ModelList
    wrapper, so high-level metadata may still label them as a single model. Their
    ``num_outputs`` property is the reliable source for selecting the correct
    acquisition implementation.
    """
    if int(getattr(model, "num_outputs", 1)) > 1:
        return qMultiOutputBinaryBALD(model=model, *args, **kwargs)
    return _SingleOutputBinaryBALD(model=model, *args, **kwargs)


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
