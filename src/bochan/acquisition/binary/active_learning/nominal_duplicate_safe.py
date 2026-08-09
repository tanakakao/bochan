"""Binary active-learning acquisitions with nominal duplicate semantics."""

from bochan.acquisition._nominal_duplicate_penalties import (
    NominalDuplicatePenaltyMixin,
)

from .hetero_multi_output import (
    qHeteroMultiOutputBinaryBALD as _qHeteroMultiOutputBinaryBALD,
    qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy as _qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy,
    qHeteroMultiOutputBinaryMarginUncertainty as _qHeteroMultiOutputBinaryMarginUncertainty,
    qHeteroMultiOutputBinaryPredictiveEntropy as _qHeteroMultiOutputBinaryPredictiveEntropy,
    qHeteroMultiOutputBinaryProbabilityVariance as _qHeteroMultiOutputBinaryProbabilityVariance,
)
from .hetero_single_output import (
    qHeteroBinaryBALD as _qHeteroBinaryBALD,
    qHeteroBinaryIntegratedPosteriorVariance as _qHeteroBinaryIntegratedPosteriorVariance,
    qHeteroBinaryMarginUncertainty as _qHeteroBinaryMarginUncertainty,
    qHeteroBinaryPredictiveEntropy as _qHeteroBinaryPredictiveEntropy,
    qHeteroBinaryProbabilityVariance as _qHeteroBinaryProbabilityVariance,
)
from .integrated_posterior_variance import (
    qBinaryIntegratedPosteriorVarianceProxy as _qBinaryIntegratedPosteriorVarianceProxy,
)
from .multi_output import (
    qMultiOutputBinaryBALD as _qMultiOutputBinaryBALD,
    qMultiOutputBinaryIntegratedPosteriorVarianceProxy as _qMultiOutputBinaryIntegratedPosteriorVarianceProxy,
    qMultiOutputBinaryMarginUncertainty as _qMultiOutputBinaryMarginUncertainty,
    qMultiOutputBinaryPredictiveEntropy as _qMultiOutputBinaryPredictiveEntropy,
    qMultiOutputBinaryProbabilityVariance as _qMultiOutputBinaryProbabilityVariance,
)
from .single_output import (
    qBinaryBALD as _qBinaryBALD,
    qBinaryGreedyJointBALD as _qBinaryGreedyJointBALD,
    qBinaryJointBALD as _qBinaryJointBALD,
    qBinaryMarginUncertainty as _qBinaryMarginUncertainty,
    qBinaryPredictiveEntropy as _qBinaryPredictiveEntropy,
    qBinaryProbabilityVariance as _qBinaryProbabilityVariance,
)


class qBinaryPredictiveEntropy(NominalDuplicatePenaltyMixin, _qBinaryPredictiveEntropy):
    pass


class qBinaryProbabilityVariance(NominalDuplicatePenaltyMixin, _qBinaryProbabilityVariance):
    pass


class qBinaryMarginUncertainty(NominalDuplicatePenaltyMixin, _qBinaryMarginUncertainty):
    pass


class qBinaryBALD(NominalDuplicatePenaltyMixin, _qBinaryBALD):
    pass


class qBinaryJointBALD(NominalDuplicatePenaltyMixin, _qBinaryJointBALD):
    pass


class qBinaryGreedyJointBALD(NominalDuplicatePenaltyMixin, _qBinaryGreedyJointBALD):
    pass


class qBinaryIntegratedPosteriorVarianceProxy(
    NominalDuplicatePenaltyMixin,
    _qBinaryIntegratedPosteriorVarianceProxy,
):
    pass


class qMultiOutputBinaryPredictiveEntropy(
    NominalDuplicatePenaltyMixin,
    _qMultiOutputBinaryPredictiveEntropy,
):
    pass


class qMultiOutputBinaryProbabilityVariance(
    NominalDuplicatePenaltyMixin,
    _qMultiOutputBinaryProbabilityVariance,
):
    pass


class qMultiOutputBinaryMarginUncertainty(
    NominalDuplicatePenaltyMixin,
    _qMultiOutputBinaryMarginUncertainty,
):
    pass


class qMultiOutputBinaryBALD(NominalDuplicatePenaltyMixin, _qMultiOutputBinaryBALD):
    pass


class qMultiOutputBinaryIntegratedPosteriorVarianceProxy(
    NominalDuplicatePenaltyMixin,
    _qMultiOutputBinaryIntegratedPosteriorVarianceProxy,
):
    pass


class qHeteroBinaryPredictiveEntropy(
    NominalDuplicatePenaltyMixin,
    _qHeteroBinaryPredictiveEntropy,
):
    pass


class qHeteroBinaryProbabilityVariance(
    NominalDuplicatePenaltyMixin,
    _qHeteroBinaryProbabilityVariance,
):
    pass


class qHeteroBinaryMarginUncertainty(
    NominalDuplicatePenaltyMixin,
    _qHeteroBinaryMarginUncertainty,
):
    pass


class qHeteroBinaryBALD(NominalDuplicatePenaltyMixin, _qHeteroBinaryBALD):
    pass


class qHeteroBinaryIntegratedPosteriorVariance(
    NominalDuplicatePenaltyMixin,
    _qHeteroBinaryIntegratedPosteriorVariance,
):
    pass


class qHeteroMultiOutputBinaryPredictiveEntropy(
    NominalDuplicatePenaltyMixin,
    _qHeteroMultiOutputBinaryPredictiveEntropy,
):
    pass


class qHeteroMultiOutputBinaryProbabilityVariance(
    NominalDuplicatePenaltyMixin,
    _qHeteroMultiOutputBinaryProbabilityVariance,
):
    pass


class qHeteroMultiOutputBinaryMarginUncertainty(
    NominalDuplicatePenaltyMixin,
    _qHeteroMultiOutputBinaryMarginUncertainty,
):
    pass


class qHeteroMultiOutputBinaryBALD(
    NominalDuplicatePenaltyMixin,
    _qHeteroMultiOutputBinaryBALD,
):
    pass


class qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy(
    NominalDuplicatePenaltyMixin,
    _qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy,
):
    pass


__all__ = [name for name in globals() if name.startswith("q")]
