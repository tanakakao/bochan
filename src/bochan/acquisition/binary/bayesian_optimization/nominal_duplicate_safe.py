"""Binary BO acquisitions that need nominal hard-duplicate semantics."""

from bochan.acquisition._nominal_duplicate_penalties import NominalDuplicatePenaltyMixin

from .multi_output import (
    qMultiOutputBinaryProbabilityOfFeasibility as _qMultiOutputBinaryProbabilityOfFeasibility,
)
from .single_output import (
    qBinaryProbabilityOfFeasibility as _qBinaryProbabilityOfFeasibility,
)


class qBinaryProbabilityOfFeasibility(
    NominalDuplicatePenaltyMixin,
    _qBinaryProbabilityOfFeasibility,
):
    """Binary PoF with hard duplicate identity evaluated on nominal candidates."""


class qMultiOutputBinaryProbabilityOfFeasibility(
    NominalDuplicatePenaltyMixin,
    _qMultiOutputBinaryProbabilityOfFeasibility,
):
    """Multi-output binary PoF with nominal hard duplicate semantics."""


__all__ = [
    "qBinaryProbabilityOfFeasibility",
    "qMultiOutputBinaryProbabilityOfFeasibility",
]
