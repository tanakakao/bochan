"""Binary level-set acquisitions with nominal duplicate semantics."""

from bochan.acquisition._nominal_duplicate_penalties import (
    NominalDuplicatePenaltyMixin,
)

from .hetero_multi_output import (
    qHeteroMultiOutputBinaryBoundaryVarianceAcquisition as _qHeteroMultiOutputBinaryBoundaryVarianceAcquisition,
    qHeteroMultiOutputBinaryClassEntropyAcquisition as _qHeteroMultiOutputBinaryClassEntropyAcquisition,
    qHeteroMultiOutputBinaryICUAcquisition as _qHeteroMultiOutputBinaryICUAcquisition,
    qHeteroMultiOutputBinaryJointLatentStraddleAcquisition as _qHeteroMultiOutputBinaryJointLatentStraddleAcquisition,
    qHeteroMultiOutputBinaryLatentStraddleAcquisition as _qHeteroMultiOutputBinaryLatentStraddleAcquisition,
)
from .hetero_single_output import (
    qHeteroBinaryBoundaryVarianceAcquisition as _qHeteroBinaryBoundaryVarianceAcquisition,
    qHeteroBinaryClassEntropyAcquisition as _qHeteroBinaryClassEntropyAcquisition,
    qHeteroBinaryICUAcquisition as _qHeteroBinaryICUAcquisition,
    qHeteroBinaryLatentStraddleAcquisition as _qHeteroBinaryLatentStraddleAcquisition,
)
from .multi_output import (
    qMultiOutputBinaryBoundaryVarianceAcquisition as _qMultiOutputBinaryBoundaryVarianceAcquisition,
    qMultiOutputBinaryClassEntropyAcquisition as _qMultiOutputBinaryClassEntropyAcquisition,
    qMultiOutputBinaryICUAcquisition as _qMultiOutputBinaryICUAcquisition,
    qMultiOutputBinaryJointLatentStraddleAcquisition as _qMultiOutputBinaryJointLatentStraddleAcquisition,
    qMultiOutputBinaryLatentStraddleAcquisition as _qMultiOutputBinaryLatentStraddleAcquisition,
)
from .single_output import (
    qBinaryBoundaryVarianceAcquisition as _qBinaryBoundaryVarianceAcquisition,
    qBinaryClassEntropyAcquisition as _qBinaryClassEntropyAcquisition,
    qBinaryICUAcquisition as _qBinaryICUAcquisition,
    qBinaryJointLatentStraddleAcquisition as _qBinaryJointLatentStraddleAcquisition,
    qBinaryLatentStraddleAcquisition as _qBinaryLatentStraddleAcquisition,
)


def _safe(name: str, base: type) -> type:
    return type(name, (NominalDuplicatePenaltyMixin, base), {"__module__": __name__})


qBinaryLatentStraddleAcquisition = _safe(
    "qBinaryLatentStraddleAcquisition",
    _qBinaryLatentStraddleAcquisition,
)
qBinaryJointLatentStraddleAcquisition = _safe(
    "qBinaryJointLatentStraddleAcquisition",
    _qBinaryJointLatentStraddleAcquisition,
)
qBinaryICUAcquisition = _safe("qBinaryICUAcquisition", _qBinaryICUAcquisition)
qBinaryBoundaryVarianceAcquisition = _safe(
    "qBinaryBoundaryVarianceAcquisition",
    _qBinaryBoundaryVarianceAcquisition,
)
qBinaryClassEntropyAcquisition = _safe(
    "qBinaryClassEntropyAcquisition",
    _qBinaryClassEntropyAcquisition,
)

qMultiOutputBinaryLatentStraddleAcquisition = _safe(
    "qMultiOutputBinaryLatentStraddleAcquisition",
    _qMultiOutputBinaryLatentStraddleAcquisition,
)
qMultiOutputBinaryJointLatentStraddleAcquisition = _safe(
    "qMultiOutputBinaryJointLatentStraddleAcquisition",
    _qMultiOutputBinaryJointLatentStraddleAcquisition,
)
qMultiOutputBinaryClassEntropyAcquisition = _safe(
    "qMultiOutputBinaryClassEntropyAcquisition",
    _qMultiOutputBinaryClassEntropyAcquisition,
)
qMultiOutputBinaryICUAcquisition = _safe(
    "qMultiOutputBinaryICUAcquisition",
    _qMultiOutputBinaryICUAcquisition,
)
qMultiOutputBinaryBoundaryVarianceAcquisition = _safe(
    "qMultiOutputBinaryBoundaryVarianceAcquisition",
    _qMultiOutputBinaryBoundaryVarianceAcquisition,
)

qHeteroBinaryLatentStraddleAcquisition = _safe(
    "qHeteroBinaryLatentStraddleAcquisition",
    _qHeteroBinaryLatentStraddleAcquisition,
)
qHeteroBinaryICUAcquisition = _safe(
    "qHeteroBinaryICUAcquisition",
    _qHeteroBinaryICUAcquisition,
)
qHeteroBinaryBoundaryVarianceAcquisition = _safe(
    "qHeteroBinaryBoundaryVarianceAcquisition",
    _qHeteroBinaryBoundaryVarianceAcquisition,
)
qHeteroBinaryClassEntropyAcquisition = _safe(
    "qHeteroBinaryClassEntropyAcquisition",
    _qHeteroBinaryClassEntropyAcquisition,
)

qHeteroMultiOutputBinaryClassEntropyAcquisition = _safe(
    "qHeteroMultiOutputBinaryClassEntropyAcquisition",
    _qHeteroMultiOutputBinaryClassEntropyAcquisition,
)
qHeteroMultiOutputBinaryICUAcquisition = _safe(
    "qHeteroMultiOutputBinaryICUAcquisition",
    _qHeteroMultiOutputBinaryICUAcquisition,
)
qHeteroMultiOutputBinaryBoundaryVarianceAcquisition = _safe(
    "qHeteroMultiOutputBinaryBoundaryVarianceAcquisition",
    _qHeteroMultiOutputBinaryBoundaryVarianceAcquisition,
)
qHeteroMultiOutputBinaryLatentStraddleAcquisition = _safe(
    "qHeteroMultiOutputBinaryLatentStraddleAcquisition",
    _qHeteroMultiOutputBinaryLatentStraddleAcquisition,
)
qHeteroMultiOutputBinaryJointLatentStraddleAcquisition = _safe(
    "qHeteroMultiOutputBinaryJointLatentStraddleAcquisition",
    _qHeteroMultiOutputBinaryJointLatentStraddleAcquisition,
)


__all__ = [name for name in globals() if name.startswith("q")]
