"""Ordinal level-set acquisitions with nominal duplicate semantics."""

from functools import wraps

from bochan.acquisition._nominal_duplicate_penalties import NominalDuplicatePenaltyMixin
from bochan.acquisition.ordinal._wide import adapt_wide_ordinal_model

from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalBoundaryVariance as _qHeteroMultiOutputOrdinalBoundaryVariance,
    qHeteroMultiOutputOrdinalLevelSetUncertainty as _qHeteroMultiOutputOrdinalLevelSetUncertainty,
    qHeteroMultiOutputOrdinalProbabilityOfExceedance as _qHeteroMultiOutputOrdinalProbabilityOfExceedance,
    qHeteroMultiOutputOrdinalStraddle as _qHeteroMultiOutputOrdinalStraddle,
)
from .hetero_single_output import (
    qHeteroOrdinalBoundaryVarianceAcquisition as _qHeteroOrdinalBoundaryVarianceAcquisition,
    qHeteroOrdinalClassEntropyAcquisition as _qHeteroOrdinalClassEntropyAcquisition,
    qHeteroOrdinalICUAcquisition as _qHeteroOrdinalICUAcquisition,
    qHeteroOrdinalLatentStraddleAcquisition as _qHeteroOrdinalLatentStraddleAcquisition,
)
from .multi_output import (
    qMultiOutputOrdinalBoundaryVarianceAcquisition as _qMultiOutputOrdinalBoundaryVarianceAcquisition,
    qMultiOutputOrdinalClassEntropyAcquisition as _qMultiOutputOrdinalClassEntropyAcquisition,
    qMultiOutputOrdinalICUAcquisition as _qMultiOutputOrdinalICUAcquisition,
    qMultiOutputOrdinalJointLatentStraddleAcquisition as _qMultiOutputOrdinalJointLatentStraddleAcquisition,
    qMultiOutputOrdinalLatentStraddleAcquisition as _qMultiOutputOrdinalLatentStraddleAcquisition,
)
from .single_output import (
    qOrdinalBoundaryVarianceAcquisition as _qOrdinalBoundaryVarianceAcquisition,
    qOrdinalClassEntropyAcquisition as _qOrdinalClassEntropyAcquisition,
    qOrdinalICUAcquisition as _qOrdinalICUAcquisition,
    qOrdinalJointLatentStraddleAcquisition as _qOrdinalJointLatentStraddleAcquisition,
    qOrdinalLatentStraddleAcquisition as _qOrdinalLatentStraddleAcquisition,
)


def _safe(name: str, base: type) -> type:
    @wraps(base.__init__)
    def __init__(self, model, *args, **kwargs) -> None:
        base.__init__(
            self,
            adapt_wide_ordinal_model(model),
            *args,
            **kwargs,
        )

    return type(
        name,
        (NominalDuplicatePenaltyMixin, base),
        {"__module__": __name__, "__init__": __init__},
    )


qOrdinalLatentStraddleAcquisition = _safe(
    "qOrdinalLatentStraddleAcquisition",
    _qOrdinalLatentStraddleAcquisition,
)
qOrdinalJointLatentStraddleAcquisition = _safe(
    "qOrdinalJointLatentStraddleAcquisition",
    _qOrdinalJointLatentStraddleAcquisition,
)
qOrdinalICUAcquisition = _safe("qOrdinalICUAcquisition", _qOrdinalICUAcquisition)
qOrdinalBoundaryVarianceAcquisition = _safe(
    "qOrdinalBoundaryVarianceAcquisition",
    _qOrdinalBoundaryVarianceAcquisition,
)
qOrdinalClassEntropyAcquisition = _safe(
    "qOrdinalClassEntropyAcquisition",
    _qOrdinalClassEntropyAcquisition,
)

qMultiOutputOrdinalLatentStraddleAcquisition = _safe(
    "qMultiOutputOrdinalLatentStraddleAcquisition",
    _qMultiOutputOrdinalLatentStraddleAcquisition,
)
qMultiOutputOrdinalJointLatentStraddleAcquisition = _safe(
    "qMultiOutputOrdinalJointLatentStraddleAcquisition",
    _qMultiOutputOrdinalJointLatentStraddleAcquisition,
)
qMultiOutputOrdinalICUAcquisition = _safe(
    "qMultiOutputOrdinalICUAcquisition",
    _qMultiOutputOrdinalICUAcquisition,
)
qMultiOutputOrdinalBoundaryVarianceAcquisition = _safe(
    "qMultiOutputOrdinalBoundaryVarianceAcquisition",
    _qMultiOutputOrdinalBoundaryVarianceAcquisition,
)
qMultiOutputOrdinalClassEntropyAcquisition = _safe(
    "qMultiOutputOrdinalClassEntropyAcquisition",
    _qMultiOutputOrdinalClassEntropyAcquisition,
)

qHeteroOrdinalLatentStraddleAcquisition = _safe(
    "qHeteroOrdinalLatentStraddleAcquisition",
    _qHeteroOrdinalLatentStraddleAcquisition,
)
qHeteroOrdinalICUAcquisition = _safe(
    "qHeteroOrdinalICUAcquisition",
    _qHeteroOrdinalICUAcquisition,
)
qHeteroOrdinalBoundaryVarianceAcquisition = _safe(
    "qHeteroOrdinalBoundaryVarianceAcquisition",
    _qHeteroOrdinalBoundaryVarianceAcquisition,
)
qHeteroOrdinalClassEntropyAcquisition = _safe(
    "qHeteroOrdinalClassEntropyAcquisition",
    _qHeteroOrdinalClassEntropyAcquisition,
)

qHeteroMultiOutputOrdinalProbabilityOfExceedance = _safe(
    "qHeteroMultiOutputOrdinalProbabilityOfExceedance",
    _qHeteroMultiOutputOrdinalProbabilityOfExceedance,
)
qHeteroMultiOutputOrdinalLevelSetUncertainty = _safe(
    "qHeteroMultiOutputOrdinalLevelSetUncertainty",
    _qHeteroMultiOutputOrdinalLevelSetUncertainty,
)
qHeteroMultiOutputOrdinalStraddle = _safe(
    "qHeteroMultiOutputOrdinalStraddle",
    _qHeteroMultiOutputOrdinalStraddle,
)
qHeteroMultiOutputOrdinalBoundaryVariance = _safe(
    "qHeteroMultiOutputOrdinalBoundaryVariance",
    _qHeteroMultiOutputOrdinalBoundaryVariance,
)


__all__ = [name for name in globals() if name.startswith("q")]
