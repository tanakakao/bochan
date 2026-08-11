"""Ordinal active-learning acquisitions with nominal duplicate semantics."""

from functools import wraps

from bochan.acquisition._nominal_duplicate_penalties import (
    NominalDuplicatePenaltyMixin,
)
from bochan.acquisition.ordinal._wide import adapt_wide_ordinal_model

from .bald import qMultiOutputOrdinalBALD as _qMultiOutputOrdinalBALD
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalBALD as _qHeteroMultiOutputOrdinalBALD,
    qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy as _qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy,
    qHeteroMultiOutputOrdinalMarginUncertainty as _qHeteroMultiOutputOrdinalMarginUncertainty,
    qHeteroMultiOutputOrdinalPredictiveEntropy as _qHeteroMultiOutputOrdinalPredictiveEntropy,
    qHeteroMultiOutputOrdinalUtilityVariance as _qHeteroMultiOutputOrdinalUtilityVariance,
)
from .hetero_single_output import (
    qHeteroOrdinalBALD as _qHeteroOrdinalBALD,
    qHeteroOrdinalIntegratedPosteriorVariance as _qHeteroOrdinalIntegratedPosteriorVariance,
    qHeteroOrdinalMarginUncertainty as _qHeteroOrdinalMarginUncertainty,
    qHeteroOrdinalPredictiveEntropy as _qHeteroOrdinalPredictiveEntropy,
    qHeteroOrdinalUtilityVariance as _qHeteroOrdinalUtilityVariance,
)
from .multi_output import (
    qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance as _qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance,
    qMultiOutputOrdinalMarginUncertainty as _qMultiOutputOrdinalMarginUncertainty,
    qMultiOutputOrdinalPredictiveEntropy as _qMultiOutputOrdinalPredictiveEntropy,
    qMultiOutputOrdinalUtilityVariance as _qMultiOutputOrdinalUtilityVariance,
)
from .single_output import (
    qOrdinalBALD as _qOrdinalBALD,
    qOrdinalFantasyNegIntegratedPosteriorVariance as _qOrdinalFantasyNegIntegratedPosteriorVariance,
    qOrdinalMarginUncertainty as _qOrdinalMarginUncertainty,
    qOrdinalPredictiveEntropy as _qOrdinalPredictiveEntropy,
    qOrdinalUtilityVariance as _qOrdinalUtilityVariance,
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


qOrdinalPredictiveEntropy = _safe("qOrdinalPredictiveEntropy", _qOrdinalPredictiveEntropy)
qOrdinalBALD = _safe("qOrdinalBALD", _qOrdinalBALD)
qOrdinalUtilityVariance = _safe("qOrdinalUtilityVariance", _qOrdinalUtilityVariance)
qOrdinalMarginUncertainty = _safe("qOrdinalMarginUncertainty", _qOrdinalMarginUncertainty)
qOrdinalFantasyNegIntegratedPosteriorVariance = _safe(
    "qOrdinalFantasyNegIntegratedPosteriorVariance",
    _qOrdinalFantasyNegIntegratedPosteriorVariance,
)

qMultiOutputOrdinalPredictiveEntropy = _safe(
    "qMultiOutputOrdinalPredictiveEntropy",
    _qMultiOutputOrdinalPredictiveEntropy,
)
qMultiOutputOrdinalBALD = _safe("qMultiOutputOrdinalBALD", _qMultiOutputOrdinalBALD)
qMultiOutputOrdinalUtilityVariance = _safe(
    "qMultiOutputOrdinalUtilityVariance",
    _qMultiOutputOrdinalUtilityVariance,
)
qMultiOutputOrdinalMarginUncertainty = _safe(
    "qMultiOutputOrdinalMarginUncertainty",
    _qMultiOutputOrdinalMarginUncertainty,
)
qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance = _safe(
    "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance",
    _qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance,
)
qMultiOutputOrdinalIntegratedPosteriorVarianceProxy = (
    qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance
)

qHeteroOrdinalPredictiveEntropy = _safe(
    "qHeteroOrdinalPredictiveEntropy",
    _qHeteroOrdinalPredictiveEntropy,
)
qHeteroOrdinalUtilityVariance = _safe(
    "qHeteroOrdinalUtilityVariance",
    _qHeteroOrdinalUtilityVariance,
)
qHeteroOrdinalMarginUncertainty = _safe(
    "qHeteroOrdinalMarginUncertainty",
    _qHeteroOrdinalMarginUncertainty,
)
qHeteroOrdinalBALD = _safe("qHeteroOrdinalBALD", _qHeteroOrdinalBALD)
qHeteroOrdinalIntegratedPosteriorVariance = _safe(
    "qHeteroOrdinalIntegratedPosteriorVariance",
    _qHeteroOrdinalIntegratedPosteriorVariance,
)

qHeteroMultiOutputOrdinalPredictiveEntropy = _safe(
    "qHeteroMultiOutputOrdinalPredictiveEntropy",
    _qHeteroMultiOutputOrdinalPredictiveEntropy,
)
qHeteroMultiOutputOrdinalUtilityVariance = _safe(
    "qHeteroMultiOutputOrdinalUtilityVariance",
    _qHeteroMultiOutputOrdinalUtilityVariance,
)
qHeteroMultiOutputOrdinalMarginUncertainty = _safe(
    "qHeteroMultiOutputOrdinalMarginUncertainty",
    _qHeteroMultiOutputOrdinalMarginUncertainty,
)
qHeteroMultiOutputOrdinalBALD = _safe(
    "qHeteroMultiOutputOrdinalBALD",
    _qHeteroMultiOutputOrdinalBALD,
)
qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy = _safe(
    "qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy",
    _qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy,
)


__all__ = [name for name in globals() if name.startswith("q")]
