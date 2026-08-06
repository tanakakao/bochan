"""Regression active-learning acquisition functions.

The implementation is split by responsibility; this module preserves the public
and internal import surface used by regression, level-set, and non-Gaussian code.
"""

from ._base import _RegressionActiveLearningBase
from ._base_common import (
    OutputReductionType,
    ReductionType,
    _ensure_q_batch,
    _reduce,
    _safe_prod,
)
from ._integrated import (
    qRegressionIntegratedPosteriorVarianceProxy,
    qRegressionNegIntegratedPosteriorVariance,
)
from ._pointwise import (
    qRegressionBALD,
    qRegressionPosteriorVariance,
    qRegressionPredictiveEntropy,
)

for _class in (
    _RegressionActiveLearningBase,
    qRegressionBALD,
    qRegressionPosteriorVariance,
    qRegressionPredictiveEntropy,
    qRegressionIntegratedPosteriorVarianceProxy,
    qRegressionNegIntegratedPosteriorVariance,
):
    _class.__module__ = __name__

__all__ = [
    "qRegressionPredictiveEntropy",
    "qRegressionBALD",
    "qRegressionPosteriorVariance",
    "qRegressionNegIntegratedPosteriorVariance",
    "qRegressionIntegratedPosteriorVarianceProxy",
    "_RegressionActiveLearningBase",
    "OutputReductionType",
    "ReductionType",
    "_ensure_q_batch",
    "_reduce",
    "_safe_prod",
]
