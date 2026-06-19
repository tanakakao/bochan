from .nipv_compat import apply_multioutput_nipv_compat
from .hetero_multi_output import (
    qHeteroMultiOutputRegressionPredictiveEntropy,
    qHeteroMultiOutputRegressionBALD,
    qHeteroMultiOutputRegressionPosteriorVariance,
    qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy,
)

from .hetero_single_output import (
    qHeteroRegressionPredictiveEntropy,
    qHeteroRegressionBALD,
    qHeteroRegressionPosteriorVariance,
    qHeteroRegressionNegIntegratedPosteriorVariance,
    qHeteroRegressionIntegratedPosteriorVarianceProxy,
)

from .multi_output import (
    qMultiOutputRegressionPredictiveEntropy,
    qMultiOutputRegressionBALD,
    qMultiOutputRegressionPosteriorVariance,
    qMultiOutputRegressionNegIntegratedPosteriorVariance,
    qMultiOutputRegressionIntegratedPosteriorVarianceProxy,
)

from .single_output import (
    qRegressionPredictiveEntropy,
    qRegressionBALD,
    qRegressionPosteriorVariance,
    qRegressionNegIntegratedPosteriorVariance,
    qRegressionIntegratedPosteriorVarianceProxy,
)

# Multi-output / hybrid models may leave integration and output dimensions in
# BoTorch qNegIntegratedPosteriorVariance results. Reduce them to t-batch shape.
apply_multioutput_nipv_compat()

__all__ = [
    "apply_multioutput_nipv_compat",
    "qHeteroMultiOutputRegressionPredictiveEntropy",
    "qHeteroMultiOutputRegressionBALD",
    "qHeteroMultiOutputRegressionPosteriorVariance",
    "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
    "qHeteroRegressionPredictiveEntropy",
    "qHeteroRegressionBALD",
    "qHeteroRegressionPosteriorVariance",
    "qHeteroRegressionNegIntegratedPosteriorVariance",
    "qHeteroRegressionIntegratedPosteriorVarianceProxy",
    "qMultiOutputRegressionPredictiveEntropy",
    "qMultiOutputRegressionBALD",
    "qMultiOutputRegressionPosteriorVariance",
    "qMultiOutputRegressionNegIntegratedPosteriorVariance",
    "qMultiOutputRegressionIntegratedPosteriorVarianceProxy",
    "qRegressionPredictiveEntropy",
    "qRegressionBALD",
    "qRegressionPosteriorVariance",
    "qRegressionNegIntegratedPosteriorVariance",
    "qRegressionIntegratedPosteriorVarianceProxy",
]
