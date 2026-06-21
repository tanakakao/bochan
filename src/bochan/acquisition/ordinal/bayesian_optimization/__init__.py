from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init

from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalNormalScoreObjective,
    qHeteroMultiOutputOrdinalExpectedUtility,
    qHeteroMultiOutputOrdinalProbabilityOfImprovement,
    qHeteroMultiOutputOrdinalExpectedImprovement,
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalNParEGO,
)

from .hetero_single_output import (
    qHeteroOrdinalExpectedUtility,
    qHeteroOrdinalExpectedImprovement,
    qHeteroOrdinalProbabilityOfImprovement,
    qHeteroOrdinalExpectedUtilityUpperConfidenceBound,
)

# Correlated Kronecker posteriors cannot use BoTorch's cached-Cholesky qNEHVI
# path. Patch the class in-place before exporting it so package-level imports and
# direct ``...multi_output`` imports share the same model-aware default.
patch_nehvi_cache_root_init(
    _multi_output.qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
)

from .multi_output import (
    qMultiOutputOrdinalUtilityObjective,
    qMultiOutputOrdinalExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNParEGO,
    compute_observed_ordinal_utility,
)

from .single_output import (
    qOrdinalProbabilityOfFeasibility,
    compute_ordinal_expected_utility_best_f,
)
from .utility_acquisitions import (
    OrdinalQBatchMode,
    OrdinalQReduction,
    qOrdinalExpectedUtility,
    qOrdinalExpectedImprovement,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)

__all__ = [
    "OrdinalQBatchMode",
    "OrdinalQReduction",
    "qHeteroMultiOutputOrdinalNormalScoreObjective",
    "qHeteroMultiOutputOrdinalExpectedUtility",
    "qHeteroMultiOutputOrdinalProbabilityOfImprovement",
    "qHeteroMultiOutputOrdinalExpectedImprovement",
    "qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement",
    "qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputOrdinalNParEGO",
    "qHeteroOrdinalExpectedUtility",
    "qHeteroOrdinalExpectedImprovement",
    "qHeteroOrdinalProbabilityOfImprovement",
    "qHeteroOrdinalExpectedUtilityUpperConfidenceBound",
    "qMultiOutputOrdinalUtilityObjective",
    "qMultiOutputOrdinalExpectedHypervolumeImprovement",
    "qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
    "qMultiOutputOrdinalNParEGO",
    "compute_observed_ordinal_utility",
    "qOrdinalExpectedUtility",
    "qOrdinalExpectedImprovement",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
    "qOrdinalProbabilityOfFeasibility",
    "compute_ordinal_expected_utility_best_f",
]
