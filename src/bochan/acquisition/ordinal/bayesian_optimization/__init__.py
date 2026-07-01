from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init

from . import multi_output as _multi_output
from ._utility_defaults import infer_multioutput_ordinal_utility_values
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
    qMultiOutputOrdinalExpectedHypervolumeImprovement as _qMultiOutputOrdinalExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement as _qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNParEGO as _qMultiOutputOrdinalNParEGO,
    compute_observed_ordinal_utility,
)


def _with_default_utility_values(model, utility_values):
    if utility_values is not None:
        return utility_values
    return infer_multioutput_ordinal_utility_values(model)


def qMultiOutputOrdinalExpectedHypervolumeImprovement(
    model,
    *args,
    utility_values=None,
    **kwargs,
):
    """Construct ordinal qEHVI with inferred utility values when omitted."""
    return _qMultiOutputOrdinalExpectedHypervolumeImprovement(
        model,
        *args,
        utility_values=_with_default_utility_values(model, utility_values),
        **kwargs,
    )


def qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
    model,
    *args,
    utility_values=None,
    **kwargs,
):
    """Construct ordinal qNEHVI with inferred utility values when omitted."""
    return _qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model,
        *args,
        utility_values=_with_default_utility_values(model, utility_values),
        **kwargs,
    )


def qMultiOutputOrdinalNParEGO(
    model,
    *args,
    utility_values=None,
    **kwargs,
):
    """Construct ordinal NParEGO with inferred utility values when omitted."""
    return _qMultiOutputOrdinalNParEGO(
        model,
        *args,
        utility_values=_with_default_utility_values(model, utility_values),
        **kwargs,
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
