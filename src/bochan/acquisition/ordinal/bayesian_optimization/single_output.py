from __future__ import annotations

from . import single_output_removed_api as _old
from .single_output_removed_api import *
from .single_output_removed_api import (
    _apply_input_transform_for_pending,
    _cat_dims_from_model,
    _coerce_pending_to_tensor,
    _mean_over_sample_dims,
    _normalize_utility_samples,
    _pairwise_distance2,
    _resolve_observed_X,
    compute_ordinal_expected_utility_best_f,
    ensure_q_batch,
    make_fixed_features_list,
    qOrdinalProbabilityOfFeasibility,
)
from .utility_acquisitions import (
    OrdinalQBatchMode,
    OrdinalQReduction,
    qOrdinalExpectedImprovement,
    qOrdinalExpectedUtility,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)

_REPLACED = {
    "qOrdinalExpectedUtility",
    "qOrdinalExpectedImprovement",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
}

__all__ = [
    name
    for name in getattr(_old, "__all__", [])
    if name not in _REPLACED
]
__all__ += [
    "OrdinalQBatchMode",
    "OrdinalQReduction",
    "qOrdinalExpectedUtility",
    "qOrdinalExpectedImprovement",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
]
