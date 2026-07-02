from functools import wraps

from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init
from bochan.acquisition._nparego_shape import (
    reduce_nparego_sample_and_q_to_tbatch,
)

import torch

from . import multi_output as _multi_output
from ._utility_defaults import infer_multioutput_ordinal_utility_values
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalNormalScoreObjective,
    qHeteroMultiOutputOrdinalExpectedUtility,
    qHeteroMultiOutputOrdinalProbabilityOfImprovement,
    qHeteroMultiOutputOrdinalExpectedImprovement,
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement as _qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalNParEGO as _qHeteroMultiOutputOrdinalNParEGO,
)

from .hetero_single_output import (
    qHeteroOrdinalExpectedUtility,
    qHeteroOrdinalExpectedImprovement,
    qHeteroOrdinalProbabilityOfImprovement,
    qHeteroOrdinalExpectedUtilityUpperConfidenceBound,
)

# Keep q=1 sequential optimization shape handling aligned across classification
# and ordinal NParEGO implementations.
_multi_output._reduce_sample_and_q_to_tbatch = (
    reduce_nparego_sample_and_q_to_tbatch
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


def _infer_multioutput_ordinal_train_y(model):
    """Infer raw ordinal training labels from correlated or wrapper models."""
    for name in ("train_Y", "train_targets"):
        value = getattr(model, name, None)
        if value is not None:
            tensor = torch.as_tensor(value)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            return tensor

    submodels = getattr(model, "models", None)
    if submodels is None:
        return None

    columns = []
    for submodel in submodels:
        value = getattr(submodel, "train_Y", None)
        if value is None:
            value = getattr(submodel, "train_targets", None)
        if value is None:
            return None
        column = torch.as_tensor(value)
        if column.ndim == 1:
            column = column.unsqueeze(-1)
        elif column.shape[-1] != 1:
            return None
        columns.append(column)

    if not columns:
        return None
    return torch.cat(columns, dim=-1)


@wraps(_qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement)
def qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
    *args,
    Y_baseline=None,
    **kwargs,
):
    """Construct hetero ordinal NEHVI with a utility-space baseline.

    The high-level API may inject raw ordinal labels through ``Y_baseline``.
    Integer tensors are therefore treated as labels rather than precomputed
    utility values and are discarded so the underlying acquisition recomputes
    the heteroscedastic utility baseline from ``X_baseline``. Floating-point
    baselines remain supported as explicit utility-space overrides.
    """
    if Y_baseline is not None:
        baseline = torch.as_tensor(Y_baseline)
        if not baseline.is_floating_point():
            Y_baseline = None
    return _qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        *args,
        Y_baseline=Y_baseline,
        **kwargs,
    )


@wraps(_qHeteroMultiOutputOrdinalNParEGO)
def qHeteroMultiOutputOrdinalNParEGO(
    *args,
    objective=None,
    **kwargs,
):
    """Construct hetero ordinal NParEGO without double scalarization.

    NParEGO performs its own augmented Chebyshev scalarization after the
    heteroscedastic ordinal utility objective has produced one value per output.
    The generic high-level objective may already reduce the output dimension,
    which would remove the ``m`` axis before NParEGO can scalarize it. Match the
    binary NParEGO behavior by accepting the compatibility argument but not
    applying it inside the utility objective.
    """
    del objective
    return _qHeteroMultiOutputOrdinalNParEGO(
        *args,
        objective=None,
        **kwargs,
    )


def qMultiOutputOrdinalExpectedHypervolumeImprovement(
    model,
    ref_point,
    *,
    partitioning=None,
    utility_values=None,
    **kwargs,
):
    """Construct ordinal qEHVI with explicit context-facing parameters.

    ``ref_point`` and ``partitioning`` are kept in the public signature so the
    high-level API can retain automatically inferred context values when it
    filters keyword arguments by callable signature.
    """
    return _qMultiOutputOrdinalExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        utility_values=_with_default_utility_values(model, utility_values),
        **kwargs,
    )


def qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
    model,
    ref_point,
    X_baseline,
    *,
    utility_values=None,
    **kwargs,
):
    """Construct ordinal qNEHVI with explicit baseline and utility defaults.

    ``X_baseline`` must be explicit: the high-level engine filters automatic
    context fields against this signature before constructing the acquisition.
    """
    return _qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=X_baseline,
        utility_values=_with_default_utility_values(model, utility_values),
        **kwargs,
    )


def qMultiOutputOrdinalNParEGO(
    model,
    X_baseline,
    ref_point,
    *,
    utility_values=None,
    train_Y=None,
    best_f=None,
    **kwargs,
):
    """Construct ordinal NParEGO with high-level API compatibility.

    The generic acquisition-default resolver may inject ``best_f`` because
    NParEGO is EI-based. This implementation computes its own scalarized
    ``best_value`` from the baseline, matching the binary implementation, so the
    generic value is accepted and intentionally ignored.

    When neither ``train_Y`` nor an explicit utility-space ``Y_baseline`` is
    supplied, raw ordinal labels are recovered from the model. The underlying
    acquisition then performs the existing label-to-utility conversion.
    """
    del best_f
    if train_Y is None and kwargs.get("Y_baseline") is None:
        train_Y = _infer_multioutput_ordinal_train_y(model)
    return _qMultiOutputOrdinalNParEGO(
        model=model,
        X_baseline=X_baseline,
        ref_point=ref_point,
        utility_values=_with_default_utility_values(model, utility_values),
        train_Y=train_Y,
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
