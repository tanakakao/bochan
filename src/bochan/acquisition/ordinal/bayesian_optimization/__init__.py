from functools import wraps

import torch

from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init
from bochan.acquisition._nparego_shape import (
    reduce_nparego_sample_and_q_to_tbatch,
)

from . import multi_output as _multi_output
from ._utility_defaults import infer_multioutput_ordinal_utility_values
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalExpectedImprovement,
    qHeteroMultiOutputOrdinalExpectedUtility,
    qHeteroMultiOutputOrdinalNormalScoreObjective,
    qHeteroMultiOutputOrdinalProbabilityOfImprovement,
)
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement as _qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
)
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalNParEGO as _qHeteroMultiOutputOrdinalNParEGO,
)
from .hetero_single_output import (
    qHeteroOrdinalExpectedImprovement,
    qHeteroOrdinalExpectedUtility,
    qHeteroOrdinalExpectedUtilityUpperConfidenceBound,
    qHeteroOrdinalProbabilityOfImprovement,
)
from .knowledge_gradient import qOrdinalKnowledgeGradient

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
    compute_observed_ordinal_utility,
    qMultiOutputOrdinalUtilityObjective,
)
from .multi_output import (
    qMultiOutputOrdinalExpectedHypervolumeImprovement as _qMultiOutputOrdinalExpectedHypervolumeImprovement,
)
from .multi_output import (
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement as _qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
)
from .multi_output import (
    qMultiOutputOrdinalNParEGO as _qMultiOutputOrdinalNParEGO,
)

_NPAREGO_TBATCH_SHAPE_ERROR = (
    "Expected scalarized NParEGO values to end in q or, for q=1, "
    "to end in the t-batch shape."
)


def _with_default_utility_values(model, utility_values):
    if utility_values is not None:
        return utility_values
    return infer_multioutput_ordinal_utility_values(model)


def _complete_ordinal_baseline_rows(train_Y):
    """Return rows with every ordinal output observed for scalarization."""

    tensor = torch.as_tensor(train_Y)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(
            "Ordinal multi-output baseline labels must have shape [n, m]. "
            f"Got shape={tuple(tensor.shape)}."
        )
    if not tensor.is_floating_point():
        return tensor

    finite = torch.isfinite(tensor)
    complete = finite.all(dim=-1)
    if bool(complete.any()):
        return tensor[complete]

    observed_counts = finite.sum(dim=0).detach().cpu().tolist()
    raise ValueError(
        "Ordinal NParEGO requires at least one training row with every output "
        "observed to construct its baseline scalarization. Partially observed "
        f"rows remain usable for model fitting. Observed counts per output={observed_counts}."
    )


def _infer_multioutput_ordinal_train_y(model):
    """Infer complete raw wide labels without mistaking long targets for outputs."""

    expected_outputs = getattr(model, "num_tasks", None)
    if expected_outputs is None:
        expected_outputs = getattr(model, "num_outputs", None)
    try:
        expected_outputs = None if expected_outputs is None else int(expected_outputs)
    except (TypeError, ValueError):
        expected_outputs = None

    for name in ("train_Y_wide", "train_Y", "train_targets"):
        value = getattr(model, name, None)
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 2:
            continue
        if expected_outputs is not None and tensor.shape[-1] != expected_outputs:
            continue
        return _complete_ordinal_baseline_rows(tensor)

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
    return _complete_ordinal_baseline_rows(torch.cat(columns, dim=-1))


class _TBatchSafeMultiOutputOrdinalNParEGO(_qMultiOutputOrdinalNParEGO):
    """Ordinal NParEGO fallback for Kronecker t-batch / latent-rank collisions.

    LMC Kronecker posteriors normally preserve optimizer t-batches. During
    batched L-BFGS-B, however, an optimizer sub-batch can have the same size as
    the latent rank. In that ambiguous case GPyTorch may consume the optimizer
    batch as the latent batch and return samples without the t-batch dimension.

    The acquisition must not broadcast that single value across candidates,
    because each optimizer row needs its own value and gradient. Instead, only
    after the specific shape failure, evaluate each t-batch row independently
    and stack the scalar acquisition values back into the original batch shape.
    """

    def _evaluate_single_tbatch(self, X_single: torch.Tensor) -> torch.Tensor:
        """Evaluate one ``q x d`` design batch without an optimizer t-batch."""
        posterior = self.model.posterior(X_single)
        samples = self.get_posterior_samples(posterior)
        values = self.base_objective(samples, X=X_single)
        scalarized = self._scalarize(values)
        improvement = (
            scalarized - self.best_value.to(scalarized)
        ).clamp_min(0.0)
        return reduce_nparego_sample_and_q_to_tbatch(improvement, X_single)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        try:
            return super().forward(X)
        except RuntimeError as err:
            if _NPAREGO_TBATCH_SHAPE_ERROR not in str(err):
                raise

            batch_shape = X.shape[:-2]
            if len(batch_shape) == 0:
                raise

            q, d = int(X.shape[-2]), int(X.shape[-1])
            X_flat = X.reshape(-1, q, d)
            values = [self._evaluate_single_tbatch(X_i) for X_i in X_flat]
            return torch.stack(values).reshape(batch_shape)


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
    binary NParEGO behavior by accepting the support argument but not
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
    if kwargs.get("train_Y") is not None:
        kwargs["train_Y"] = _complete_ordinal_baseline_rows(kwargs["train_Y"])
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
    """Construct ordinal NParEGO with high-level API support.

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
    if train_Y is not None:
        train_Y = _complete_ordinal_baseline_rows(train_Y)
    return _TBatchSafeMultiOutputOrdinalNParEGO(
        model=model,
        X_baseline=X_baseline,
        ref_point=ref_point,
        utility_values=_with_default_utility_values(model, utility_values),
        train_Y=train_Y,
        **kwargs,
    )


from .single_output import (
    compute_ordinal_expected_utility_best_f,
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
    "qOrdinalKnowledgeGradient",
    "qOrdinalProbabilityOfFeasibility",
    "compute_ordinal_expected_utility_best_f",
]
