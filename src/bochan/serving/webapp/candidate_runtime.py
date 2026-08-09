"""Request-local candidate execution defaults for the Web workbench."""

from __future__ import annotations

import copy
from typing import Any

_TREE_ENSEMBLE_MODELS = frozenset(
    {
        "random_forest",
        "lightgbm_ensemble",
        "ngboost_ensemble",
    }
)
_GA_OPTIMIZERS = frozenset({"ga", "evo", "optimize_acqf_evo"})
_NGBOOST_EXTRA_JOINT_OPTIMIZERS = frozenset({"pso", "sa"})
_JOINT_BATCH_MAX_Q = 3


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field without serializing nested Pydantic model values."""

    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_name(value: Any) -> str:
    """Normalize public runtime names for stable comparisons."""

    return str(value or "").replace("-", "_").lower()


def _joint_optimizer_names(model_type: str) -> frozenset[str]:
    """Return evolutionary optimizers safe and useful for automatic joint q-batches."""

    if model_type == "ngboost_ensemble":
        return _GA_OPTIMIZERS | _NGBOOST_EXTRA_JOINT_OPTIMIZERS
    return _GA_OPTIMIZERS


def uses_tree_ensemble_joint_batch(request: Any) -> bool:
    """Return whether Web should optimize a tree-ensemble q-batch jointly.

    Joint evolutionary optimization reduces repeated Python / estimator-call
    round trips for tree ensembles while preserving the native q-acquisition
    objective. The automatic policy is deliberately conservative:

    - Random Forest and LightGBM use joint batches for GA/evo only.
    - NGBoost additionally uses joint PSO and simulated annealing.
    - q is limited to 2-3 so the joint search dimension does not grow too far.
    - CMA-ES, q=1, q>3, Active Learning, and level-set estimation retain the
      requested sequential behavior.
    """

    model_type = _normalized_name(_field(request, "model_type"))
    if model_type not in _TREE_ENSEMBLE_MODELS:
        return False

    optimizer = _field(request, "optimizer")
    optimizer_name = _normalized_name(_field(optimizer, "name"))
    if optimizer_name not in _joint_optimizer_names(model_type):
        return False

    try:
        q = int(_field(optimizer, "q", 1))
    except (TypeError, ValueError):
        return False
    if not 1 < q <= _JOINT_BATCH_MAX_Q:
        return False
    if not bool(_field(optimizer, "sequential", True)):
        return False

    acquisition = _field(request, "acquisition")
    acqf_kwargs = _field(acquisition, "acqf_kwargs", {})
    family = _normalized_name(
        _field(acqf_kwargs, "web_family", "bayesian_optimization")
    )
    return family == "bayesian_optimization"


def uses_ngboost_joint_batch(request: Any) -> bool:
    """Backward-compatible NGBoost-specific joint-batch predicate."""

    if _normalized_name(_field(request, "model_type")) != "ngboost_ensemble":
        return False
    return uses_tree_ensemble_joint_batch(request)


def apply_web_candidate_runtime_defaults(request: Any) -> Any:
    """Return an execution copy with Web-only candidate runtime defaults applied.

    The input request is never mutated. Qualifying tree-ensemble evolutionary
    requests use ``optimizer.sequential=False`` so the backend receives one joint
    q-batch optimization problem.
    """

    if not uses_tree_ensemble_joint_batch(request):
        return request

    optimizer = _field(request, "optimizer")
    if hasattr(optimizer, "model_copy"):
        resolved_optimizer = optimizer.model_copy(update={"sequential": False})
    elif isinstance(optimizer, dict):
        resolved_optimizer = dict(optimizer)
        resolved_optimizer["sequential"] = False
    else:
        resolved_optimizer = copy.copy(optimizer)
        resolved_optimizer.sequential = False

    if hasattr(request, "model_copy"):
        return request.model_copy(update={"optimizer": resolved_optimizer})
    if isinstance(request, dict):
        resolved = dict(request)
        resolved["optimizer"] = resolved_optimizer
        return resolved

    resolved = copy.copy(request)
    resolved.optimizer = resolved_optimizer
    return resolved


__all__ = [
    "apply_web_candidate_runtime_defaults",
    "uses_ngboost_joint_batch",
    "uses_tree_ensemble_joint_batch",
]
