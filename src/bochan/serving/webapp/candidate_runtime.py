"""Request-local candidate execution defaults for the Web workbench."""

from __future__ import annotations

import copy
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    """Return a shallow mapping for Pydantic, dict, or namespace values."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def _normalized_name(value: Any) -> str:
    """Normalize public runtime names for stable comparisons."""

    return str(value or "").replace("-", "_").lower()


def uses_ngboost_joint_batch(request: Any) -> bool:
    """Return whether Web NGBoost should optimize the requested q-batch jointly.

    NGBoost prediction is Python / estimator-call heavy. Sequential evolutionary
    optimization repeats the complete GA loop once for every requested candidate.
    A joint q-batch keeps the same q-acquisition semantics while allowing each
    ensemble member to predict all q points in one vectorized estimator call.

    The policy is deliberately narrow: only Bayesian-optimization requests using
    NGBoost with the GA/evolutionary backend and q > 1 are changed. Active
    learning, level-set estimation, other optimizers, and other model families
    retain the user's sequential setting.
    """

    if _normalized_name(getattr(request, "model_type", None)) != "ngboost_ensemble":
        return False

    optimizer = getattr(request, "optimizer", None)
    optimizer_name = _normalized_name(_mapping(optimizer).get("name"))
    if optimizer_name not in {"ga", "evo", "optimize_acqf_evo"}:
        return False

    optimizer_values = _mapping(optimizer)
    try:
        q = int(optimizer_values.get("q", 1))
    except (TypeError, ValueError):
        return False
    if q <= 1 or not bool(optimizer_values.get("sequential", True)):
        return False

    acquisition = _mapping(getattr(request, "acquisition", None))
    acqf_kwargs = _mapping(acquisition.get("acqf_kwargs"))
    family = _normalized_name(acqf_kwargs.get("web_family", "bayesian_optimization"))
    return family == "bayesian_optimization"


def apply_web_candidate_runtime_defaults(request: Any) -> Any:
    """Return an execution copy with Web-only candidate runtime defaults applied.

    The input request is never mutated. For qualifying NGBoost GA requests the
    returned request uses ``optimizer.sequential=False`` so the evolutionary
    backend receives one joint ``q`` optimization problem.
    """

    if not uses_ngboost_joint_batch(request):
        return request

    optimizer = getattr(request, "optimizer", None)
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
]
