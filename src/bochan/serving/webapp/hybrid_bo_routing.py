"""Preserve Web Hybrid objective semantics for single-objective BO."""

from __future__ import annotations

import copy
from functools import wraps
from typing import Any

_INSTALLED = False


def _normalize_acquisition_name(name: str) -> str:
    """Return a separator-free acquisition identifier."""

    return (
        str(name)
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )


def _generic_single_objective_bo_acqf_cls(name: str) -> Any | None:
    """Resolve generic BoTorch BO acquisitions operating on Hybrid objectives."""

    normalized = _normalize_acquisition_name(name)
    from botorch.acquisition.monte_carlo import (
        qExpectedImprovement,
        qNoisyExpectedImprovement,
        qProbabilityOfImprovement,
        qUpperConfidenceBound,
    )

    if normalized in {
        "ei",
        "qei",
        "expectedimprovement",
        "qexpectedimprovement",
    }:
        return qExpectedImprovement
    if normalized in {
        "nei",
        "qnei",
        "noisyexpectedimprovement",
        "qnoisyexpectedimprovement",
    }:
        return qNoisyExpectedImprovement
    if normalized in {
        "pi",
        "qpi",
        "probabilityofimprovement",
        "qprobabilityofimprovement",
    }:
        return qProbabilityOfImprovement
    if normalized in {
        "ucb",
        "qucb",
        "upperconfidencebound",
        "qupperconfidencebound",
    }:
        return qUpperConfidenceBound
    return None


class _WebHybridObjectiveBOResolver:
    """Pin a Web BO acquisition before task-aware submodel routing unwraps Hybrid."""

    def _resolve_acqf_kwargs(
        self,
        *,
        name: str,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        acqf_cls = _generic_single_objective_bo_acqf_cls(name)
        if acqf_cls is None:
            return kwargs, None

        resolved = dict(kwargs)
        # AcquisitionConfig's structured resolver hook is carried by the
        # ``thresholds`` entry. It is a transport-only value here and must never
        # reach the actual BoTorch acquisition constructor.
        resolved.pop("thresholds", None)
        return resolved, acqf_cls


_WEB_HYBRID_OBJECTIVE_BO_RESOLVER = _WebHybridObjectiveBOResolver()


def _copy_with_update(value: Any, **updates: Any) -> Any:
    """Copy Pydantic models and lightweight namespaces without mutating requests."""

    if hasattr(value, "model_copy"):
        return value.model_copy(update=updates)
    cloned = copy.copy(value)
    for key, item in updates.items():
        setattr(cloned, key, item)
    return cloned


def _uses_single_hybrid_objective(request: Any) -> bool:
    """Return whether Web fitting will use Hybrid with one optimized output."""

    from . import workflows_tabular
    from .target_roles import apply_target_roles, optimized_targets

    if workflows_tabular._resolve_direct_multitask_model_type(str(request.model_type)) is not None:
        return False

    target_columns, directions = workflows_tabular._resolve_targets(request)
    target_settings, model_kwargs = workflows_tabular._resolve_target_settings(
        request,
        target_columns=target_columns,
        directions=directions,
    )
    target_settings, _ = apply_target_roles(
        target_settings,
        model_kwargs,
        directions=directions,
    )
    return len(optimized_targets(target_settings)) == 1


def prepare_hybrid_objective_bo_request(request: Any) -> Any:
    """Attach a request-local generic BO resolver for single Hybrid objectives.

    The Web workbench intentionally represents classification and ordinal targets
    through ``HybridMultiOutputModel`` so selected classes, utility values, target
    directions, and target-value transforms are defined once by ``OutputSpec``.
    Task-aware short-name routing would otherwise unwrap a one-output Hybrid model
    and select binary / ordinal / multiclass acquisitions that operate on the raw
    submodel, bypassing those Web objective semantics.
    """

    acquisition = request.acquisition
    acqf_kwargs = dict(getattr(acquisition, "acqf_kwargs", None) or {})
    family = str(acqf_kwargs.get("web_family", "bayesian_optimization")).lower()
    if family != "bayesian_optimization":
        return request
    if _generic_single_objective_bo_acqf_cls(str(acquisition.name)) is None:
        return request
    if not _uses_single_hybrid_objective(request):
        return request

    # Respect an explicit structured resolver if a caller supplied one. Normal
    # Web BO requests do not use ``thresholds``; level-set requests are excluded
    # by the family check above.
    if "thresholds" in acqf_kwargs:
        return request

    acqf_kwargs["thresholds"] = _WEB_HYBRID_OBJECTIVE_BO_RESOLVER
    acquisition = _copy_with_update(acquisition, acqf_kwargs=acqf_kwargs)
    return _copy_with_update(request, acquisition=acquisition)


def install_web_hybrid_objective_bo_routing() -> None:
    """Install request-local Hybrid objective routing before workflows are bound."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import workflows_tabular

    original_workflow = workflows_tabular.run_regression_web_workflow

    @wraps(original_workflow)
    def workflow_adapter(request: Any, store: Any) -> dict[str, Any]:
        return original_workflow(
            prepare_hybrid_objective_bo_request(request),
            store,
        )

    workflows_tabular.run_regression_web_workflow = workflow_adapter
    _INSTALLED = True


__all__ = [
    "install_web_hybrid_objective_bo_routing",
    "prepare_hybrid_objective_bo_request",
]
