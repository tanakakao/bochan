"""Web-only runtime defaults for expensive surrogate models.

The core model implementations intentionally keep their library-facing defaults.
The Web workbench has a different requirement: interactive model fitting and
candidate suggestion must finish with practical latency.  This module applies
request-local defaults without mutating model classes or optimizer functions.
"""

from __future__ import annotations

from typing import Any

_NGBOOST_ENSEMBLE_MODEL = "ngboost_ensemble"
_NGBOOST_WEB_ENSEMBLE_SIZE = 5

_NGBOOST_WEB_EVO_OPTIONS: dict[str, dict[str, int]] = {
    "ga": {"pop_size": 32, "num_generations": 30},
    "pso": {"swarm_size": 32, "num_iterations": 30},
    "sa": {"sa_steps": 150},
    "cmaes": {"maxiter": 60},
}


def apply_web_model_runtime_defaults(
    model_kwargs: dict[str, Any],
    *,
    model_type: str,
    fit_maxiter: int,
) -> dict[str, Any]:
    """Apply interactive Web defaults while preserving explicit model kwargs.

    NGBoost's estimator count is a constructor argument rather than a generic
    ``FitConfig`` option.  Therefore the Web ``fit_maxiter`` control must be
    translated explicitly for NGBoost.  The ensemble is also intentionally
    smaller than the core model default because each member is a complete
    NGBoost fit.
    """

    kwargs = dict(model_kwargs)
    if str(model_type).lower() != _NGBOOST_ENSEMBLE_MODEL:
        return kwargs

    iterations = int(fit_maxiter)
    if iterations <= 0:
        raise ValueError("fit_maxiter must be positive for NGBoost Web fitting.")

    kwargs.setdefault("ensemble_size", _NGBOOST_WEB_ENSEMBLE_SIZE)
    kwargs.setdefault("n_estimators", iterations)
    kwargs.setdefault("verbose", False)
    return kwargs


def apply_web_optimizer_runtime_defaults(
    optimizer_kwargs: dict[str, Any],
    *,
    model_type: str,
    search_method: str,
) -> dict[str, Any]:
    """Reduce derivative-free search cost for the Web NGBoost ensemble.

    The generic evolutionary optimizer deliberately uses comparatively thorough
    defaults.  Those defaults are suitable for cheap differentiable / vectorized
    acquisitions, but a finite NGBoost ensemble evaluates several sklearn-backed
    predictors for every acquisition call.  The Web therefore uses a smaller
    interactive budget.  Explicit optimizer ``options`` always take precedence.
    """

    kwargs = dict(optimizer_kwargs)
    if str(model_type).lower() != _NGBOOST_ENSEMBLE_MODEL:
        return kwargs

    method = str(search_method or "").replace("-", "_").lower()
    if method in {"evo", "optimize_acqf_evo"}:
        method = str(kwargs.get("method", "ga")).replace("-", "_").lower()

    defaults = _NGBOOST_WEB_EVO_OPTIONS.get(method)
    if defaults is None:
        return kwargs

    options = dict(kwargs.get("options") or {})
    for key, value in defaults.items():
        options.setdefault(key, value)
    kwargs["options"] = options
    return kwargs


def web_runtime_metadata(
    *,
    model_type: str,
    model_kwargs: dict[str, Any],
    optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return concise effective runtime settings for Web diagnostics."""

    if str(model_type).lower() != _NGBOOST_ENSEMBLE_MODEL:
        return {}
    return {
        "model": {
            "ensemble_size": model_kwargs.get("ensemble_size"),
            "n_estimators": model_kwargs.get("n_estimators"),
        },
        "optimizer_options": dict(optimizer_kwargs.get("options") or {}),
    }


__all__ = [
    "apply_web_model_runtime_defaults",
    "apply_web_optimizer_runtime_defaults",
    "web_runtime_metadata",
]
