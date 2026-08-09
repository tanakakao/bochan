"""Web-only runtime defaults for expensive surrogate models.

The core model implementations intentionally keep their library-facing defaults.
The Web workbench has a different requirement: interactive model fitting and
candidate suggestion must finish with practical latency. This module applies
request-local defaults without mutating model classes or estimator libraries.
"""

from __future__ import annotations

from typing import Any

_NGBOOST_ENSEMBLE_MODEL = "ngboost_ensemble"
_NGBOOST_WEB_ENSEMBLE_SIZE = 3
_TABPFN_MODEL = "tabpfn"
_TABPFN_WEB_N_ESTIMATORS = 4


def apply_web_model_runtime_defaults(
    model_kwargs: dict[str, Any],
    *,
    model_type: str,
    fit_maxiter: int,
) -> dict[str, Any]:
    """Apply interactive Web defaults while preserving explicit model kwargs.

    NGBoost's estimator count is a constructor argument rather than a generic
    ``FitConfig`` option. Therefore the Web ``fit_maxiter`` control is translated
    explicitly for NGBoost. Its ensemble is also intentionally smaller than the
    core model default because every member is a complete NGBoost fit and is
    evaluated again during acquisition optimization.

    TabPFN already performs its own in-context ensemble inference. Web candidate
    optimization invokes prediction many times, so use fewer TabPFN ensemble
    members by default while retaining estimator-side automatic feature coverage.
    Explicit user model kwargs always override these Web defaults. ``fit_maxiter``
    is deliberately not mapped to TabPFN because it has no iteration semantics in
    the foundation-model estimator.
    """

    kwargs = dict(model_kwargs)
    normalized_model_type = str(model_type).lower()

    if normalized_model_type == _TABPFN_MODEL:
        kwargs.setdefault("n_estimators", _TABPFN_WEB_N_ESTIMATORS)
        kwargs.setdefault("show_progress_bar", False)
        kwargs.setdefault("n_preprocessing_jobs", 1)
        return kwargs

    if normalized_model_type != _NGBOOST_ENSEMBLE_MODEL:
        return kwargs

    iterations = int(fit_maxiter)
    if iterations <= 0:
        raise ValueError("fit_maxiter must be positive for NGBoost Web fitting.")

    kwargs.setdefault("ensemble_size", _NGBOOST_WEB_ENSEMBLE_SIZE)
    kwargs.setdefault("n_estimators", iterations)
    kwargs.setdefault("verbose", False)
    return kwargs


__all__ = ["apply_web_model_runtime_defaults"]
