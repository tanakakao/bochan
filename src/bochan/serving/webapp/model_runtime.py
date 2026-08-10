"""Web-only runtime defaults for expensive surrogate models.

The core model implementations intentionally keep their library-facing defaults.
The Web workbench has a different requirement: interactive model fitting and
candidate suggestion must finish with practical latency. This module applies
Web-process defaults without mutating model classes or estimator libraries.
"""

from __future__ import annotations

import os
from typing import Any

from .tabpfn_assets import require_preloaded_tabpfn_assets

_NGBOOST_ENSEMBLE_MODEL = "ngboost_ensemble"
_NGBOOST_WEB_ENSEMBLE_SIZE = 3
_TABPFN_MODEL = "tabpfn"
_TABPFN_WEB_N_ESTIMATORS = 4
_TABPFN_NO_BROWSER_ENV = "TABPFN_NO_BROWSER"


def _configure_tabpfn_web_environment() -> None:
    """Disable browser authentication inside the public Web server process.

    bochan Web uses a deployment-time preload contract: model checkpoints must
    already be present before the server handles user requests. Browser login is
    therefore never an acceptable runtime fallback, even if an operator happened
    to define ``TABPFN_NO_BROWSER=0`` in the server environment.
    """

    os.environ[_TABPFN_NO_BROWSER_ENV] = "1"


def _web_request_context_active() -> bool:
    """Return whether model defaults are being resolved inside a real Web run."""

    # Import lazily to avoid a module cycle: target_missing_policy imports this
    # module while resolving target settings inside its request-local context.
    from .target_missing_policy import current_target_missing_state

    return current_target_missing_state() is not None


def apply_web_model_runtime_defaults(
    model_kwargs: dict[str, Any],
    *,
    model_type: str,
    fit_maxiter: int,
) -> dict[str, Any]:
    """Apply interactive Web defaults while preserving explicit estimator injection.

    NGBoost's estimator count is a constructor argument rather than a generic
    ``FitConfig`` option. Therefore the Web ``fit_maxiter`` control is translated
    explicitly for NGBoost. Its ensemble is also intentionally smaller than the
    core model default because every member is a complete NGBoost fit and is
    evaluated again during acquisition optimization.

    TabPFN already performs its own in-context ensemble inference. Web candidate
    optimization invokes prediction many times, so use fewer TabPFN ensemble
    members by default while retaining estimator-side automatic feature coverage.

    During an actual Web request, official TabPFN v3 classifier/regressor
    checkpoints must be preloaded before model construction. Missing assets fail
    immediately before TabPFN can authenticate or download. The helper remains
    usable as a pure runtime-default resolver outside a Web request (for example in
    unit tests and configuration tooling), while Core / Notebook behavior is
    unchanged.

    Explicit injected ``estimator`` / ``estimators`` objects are already fully
    constructed and do not require bochan-managed runtime assets. Constructor-only
    Web defaults must not be added in that case.
    """

    kwargs = dict(model_kwargs)
    normalized_model_type = str(model_type).lower()

    if normalized_model_type == _TABPFN_MODEL:
        _configure_tabpfn_web_environment()
        if kwargs.get("estimator") is not None:
            return kwargs
        if _web_request_context_active():
            require_preloaded_tabpfn_assets()
        kwargs.setdefault("n_estimators", _TABPFN_WEB_N_ESTIMATORS)
        kwargs.setdefault("show_progress_bar", False)
        kwargs.setdefault("n_preprocessing_jobs", 1)
        return kwargs

    if normalized_model_type != _NGBOOST_ENSEMBLE_MODEL:
        return kwargs

    if kwargs.get("estimators") is not None:
        return kwargs

    iterations = int(fit_maxiter)
    if iterations <= 0:
        raise ValueError("fit_maxiter must be positive for NGBoost Web fitting.")

    kwargs.setdefault("ensemble_size", _NGBOOST_WEB_ENSEMBLE_SIZE)
    kwargs.setdefault("n_estimators", iterations)
    kwargs.setdefault("verbose", False)
    return kwargs


__all__ = ["apply_web_model_runtime_defaults"]
