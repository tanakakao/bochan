"""Web-only runtime defaults for expensive surrogate models.

The core model implementations intentionally keep their library-facing defaults.
The Web workbench has a different requirement: interactive model fitting and
candidate suggestion must finish with practical latency. This module applies
Web-process defaults without mutating model classes or estimator libraries.
"""

from __future__ import annotations

import os
from typing import Any

_NGBOOST_ENSEMBLE_MODEL = "ngboost_ensemble"
_NGBOOST_WEB_ENSEMBLE_SIZE = 3
_TABPFN_MODEL = "tabpfn"
_TABPFN_WEB_N_ESTIMATORS = 4
_TABPFN_NO_BROWSER_ENV = "TABPFN_NO_BROWSER"


def _configure_tabpfn_web_environment() -> None:
    """Prevent library-managed browser login inside the Web server process.

    TabPFN performs a one-time license/authentication flow when model weights are
    not yet available. Its default graphical flow opens a browser and starts a
    loopback callback listener. That behavior is appropriate for an interactive
    Python session, but not for a Web backend handling a model-fit request.

    The official TabPFN non-interactive contract is ``TABPFN_NO_BROWSER`` plus
    ``TABPFN_TOKEN`` (or previously cached credentials/model weights). Use
    ``setdefault`` so an operator can explicitly opt back into TabPFN's browser
    flow by defining the environment variable before starting bochan.
    """

    os.environ.setdefault(_TABPFN_NO_BROWSER_ENV, "1")


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
    The Web runtime also disables TabPFN's automatic browser authentication flow;
    operators should authenticate non-interactively with ``TABPFN_TOKEN`` or use
    already cached credentials/model weights. Core / Notebook behavior is not
    changed because this helper is only called by the Web workbench.

    Explicit user model kwargs always override these Web defaults. ``fit_maxiter``
    is deliberately not mapped to TabPFN because it has no iteration semantics in
    the foundation-model estimator.

    Injected ``estimator`` / ``estimators`` objects are already fully constructed.
    Constructor-only Web defaults must not be added in that case: doing so can
    create inconsistent ensemble-size contracts and makes custom estimator reuse
    unexpectedly depend on Web defaults.
    """

    kwargs = dict(model_kwargs)
    normalized_model_type = str(model_type).lower()

    if normalized_model_type == _TABPFN_MODEL:
        _configure_tabpfn_web_environment()
        if kwargs.get("estimator") is not None:
            return kwargs
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
