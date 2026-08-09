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


def apply_web_model_runtime_defaults(
    model_kwargs: dict[str, Any],
    *,
    model_type: str,
    fit_maxiter: int,
) -> dict[str, Any]:
    """Apply interactive Web defaults while preserving explicit model kwargs.

    NGBoost's estimator count is a constructor argument rather than a generic
    ``FitConfig`` option. Therefore the Web ``fit_maxiter`` control must be
    translated explicitly for NGBoost. The ensemble is also intentionally
    smaller than the core model default because each member is a complete
    NGBoost fit and is evaluated again during acquisition optimization.
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


__all__ = ["apply_web_model_runtime_defaults"]
