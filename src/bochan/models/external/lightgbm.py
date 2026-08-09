"""Shared helpers for optional LightGBM-backed estimators."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Any


_LIGHTGBM_MIN_DATA_ALIASES = (
    "min_child_samples",
    "min_data_in_leaf",
    "min_data_per_leaf",
    "min_data",
    "min_samples_leaf",
)


def _default_lightgbm_min_child_samples(n_samples: int) -> int:
    """Return a small-data-friendly LightGBM leaf-size default.

    LightGBM's sklearn API defaults ``min_child_samples`` to 20. That setting
    prevents any split when fewer than 40 observations are available, which is
    common in Bayesian optimization and active-learning studies. Use roughly
    ten percent of the available observations per leaf for small datasets, and
    retain LightGBM's standard value once the dataset is large enough.
    """
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive.")
    return min(20, max(1, ceil(int(n_samples) * 0.1)))


def _resolve_lightgbm_estimator_kwargs(
    kwargs: Mapping[str, Any],
    *,
    n_samples: int,
) -> dict[str, Any]:
    """Apply bochan's small-data defaults without overriding explicit values."""
    resolved = dict(kwargs)
    if not any(alias in resolved for alias in _LIGHTGBM_MIN_DATA_ALIASES):
        resolved["min_child_samples"] = _default_lightgbm_min_child_samples(n_samples)
    return resolved


def _resolve_lightgbm_callbacks(
    callbacks: Sequence[Callable[..., Any]] | None,
    *,
    early_stopping_rounds: int | None,
    has_validation: bool,
) -> list[Callable[..., Any]] | None:
    """Build LightGBM callbacks without importing LightGBM unless needed."""
    result = list(callbacks or [])
    if early_stopping_rounds is not None:
        if int(early_stopping_rounds) <= 0:
            raise ValueError("early_stopping_rounds must be positive.")
        if not has_validation:
            raise ValueError("early_stopping_rounds requires validation data.")
        try:
            from lightgbm import early_stopping
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LightGBM early stopping requires the optional `lightgbm` dependency."
            ) from exc
        result.append(early_stopping(int(early_stopping_rounds), verbose=False))
    return result or None


__all__ = [
    "_default_lightgbm_min_child_samples",
    "_resolve_lightgbm_callbacks",
    "_resolve_lightgbm_estimator_kwargs",
]
