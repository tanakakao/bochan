"""Shared helpers for optional LightGBM-backed estimators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


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


__all__ = ["_resolve_lightgbm_callbacks"]
