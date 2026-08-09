"""Target settings with request-local missing-value handling.

The stable target semantics live in :mod:`target_settings_core`. Missing-value
behavior is selected through ordinary function imports from
:mod:`target_missing_policy`; no workflow function is replaced at runtime.
"""

from __future__ import annotations

from typing import Any

from bochan.api.nan_multiobjective import make_nan_safe_default_ref_point

from .model_runtime import apply_web_model_runtime_defaults
from .target_missing_policy import (
    clean_rows as _clean_rows,
)
from .target_missing_policy import (
    encode_targets as _encode_targets,
)
from .target_missing_policy import (
    resolve_target_settings as _resolve_target_settings_with_missing_policy,
)
from .target_settings_core import (
    _as_2d,
    _build_outcome_constraint_config,
    _build_target_constraint_config,
    _model_kwargs,
    _objective_values_direct,
    _output_spec_kwargs,
    _resolve_targets,
    _validate_columns,
)


def _resolve_target_settings(
    request: Any,
    *,
    target_columns: list[str],
    directions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve target settings and apply request-local Web model defaults."""

    settings, model_kwargs = _resolve_target_settings_with_missing_policy(
        request,
        target_columns=target_columns,
        directions=directions,
    )
    runtime_kwargs = apply_web_model_runtime_defaults(
        model_kwargs,
        model_type=str(getattr(request, "model_type", "base")),
        fit_maxiter=int(getattr(request, "fit_maxiter", 128)),
    )
    return settings, runtime_kwargs


def _reference_point(values):
    """Build a dominated reference point from each objective's finite values."""

    return make_nan_safe_default_ref_point(values)


__all__ = [
    "_as_2d",
    "_build_outcome_constraint_config",
    "_build_target_constraint_config",
    "_clean_rows",
    "_encode_targets",
    "_model_kwargs",
    "_objective_values_direct",
    "_output_spec_kwargs",
    "_reference_point",
    "_resolve_target_settings",
    "_resolve_targets",
    "_validate_columns",
]
