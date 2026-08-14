"""Target settings with request-local missing-value handling.

The stable target semantics live in :mod:`bochan.serving.webapp.targets.settings`. Missing-value
behavior is selected through ordinary function imports from
:mod:`bochan.serving.webapp.targets.missing`; no workflow function is replaced at runtime.
"""

from __future__ import annotations

from bochan.api.acquisition.defaults.nan import make_nan_safe_default_ref_point

from ..targets.missing import (
    clean_rows as _clean_rows,
)
from ..targets.missing import (
    encode_targets as _encode_targets,
)
from ..targets.missing import (
    resolve_target_settings as _resolve_target_settings,
)
from ..targets.settings import (
    _as_2d,
    _build_outcome_constraint_config,
    _build_target_constraint_config,
    _model_kwargs,
    _objective_values_direct,
    _output_spec_kwargs,
    _resolve_targets,
    _validate_columns,
)


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
