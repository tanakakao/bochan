"""Compatibility exports for tabular outcome-constraint support.

Outcome constraints are now composed by :mod:`bochan.api.factory` through the
normal acquisition build path.  This module remains only so older imports of
``apply_tabular_outcome_constraints`` keep working without mutating runtime
functions.
"""

from __future__ import annotations

from bochan.api.feasibility_defaults import (
    _constraint_specs,
    _explicitly_accepts_keyword,
    apply_feasibility_build_plan,
    prepare_feasibility_build,
    resolve_outcome_constraint_config,
)


def apply_tabular_outcome_constraints() -> None:
    """Compatibility no-op; feasibility is integrated into the core API."""

    return None


__all__ = [
    "_constraint_specs",
    "_explicitly_accepts_keyword",
    "apply_feasibility_build_plan",
    "apply_tabular_outcome_constraints",
    "prepare_feasibility_build",
    "resolve_outcome_constraint_config",
]
