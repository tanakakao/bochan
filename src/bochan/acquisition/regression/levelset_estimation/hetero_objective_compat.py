"""Deprecated compatibility imports for hetero regression level-set objectives.

Heteroscedastic regression LSE inherits the native objective handling from the
standard regression level-set base.  This module remains import-compatible but
performs no class mutation or runtime patching.
"""

from __future__ import annotations

from .single_output import (
    _apply_regression_levelset_objective_to_score as _apply_objective_to_score,
    _is_joint_score,
    _objective_X_for_perturbed_score,
)

__all__ = [
    "_apply_objective_to_score",
    "_is_joint_score",
    "_objective_X_for_perturbed_score",
]
