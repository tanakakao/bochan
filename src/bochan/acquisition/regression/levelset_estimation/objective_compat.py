"""Deprecated compatibility imports for regression level-set objectives.

The InputPerturbation / joint-score objective handling now lives directly in
``regression.levelset_estimation.single_output``.  This module is kept only for
private import compatibility and intentionally performs no runtime mutation.
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
