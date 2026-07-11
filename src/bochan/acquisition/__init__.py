"""Unified acquisition package.

Objective helpers live in :mod:`acquisition.objective`.
"""

from .classification_constraints import (
    apply_classification_constraints,
)
from .multiclass_constraint_install import apply_multiclass_constraint_support
from .nsgaii_constraint_install import install_nsgaii_constraints
from .ordinal_constraint_install import apply_ordinal_constraint_support
from .ordinal_multitask import apply_ordinal_multitask

apply_ordinal_multitask()
apply_classification_constraints()
apply_multiclass_constraint_support()
apply_ordinal_constraint_support()
install_nsgaii_constraints()


__all__ = [
    "apply_classification_constraints",
    "apply_multiclass_constraint_support",
    "apply_ordinal_constraint_support",
    "apply_ordinal_multitask",
    "install_nsgaii_constraints",
]
