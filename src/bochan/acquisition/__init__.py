"""Unified acquisition package.

Objective helpers live in :mod:`acquisition.objective`.
"""

from .classification_constraint_compat import (
    apply_classification_constraint_compat,
)
from .multiclass_constraint_install import apply_multiclass_constraint_compat
from .nsgaii_constraint_install import install_nsgaii_constraint_compat
from .ordinal_constraint_install import apply_ordinal_constraint_compat
from .ordinal_multitask_compat import apply_ordinal_multitask_compat

apply_ordinal_multitask_compat()
apply_classification_constraint_compat()
apply_multiclass_constraint_compat()
apply_ordinal_constraint_compat()
install_nsgaii_constraint_compat()


__all__ = [
    "apply_classification_constraint_compat",
    "apply_multiclass_constraint_compat",
    "apply_ordinal_constraint_compat",
    "apply_ordinal_multitask_compat",
    "install_nsgaii_constraint_compat",
]
