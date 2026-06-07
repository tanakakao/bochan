"""Canonical multiclass base model module.

This module mirrors ``bochan.models.classification.binary.base.models`` and
re-exports the existing multiclass implementation.  The legacy
``bochan.models.classification.multiclass.multiclass`` module is kept for
backward compatibility.
"""

from __future__ import annotations

from bochan.models.classification.multiclass.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
    build_mixed_multiclass_kernel,
)

__all__ = [
    "MulticlassClassificationGPModel",
    "MulticlassClassificationMixedGPModel",
    "build_mixed_multiclass_kernel",
]
