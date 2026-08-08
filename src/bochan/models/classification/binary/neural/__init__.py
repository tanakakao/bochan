"""Neural models for binary classification."""

from .deep_ensemble import (
    DeepEnsembleBinaryClassificationModel,
    DeepEnsembleMixedBinaryClassificationModel,
)

__all__ = [
    "DeepEnsembleBinaryClassificationModel",
    "DeepEnsembleMixedBinaryClassificationModel",
]
