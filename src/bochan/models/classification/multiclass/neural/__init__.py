"""Neural models for multiclass classification."""

from .deep_ensemble import (
    DeepEnsembleMixedMulticlassClassificationModel,
    DeepEnsembleMulticlassClassificationModel,
)

__all__ = [
    "DeepEnsembleMixedMulticlassClassificationModel",
    "DeepEnsembleMulticlassClassificationModel",
]
