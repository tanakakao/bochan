"""External-estimator models for multiclass classification."""

from .ngboost import (
    NGBoostMixedMulticlassClassificationModel,
    NGBoostMixedMulticlassEnsembleModel,
    NGBoostMulticlassClassificationModel,
    NGBoostMulticlassEnsembleModel,
)
from .random_forest import (
    RandomForestMixedMulticlassClassificationModel,
    RandomForestMulticlassClassificationModel,
)

__all__ = [
    "NGBoostMixedMulticlassClassificationModel",
    "NGBoostMixedMulticlassEnsembleModel",
    "NGBoostMulticlassClassificationModel",
    "NGBoostMulticlassEnsembleModel",
    "RandomForestMixedMulticlassClassificationModel",
    "RandomForestMulticlassClassificationModel",
]
