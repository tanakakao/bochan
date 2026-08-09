"""External-estimator models for multiclass classification."""

from .lightgbm import (
    LightGBMMixedMulticlassClassificationModel,
    LightGBMMixedMulticlassEnsembleModel,
    LightGBMMulticlassClassificationModel,
    LightGBMMulticlassEnsembleModel,
)
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
    "LightGBMMixedMulticlassClassificationModel",
    "LightGBMMixedMulticlassEnsembleModel",
    "LightGBMMulticlassClassificationModel",
    "LightGBMMulticlassEnsembleModel",
    "NGBoostMixedMulticlassClassificationModel",
    "NGBoostMixedMulticlassEnsembleModel",
    "NGBoostMulticlassClassificationModel",
    "NGBoostMulticlassEnsembleModel",
    "RandomForestMixedMulticlassClassificationModel",
    "RandomForestMulticlassClassificationModel",
]
