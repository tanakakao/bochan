"""NGBoost models for binary classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.ngboost import (
    _NGBoostClassificationEnsembleModel,
    _NGBoostClassificationModel,
)
from bochan.models.external.common import _MixedCategoricalMixin


class NGBoostBinaryClassificationModel(_NGBoostClassificationModel):
    """Single probabilistic NGBoost binary classifier."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, binary=True, num_classes=2, **kwargs)


class NGBoostBinaryEnsembleModel(_NGBoostClassificationEnsembleModel):
    """Bootstrap NGBoost binary ensemble with epistemic probability samples."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, binary=True, num_classes=2, **kwargs)


class NGBoostMixedBinaryClassificationModel(
    _MixedCategoricalMixin,
    NGBoostBinaryClassificationModel,
):
    """Single NGBoost binary classifier for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


class NGBoostMixedBinaryEnsembleModel(
    _MixedCategoricalMixin,
    NGBoostBinaryEnsembleModel,
):
    """Bootstrap NGBoost binary ensemble for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


__all__ = [
    "NGBoostBinaryClassificationModel",
    "NGBoostBinaryEnsembleModel",
    "NGBoostMixedBinaryClassificationModel",
    "NGBoostMixedBinaryEnsembleModel",
]
