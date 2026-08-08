"""NGBoost models for multiclass classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.ngboost import (
    _NGBoostClassificationEnsembleModel,
    _NGBoostClassificationModel,
)
from bochan.models.external.common import _MixedCategoricalMixin


class NGBoostMulticlassClassificationModel(_NGBoostClassificationModel):
    """Single NGBoost multiclass classifier."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            binary=False,
            num_classes=num_classes,
            **kwargs,
        )


class NGBoostMulticlassEnsembleModel(_NGBoostClassificationEnsembleModel):
    """Bootstrap NGBoost multiclass ensemble."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            binary=False,
            num_classes=num_classes,
            **kwargs,
        )


class NGBoostMixedMulticlassClassificationModel(
    _MixedCategoricalMixin,
    NGBoostMulticlassClassificationModel,
):
    """Single NGBoost multiclass classifier for mixed inputs."""

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


class NGBoostMixedMulticlassEnsembleModel(
    _MixedCategoricalMixin,
    NGBoostMulticlassEnsembleModel,
):
    """Bootstrap NGBoost multiclass ensemble for mixed inputs."""

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
    "NGBoostMixedMulticlassClassificationModel",
    "NGBoostMixedMulticlassEnsembleModel",
    "NGBoostMulticlassClassificationModel",
    "NGBoostMulticlassEnsembleModel",
]
