"""LightGBM models for binary classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.lightgbm import (
    _LightGBMClassificationEnsembleModel,
    _LightGBMClassificationModel,
)
from bochan.models.external.native_categorical import _NativeCategoricalMixin


class LightGBMBinaryClassificationModel(_LightGBMClassificationModel):
    """Single LightGBM binary classifier."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, binary=True, num_classes=2, **kwargs)


class LightGBMBinaryEnsembleModel(_LightGBMClassificationEnsembleModel):
    """Bootstrap LightGBM binary ensemble for epistemic probability uncertainty."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, binary=True, num_classes=2, **kwargs)


class LightGBMMixedBinaryClassificationModel(
    _NativeCategoricalMixin,
    LightGBMBinaryClassificationModel,
):
    """Binary LightGBM classifier using native categorical mixed inputs."""

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
        self._configure_native_categorical_encoder(train_X, cat_dims, categorical_atol)


class LightGBMMixedBinaryEnsembleModel(
    _NativeCategoricalMixin,
    LightGBMBinaryEnsembleModel,
):
    """Bootstrap binary LightGBM ensemble with native categorical inputs."""

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
        self._configure_native_categorical_encoder(train_X, cat_dims, categorical_atol)


__all__ = [
    "LightGBMBinaryClassificationModel",
    "LightGBMBinaryEnsembleModel",
    "LightGBMMixedBinaryClassificationModel",
    "LightGBMMixedBinaryEnsembleModel",
]
