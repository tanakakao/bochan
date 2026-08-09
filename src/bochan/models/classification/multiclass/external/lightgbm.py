"""LightGBM models for multiclass classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.lightgbm import (
    _LightGBMClassificationEnsembleModel,
    _LightGBMClassificationModel,
)
from bochan.models.external.native_categorical import _NativeCategoricalMixin


class LightGBMMulticlassClassificationModel(_LightGBMClassificationModel):
    """Single LightGBM multiclass classifier."""

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


class LightGBMMulticlassEnsembleModel(_LightGBMClassificationEnsembleModel):
    """Bootstrap LightGBM multiclass ensemble."""

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


class LightGBMMixedMulticlassClassificationModel(
    _NativeCategoricalMixin,
    LightGBMMulticlassClassificationModel,
):
    """Multiclass LightGBM classifier using native categorical mixed inputs."""

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


class LightGBMMixedMulticlassEnsembleModel(
    _NativeCategoricalMixin,
    LightGBMMulticlassEnsembleModel,
):
    """Bootstrap multiclass LightGBM ensemble with native categorical inputs."""

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
    "LightGBMMixedMulticlassClassificationModel",
    "LightGBMMixedMulticlassEnsembleModel",
    "LightGBMMulticlassClassificationModel",
    "LightGBMMulticlassEnsembleModel",
]
