"""Random Forest models for multiclass classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.random_forest import _RandomForestClassificationModel
from bochan.models.external.common import _MixedCategoricalMixin


class RandomForestMulticlassClassificationModel(_RandomForestClassificationModel):
    """Multiclass Random Forest classifier with tree-level probability samples."""

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


class RandomForestMixedMulticlassClassificationModel(
    _MixedCategoricalMixin,
    RandomForestMulticlassClassificationModel,
):
    """Multiclass Random Forest classifier for mixed inputs."""

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
    "RandomForestMixedMulticlassClassificationModel",
    "RandomForestMulticlassClassificationModel",
]
