"""TabPFN foundation models for multiclass classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.tabpfn import _TabPFNClassificationModel
from bochan.models.external.native_categorical import _NativeCategoricalMixin


class TabPFNMulticlassClassificationModel(_TabPFNClassificationModel):
    """Multiclass TabPFN classifier exposing final class probabilities."""

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


class TabPFNMixedMulticlassClassificationModel(
    _NativeCategoricalMixin,
    TabPFNMulticlassClassificationModel,
):
    """Multiclass TabPFN classifier with estimator-native categorical features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        num_classes: int | None = None,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            _categorical_features_indices=cat_dims,
            **kwargs,
        )
        self._configure_native_categorical_encoder(
            train_X,
            cat_dims,
            categorical_atol,
        )


__all__ = [
    "TabPFNMixedMulticlassClassificationModel",
    "TabPFNMulticlassClassificationModel",
]
