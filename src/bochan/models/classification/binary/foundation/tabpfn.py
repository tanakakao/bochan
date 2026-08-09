"""TabPFN foundation models for binary classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.tabpfn import _TabPFNClassificationModel
from bochan.models.external.native_categorical import _NativeCategoricalMixin


class TabPFNBinaryClassificationModel(_TabPFNClassificationModel):
    """Binary TabPFN classifier exposing the final predictive probability."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            binary=True,
            num_classes=2,
            **kwargs,
        )


class TabPFNMixedBinaryClassificationModel(
    _NativeCategoricalMixin,
    TabPFNBinaryClassificationModel,
):
    """Binary TabPFN classifier with estimator-native categorical features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            _categorical_features_indices=cat_dims,
            **kwargs,
        )
        self._configure_native_categorical_encoder(
            train_X,
            cat_dims,
            categorical_atol,
        )


__all__ = [
    "TabPFNBinaryClassificationModel",
    "TabPFNMixedBinaryClassificationModel",
]
