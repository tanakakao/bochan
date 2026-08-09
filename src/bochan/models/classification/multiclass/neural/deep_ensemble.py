"""Deep Ensemble models for multiclass classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from bochan.models.classification.common.deep_ensemble import _DeepEnsembleClassificationModel
from bochan.models.regression.neural.deep_ensemble import _MixedCategoricalEncoder


class DeepEnsembleMulticlassClassificationModel(_DeepEnsembleClassificationModel):
    """Multiclass Deep Ensemble classifier with simplex-valued member samples."""

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


class DeepEnsembleMixedMulticlassClassificationModel(
    DeepEnsembleMulticlassClassificationModel
):
    """Multiclass Deep Ensemble classifier for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        encoder = _MixedCategoricalEncoder(
            train_X=train_X,
            cat_dims=cat_dims,
            atol=categorical_atol,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            feature_encoder=encoder,
            encoded_input_dim=encoder.encoded_dim,
            **kwargs,
        )
        self.cat_dims = list(encoder.cat_dims)

    @property
    def categorical_values(self) -> dict[int, tuple[float, ...]]:
        encoder = self.feature_encoder
        if not isinstance(encoder, _MixedCategoricalEncoder):  # pragma: no cover
            raise RuntimeError("Mixed Deep Ensemble categorical encoder is unavailable.")
        return encoder.categorical_values


__all__ = [
    "DeepEnsembleMixedMulticlassClassificationModel",
    "DeepEnsembleMulticlassClassificationModel",
]
