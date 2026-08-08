"""Mixed-input NGBoost surrogate models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor
from torch.nn import Module

from .common import _MixedCategoricalMixin
from .ngboost import NGBoostEnsembleModel, NGBoostRegressorModel


class NGBoostMixedRegressorModel(_MixedCategoricalMixin, NGBoostRegressorModel):
    """BoTorch-compatible NGBoost regressor for continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimator: Any | None = None,
        min_scale: float = 1e-8,
        categorical_atol: float = 1e-8,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            estimator=estimator,
            min_scale=min_scale,
            **ngboost_kwargs,
        )
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


class NGBoostMixedEnsembleModel(_MixedCategoricalMixin, NGBoostEnsembleModel):
    """Bootstrap NGBoost ensemble for mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        ensemble_size: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: Module | None = None,
        outcome_transform: Module | None = None,
        estimators: Sequence[Any] | None = None,
        estimator_factory: Callable[[], Any] | None = None,
        weights: Tensor | None = None,
        categorical_atol: float = 1e-8,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            ensemble_size=ensemble_size,
            bootstrap=bootstrap,
            random_state=random_state,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            estimators=estimators,
            estimator_factory=estimator_factory,
            weights=weights,
            **ngboost_kwargs,
        )
        self._configure_categorical_encoder(train_X, cat_dims, categorical_atol)


__all__ = [
    "NGBoostMixedEnsembleModel",
    "NGBoostMixedRegressorModel",
]
