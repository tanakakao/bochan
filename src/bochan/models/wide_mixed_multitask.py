"""Wide partial-observation multi-task GP for mixed continuous/categorical inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    normalize_mixed_dims,
    validate_mixed_input_transform_for_training,
)

from .wide_multitask_variants import WideMultiTaskGP


class WideMixedMultiTaskGP(WideMultiTaskGP):
    """ICM multi-task GP with a mixed kernel over the non-task features.

    Targets use the same wide ``[n, m]`` representation as
    :class:`WideMultiTaskGP`, including NaN cells for unobserved task values. The
    public categorical columns are modeled with a categorical kernel instead of
    being treated as continuous coordinates. The appended task id remains the
    dedicated ICM task feature managed by BoTorch ``MultiTaskGP``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        **kwargs: Any,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        categorical_dims = normalize_mixed_dims(cat_dims, int(train_X.shape[-1]))

        input_transform = kwargs.get("input_transform")
        validate_mixed_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=categorical_dims,
        )

        prepared = dict(kwargs)
        if prepared.get("covar_module") is None:
            prepared["covar_module"] = build_mixed_kronecker_kernel(
                d=int(train_X.shape[-1]),
                cat_dims=categorical_dims,
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            **prepared,
        )
        self.cat_dims = list(categorical_dims)


__all__ = ["WideMixedMultiTaskGP"]
