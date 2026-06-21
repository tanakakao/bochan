from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from torch import Tensor

from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    get_continuous_dims,
    normalize_mixed_dims,
    transform_mixed_inputs,
    validate_mixed_input_transform_for_training,
)


class MixedKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    r"""Exact Gaussian Kronecker multi-task GP for mixed inputs.

    The model retains BoTorch's block-design Gaussian multi-task likelihood and
    replaces the data covariance with an additive-plus-interaction kernel over
    continuous and categorical features.

    Args:
        train_X: Shared mixed input locations with shape ``[n, d]``.
        train_Y: Continuous task observations with shape ``[n, m]``.
        cat_dims: Categorical feature indices.
        data_covar_module: Optional custom mixed-input data kernel. When omitted,
            ``continuous + categorical + continuous * categorical`` is used.
        input_transform: Optional transform that must leave categorical columns
            unchanged.
        **kwargs: Additional arguments accepted by
            :class:`botorch.models.multitask.KroneckerMultiTaskGP`.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        data_covar_module: Optional[Kernel] = None,
        input_transform: Optional[InputTransform] = None,
        **kwargs: Any,
    ) -> None:
        raw_train_X = torch.as_tensor(train_X).contiguous()
        normalized_cat_dims = normalize_mixed_dims(
            cat_dims,
            raw_train_X.shape[-1],
        )
        validate_mixed_input_transform_for_training(
            raw_train_X,
            input_transform,
            cat_dims=normalized_cat_dims,
        )
        if data_covar_module is None:
            data_covar_module = build_mixed_kronecker_kernel(
                d=raw_train_X.shape[-1],
                cat_dims=normalized_cat_dims,
            )

        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )
        super().__init__(
            train_X=raw_train_X,
            train_Y=train_Y,
            data_covar_module=data_covar_module,
            input_transform=input_transform,
            **kwargs,
        )
        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Optional[InputTransform] = None,
    ) -> Tensor:
        transform = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        return transform_mixed_inputs(
            X,
            transform,
            cat_dims=self.cat_dims,
        )


# Backward-compatible alternative naming order.
KroneckerMultiTaskMixedGP = MixedKroneckerMultiTaskGP


__all__ = [
    "KroneckerMultiTaskMixedGP",
    "MixedKroneckerMultiTaskGP",
]
