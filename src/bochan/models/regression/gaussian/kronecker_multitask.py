from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.input import ChainedInputTransform, InputTransform
from gpytorch.kernels import Kernel
from torch import Tensor

from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    check_categorical_columns_unchanged,
    get_continuous_dims,
    normalize_mixed_dims,
    transform_mixed_inputs,
    validate_mixed_input_transform_for_training,
)


def _transform_without_one_to_many(
    X: Tensor,
    transform: Optional[InputTransform],
) -> Tensor:
    """Apply only transforms that preserve the number of input rows.

    During training, pointwise transforms such as ``Normalize`` must remain
    active, while one-to-many transforms such as ``InputPerturbation`` must not
    expand the block-design training data.
    """

    if transform is None:
        return X
    if isinstance(transform, ChainedInputTransform):
        transformed = X
        for subtransform in transform.values():
            transformed = _transform_without_one_to_many(
                transformed,
                subtransform,
            )
        return transformed
    if bool(getattr(transform, "is_one_to_many", False)):
        return X

    transformed = transform(X)
    if isinstance(transformed, tuple):
        transformed = transformed[0]
    return transformed


def _is_stored_training_input(model: Any, X: Tensor) -> bool:
    """Return whether ``X`` is one of the tensors stored by ``ExactGP``."""

    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple):
        return any(X is train_input for train_input in train_inputs)
    return X is train_inputs


class PerturbationCompatibleKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    """Kronecker GP compatible with one-to-many input perturbations.

    In affected BoTorch versions, ``KroneckerMultiTaskGP.posterior`` and its
    cached properties transform the already-preprocessed ``train_inputs`` a
    second time. With ``InputPerturbation`` this can repeatedly multiply the
    training row count while ``train_targets`` remains unchanged.

    This subclass keeps three paths separate:

    - training forward: apply only pointwise transforms;
    - stored evaluation training inputs: return them unchanged;
    - independent evaluation candidates: apply the full transform chain.
    """

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
        if transform is None:
            return X

        if bool(getattr(self, "training", True)):
            return _transform_without_one_to_many(X, transform).contiguous()

        # ``Model.eval()`` preprocesses and stores the training inputs once.
        # Old Kronecker posterior/cache implementations call transform_inputs on
        # these stored tensors again; returning them directly mirrors BoTorch's
        # upstream fix for the duplicate transformation.
        if _is_stored_training_input(self, X):
            return X

        return super().transform_inputs(
            X,
            input_transform=transform,
        ).contiguous()


class MixedKroneckerMultiTaskGP(PerturbationCompatibleKroneckerMultiTaskGP):
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
        if transform is None:
            return X

        if bool(getattr(self, "training", True)):
            X_tf = _transform_without_one_to_many(X, transform).contiguous()
            check_categorical_columns_unchanged(
                X,
                X_tf,
                cat_dims=self.cat_dims,
            )
            return X_tf

        if _is_stored_training_input(self, X):
            return X

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
    "PerturbationCompatibleKroneckerMultiTaskGP",
]
