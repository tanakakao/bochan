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

    ``KroneckerMultiTaskGP.posterior`` transforms both test inputs and the stored
    training inputs while the model is in evaluation mode. A one-to-many
    transform such as ``InputPerturbation`` must be applied to the test inputs,
    but applying it to the stored training inputs changes ``n`` and breaks the
    Kronecker training caches.
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
    """Return whether ``X`` is one of the exact tensors stored by ExactGP."""

    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple):
        return any(X is train_input for train_input in train_inputs)
    return X is train_inputs


def _must_preserve_kronecker_training_shape(model: Any, X: Tensor) -> bool:
    """Return whether one-to-many input transforms must be skipped."""

    # During construction ``training`` may not yet be initialized. Treat that
    # state as training so constructor validation sees the original number of
    # rows. During posterior evaluation, the stored training tensor is detected
    # by identity and kept pointwise, while independent candidate tensors are
    # still expanded by InputPerturbation.
    return bool(getattr(model, "training", True)) or _is_stored_training_input(
        model,
        X,
    )


class PerturbationCompatibleKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    """Kronecker GP that applies one-to-many transforms only to test inputs.

    BoTorch's Kronecker posterior calls ``transform_inputs`` for both ``X`` and
    ``self.train_inputs[0]`` in evaluation mode. With ``InputPerturbation`` this
    expands the stored training design from ``n`` to ``n * n_w`` while
    ``train_targets`` remains ``n x m``. This subclass preserves pointwise
    transforms such as ``Normalize`` for training inputs and limits the
    one-to-many perturbation expansion to candidate inputs.
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
        if _must_preserve_kronecker_training_shape(self, X):
            return _transform_without_one_to_many(X, transform).contiguous()
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

        if _must_preserve_kronecker_training_shape(self, X):
            X_tf = _transform_without_one_to_many(X, transform).contiguous()
            check_categorical_columns_unchanged(
                X,
                X_tf,
                cat_dims=self.cat_dims,
            )
            return X_tf

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
