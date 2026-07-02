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
    """Apply only transforms that preserve the number of input rows."""

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


def _install_kronecker_input_transform_compatibility() -> None:
    """Backport BoTorch's no-double-transform behavior for Kronecker GPs.

    Affected BoTorch versions transform stored training inputs again inside the
    Kronecker posterior and cached properties. This wrapper leaves already stored
    evaluation inputs untouched, while retaining full transforms for independent
    candidate tensors. During training, one-to-many transforms are excluded so
    the block-design row count remains aligned with ``train_targets``.

    The patch modifies only ``transform_inputs`` and preserves the identity of
    BoTorch's public ``KroneckerMultiTaskGP`` class for API compatibility.
    """

    marker = "_bochan_input_perturbation_compat_installed"
    if bool(getattr(KroneckerMultiTaskGP, marker, False)):
        return

    original_transform_inputs = KroneckerMultiTaskGP.transform_inputs

    def transform_inputs_compat(
        self: KroneckerMultiTaskGP,
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

        # ``Model.eval()`` preprocesses and stores training inputs once. Old
        # Kronecker posterior/cache implementations pass these tensors through
        # ``transform_inputs`` again; return them unchanged to avoid duplicate
        # Normalize and InputPerturbation application.
        if _is_stored_training_input(self, X):
            return X

        return original_transform_inputs(
            self,
            X,
            input_transform=transform,
        ).contiguous()

    setattr(
        KroneckerMultiTaskGP,
        "_bochan_original_transform_inputs",
        original_transform_inputs,
    )
    KroneckerMultiTaskGP.transform_inputs = transform_inputs_compat
    setattr(KroneckerMultiTaskGP, marker, True)


def _install_high_level_objective_defaults() -> None:
    """Install Kronecker-specific ``n_w`` aggregation in the high-level API."""

    from bochan.api.kronecker_input_perturbation_defaults import (
        install_kronecker_input_perturbation_objective_defaults,
    )

    install_kronecker_input_perturbation_objective_defaults()


_install_kronecker_input_transform_compatibility()
_install_high_level_objective_defaults()

# Public compatibility name. This is intentionally an alias, not a subclass, so
# code and tests comparing against BoTorch's class by identity keep working.
PerturbationCompatibleKroneckerMultiTaskGP = KroneckerMultiTaskGP


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
