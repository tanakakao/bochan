from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Optional

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, ScaleKernel
from torch import Tensor


def normalize_mixed_dims(cat_dims: Sequence[int], d: int) -> list[int]:
    """Normalize categorical feature indices.

    Args:
        cat_dims: Categorical feature indices. Negative indices are accepted.
        d: Total input dimension.

    Returns:
        Sorted unique non-negative categorical indices.
    """
    normalized: list[int] = []
    for raw_index in cat_dims:
        index = int(raw_index)
        if index < 0:
            index += int(d)
        if index < 0 or index >= int(d):
            raise ValueError(f"Invalid categorical dim {raw_index} for input dim {d}.")
        normalized.append(index)
    normalized = sorted(set(normalized))
    if len(normalized) == 0:
        raise ValueError("cat_dims must contain at least one categorical feature index.")
    return normalized


def get_continuous_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
    """Return feature indices not included in ``cat_dims``."""
    categorical = set(normalize_mixed_dims(cat_dims, d))
    return [index for index in range(int(d)) if index not in categorical]


def _make_continuous_kernel(
    *,
    cont_dims: Sequence[int],
    batch_shape: torch.Size,
) -> Optional[Kernel]:
    if len(cont_dims) == 0:
        return None
    return MaternKernel(
        nu=2.5,
        ard_num_dims=len(cont_dims),
        active_dims=tuple(int(index) for index in cont_dims),
        batch_shape=batch_shape,
    )


def _make_categorical_kernel(
    *,
    cat_dims: Sequence[int],
    batch_shape: torch.Size,
) -> Optional[Kernel]:
    if len(cat_dims) == 0:
        return None
    return CategoricalKernel(
        ard_num_dims=len(cat_dims),
        active_dims=tuple(int(index) for index in cat_dims),
        batch_shape=batch_shape,
        lengthscale_constraint=GreaterThan(1e-6),
    )


def build_mixed_kronecker_kernel(
    *,
    d: int,
    cat_dims: Sequence[int],
    batch_shape: torch.Size = torch.Size(),
) -> Kernel:
    r"""Build a continuous/categorical kernel for Kronecker multi-task models.

    The default structure follows the mixed-input models used elsewhere in
    bochan:

    .. math::

        K_X = s\left(K_{c,1} + K_{g,1} + K_{c,2}K_{g,2}\right),

    where ``c`` denotes continuous features and ``g`` categorical features.
    For all-categorical input only the categorical term is used. The outer
    :class:`ScaleKernel` supplies one shared output scale per ``batch_shape``.

    Args:
        d: Total input dimension.
        cat_dims: Categorical feature indices.
        batch_shape: Kernel parameter batch shape. Use ``[]`` for binary,
            ordinal, and Gaussian Kronecker models, and ``[C, 1]`` for the
            class-batched multiclass Kronecker model.

    Returns:
        Mixed-input covariance module.
    """
    cat_dims = normalize_mixed_dims(cat_dims, d)
    cont_dims = [index for index in range(int(d)) if index not in set(cat_dims)]
    batch_shape = torch.Size(batch_shape)

    cont_1 = _make_continuous_kernel(cont_dims=cont_dims, batch_shape=batch_shape)
    cat_1 = _make_categorical_kernel(cat_dims=cat_dims, batch_shape=batch_shape)

    if cont_1 is None:
        if cat_1 is None:
            raise RuntimeError("Failed to construct a mixed-input kernel.")
        base_kernel = cat_1
    elif cat_1 is None:
        base_kernel = cont_1
    else:
        cont_2 = _make_continuous_kernel(cont_dims=cont_dims, batch_shape=batch_shape)
        cat_2 = _make_categorical_kernel(cat_dims=cat_dims, batch_shape=batch_shape)
        if cont_2 is None or cat_2 is None:
            raise RuntimeError("Failed to construct mixed interaction kernels.")
        base_kernel = cont_1 + cat_1 + ProductKernel(cont_2, cat_2)

    return ScaleKernel(base_kernel, batch_shape=batch_shape)


def _expand_raw_to_transformed_shape(X: Tensor, X_tf: Tensor) -> Tensor:
    """Expand raw candidates when an input transform expands the q dimension."""
    if X.shape == X_tf.shape:
        return X
    if X.ndim < 2 or X_tf.ndim < 2 or X.shape[-1] != X_tf.shape[-1]:
        return X
    if X.shape[:-2] == X_tf.shape[:-2]:
        q = int(X.shape[-2])
        q_like = int(X_tf.shape[-2])
        if q > 0 and q_like % q == 0:
            return X.repeat_interleave(q_like // q, dim=-2)
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)
    return X


def check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    *,
    cat_dims: Sequence[int],
) -> None:
    """Validate that an input transform leaves categorical columns unchanged."""
    cat_dims = normalize_mixed_dims(cat_dims, X.shape[-1])
    X_cmp = _expand_raw_to_transformed_shape(X, X_tf)
    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw and transformed mixed inputs for categorical "
            "column validation: "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}."
        )
    if not torch.allclose(X_cmp[..., cat_dims], X_tf[..., cat_dims]):
        raise ValueError(
            "input_transform must not modify categorical columns. "
            "Normalize or perturb continuous columns only."
        )


def validate_mixed_input_transform_for_training(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Sequence[int],
) -> None:
    """Validate a copy of a transform without changing the caller's module state."""
    if input_transform is None:
        return
    transform = copy.deepcopy(input_transform).to(device=X.device, dtype=X.dtype)
    transform.train()
    with torch.no_grad():
        X_tf = transform(X).contiguous()
    check_categorical_columns_unchanged(X, X_tf, cat_dims=cat_dims)


def transform_mixed_inputs(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Sequence[int],
) -> Tensor:
    """Apply an evaluation transform and validate categorical columns."""
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    if isinstance(X_tf, tuple):
        X_tf = X_tf[0]
    X_tf = X_tf.contiguous()
    check_categorical_columns_unchanged(X, X_tf, cat_dims=cat_dims)
    return X_tf


__all__ = [
    "build_mixed_kronecker_kernel",
    "check_categorical_columns_unchanged",
    "get_continuous_dims",
    "normalize_mixed_dims",
    "transform_mixed_inputs",
    "validate_mixed_input_transform_for_training",
]
