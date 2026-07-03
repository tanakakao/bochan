"""One-to-many input-shape compatibility for projected model wrappers."""

from __future__ import annotations

from functools import wraps
from math import prod
from typing import Any, Iterable

from torch import Tensor


def flatten_projected_one_to_many_point_axes(
    X: Tensor,
    transformed: Tensor,
) -> Tensor:
    """Normalize projected one-to-many inputs to ``batch x (q*n_w) x d``.

    Projected wrappers manually apply a raw-space input transform and then pass
    the result to an inner model without that transform.  Depending on the
    BoTorch version or custom transform, a one-to-many transform can retain
    multiple point axes, for example ``batch x q x n_w x d``.  The inner model
    must instead receive one point axis.

    Args:
        X: Raw candidate tensor with shape ``batch_shape x q x d``.
        transformed: Input-transformed and projected candidate tensor.

    Returns:
        A tensor with one point axis.  Already flattened tensors are returned
        unchanged.
    """

    if isinstance(X, tuple):
        X = X[0]
    if transformed.ndim <= X.ndim or X.ndim < 2:
        return transformed

    batch_shape = tuple(X.shape[:-2])
    q = int(X.shape[-2])
    if q <= 0:
        return transformed

    batch_ndim = len(batch_shape)
    if tuple(transformed.shape[:batch_ndim]) != batch_shape:
        return transformed

    point_shape = tuple(int(size) for size in transformed.shape[batch_ndim:-1])
    if len(point_shape) < 2:
        return transformed

    q_like = prod(point_shape)
    if q_like % q != 0:
        return transformed

    return transformed.reshape(*batch_shape, q_like, transformed.shape[-1])


def patch_projected_transform_inputs(cls: type) -> None:
    """Install one-to-many point-axis normalization on one model class."""

    if getattr(cls, "_bochan_projected_perturbation_patched", False):
        return

    original = cls.transform_inputs

    @wraps(original)
    def compatible_transform_inputs(self: Any, X: Tensor) -> Tensor:
        transformed = original(self, X)
        return flatten_projected_one_to_many_point_axes(X, transformed)

    cls.transform_inputs = compatible_transform_inputs
    cls._bochan_projected_perturbation_patched = True


def patch_projected_model_classes(classes: Iterable[type]) -> None:
    """Patch each projected model class in ``classes``."""

    for cls in classes:
        patch_projected_transform_inputs(cls)


__all__ = [
    "flatten_projected_one_to_many_point_axes",
    "patch_projected_model_classes",
    "patch_projected_transform_inputs",
]
