"""InputPerturbation shape compatibility for projected binary classifiers.

Projected PCA / REMBO wrappers apply the raw-space input transform before the
projection and then pass the projected tensor to an inner GP that no longer owns
that transform.  Some BoTorch versions represent a one-to-many transform as
``batch_shape x q x n_w x d`` instead of flattening the two point axes.  The
inner GP interprets the additional axis as a model batch axis, which corrupts
the tensor layout expected by multi-output EHVI / NEHVI.
"""

from __future__ import annotations

from functools import wraps
from math import prod
from typing import Any

from torch import Tensor


_PATCHED = False


def _flatten_one_to_many_point_axes(X: Tensor, transformed: Tensor) -> Tensor:
    """Normalize projected one-to-many inputs to ``batch x (q*n_w) x d``.

    Args:
        X: Raw candidate tensor with shape ``batch_shape x q x d``.
        transformed: Projected candidate tensor.  A one-to-many transform may
            return either ``batch_shape x (q*n_w) x d_projected`` or a tensor
            with multiple point axes such as
            ``batch_shape x q x n_w x d_projected``.

    Returns:
        Tensor with a single point axis.  Tensors that already follow the
        standard ``batch_shape x q_like x d`` convention are returned unchanged.
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


def _patch_transform_inputs(cls: type) -> None:
    """Install one-to-many point-axis normalization on a projected model class."""

    if getattr(cls, "_bochan_projected_perturbation_patched", False):
        return

    original = cls.transform_inputs

    @wraps(original)
    def compatible_transform_inputs(self: Any, X: Tensor) -> Tensor:
        transformed = original(self, X)
        return _flatten_one_to_many_point_axes(X, transformed)

    cls.transform_inputs = compatible_transform_inputs
    cls._bochan_projected_perturbation_patched = True


def apply_projected_binary_perturbation_compat() -> None:
    """Patch PCA and REMBO binary classifiers once."""

    global _PATCHED
    if _PATCHED:
        return

    from .decomposition import (
        PCABinaryClassificationGPModel,
        REMBOBinaryClassificationGPModel,
    )

    _patch_transform_inputs(PCABinaryClassificationGPModel)
    _patch_transform_inputs(REMBOBinaryClassificationGPModel)
    _PATCHED = True


__all__ = [
    "apply_projected_binary_perturbation_compat",
]
