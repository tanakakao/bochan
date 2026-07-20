"""One-to-many input-shape support for projected model wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from math import prod
from typing import Any

from torch import Tensor


def flatten_projected_one_to_many_point_axes(
    X: Tensor,
    transformed: Tensor,
) -> Tensor:
    """Normalize projected one-to-many inputs to ``batch x (q*n_w) x d``.

    Projected wrappers manually apply a raw-space input transform and then pass
    the result to an inner model without that transform. Depending on the
    BoTorch version or custom transform, a one-to-many transform can retain
    multiple point axes, for example ``batch x q x n_w x d``. The inner model
    must instead receive one point axis.

    Args:
        X: Raw candidate tensor with shape ``batch_shape x q x d``.
        transformed: Input-transformed and projected candidate tensor.

    Returns:
        A tensor with one point axis. Already flattened tensors are returned
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


def _call_projected_transform_in_eval_mode(
    model: Any,
    transform_inputs: Any,
    X: Tensor,
) -> Tensor:
    """Apply a projected wrapper transform using inference-time semantics.

    Some projected classification wrappers leave their cloned input transform
    in training mode after preparing training data. ``InputPerturbation`` with
    ``transform_on_train=False`` then skips one-to-many expansion during
    posterior evaluation, while a regression submodel in the same hybrid model
    expands ``q`` to ``q * n_w``. Temporarily switching the raw-space transform
    to evaluation mode keeps all projected submodels on the same candidate
    axis without permanently changing their training state.
    """

    input_transform = getattr(model, "input_transform", None)
    if input_transform is None:
        return transform_inputs(model, X)

    was_training = getattr(input_transform, "training", None)
    if hasattr(input_transform, "eval"):
        input_transform.eval()
    try:
        return transform_inputs(model, X)
    finally:
        if was_training is True and hasattr(input_transform, "train"):
            input_transform.train()


def configure_projected_transform_inputs(cls: type) -> None:
    """Install one-to-many point-axis normalization on one model class."""

    if getattr(cls, "_bochan_projected_perturbation_patched", False):
        return

    original = cls.transform_inputs

    @wraps(original)
    def supported_transform_inputs(self: Any, X: Tensor) -> Tensor:
        transformed = _call_projected_transform_in_eval_mode(
            self,
            original,
            X,
        )
        return flatten_projected_one_to_many_point_axes(X, transformed)

    cls.transform_inputs = supported_transform_inputs
    cls._bochan_projected_perturbation_patched = True


def configure_projected_model_classes(classes: Iterable[type]) -> None:
    """Patch each projected model class in ``classes``."""

    for cls in classes:
        configure_projected_transform_inputs(cls)


__all__ = [
    "flatten_projected_one_to_many_point_axes",
    "configure_projected_model_classes",
    "configure_projected_transform_inputs",
]
