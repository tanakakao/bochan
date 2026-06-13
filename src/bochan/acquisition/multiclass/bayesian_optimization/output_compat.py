from __future__ import annotations

import torch
from torch import Tensor

from . import multi_output as _multi_output


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _shape_endswith(shape: torch.Size | tuple[int, ...], suffix: tuple[int, ...]) -> bool:
    if len(suffix) == 0:
        return True
    if len(shape) < len(suffix):
        return False
    return tuple(shape[-len(suffix) :]) == tuple(suffix)


def _finalize_acq_output(value: Tensor, X: Tensor) -> Tensor:
    """Align acquisition output to optimize_acqf's expected t-batch shape.

    DeepGP posteriors can leave extra leading sample / latent dimensions in qEHVI,
    for example ``value.shape == [S, raw_samples]`` while BoTorch expects
    ``[raw_samples]``. The original helper reduced extra dimensions from the
    right, which turned ``[10, 128]`` into ``[10]``. This helper preserves the
    target t-batch suffix and averages only leading sample-like dimensions.
    """
    target_shape = tuple(X.shape[:-2])
    if value.shape == target_shape:
        return value
    if value.ndim == 0:
        return value.expand(*target_shape) if len(target_shape) > 0 else value

    # Remove trailing singleton q/output leftovers first.
    while value.ndim > len(target_shape) and value.shape[-1] == 1:
        value = value.squeeze(-1)
        if value.shape == target_shape:
            return value

    # Key DeepGP case: value=(S, *target_shape). Average leading sample axes.
    if len(target_shape) > 0 and _shape_endswith(value.shape, target_shape):
        leading_ndim = value.ndim - len(target_shape)
        if leading_ndim > 0:
            value = value.mean(dim=tuple(range(leading_ndim)))
        if value.shape == target_shape:
            return value

    # If target shape is scalar, reduce everything to scalar.
    if len(target_shape) == 0:
        return value.mean()

    # If target shape appears contiguously inside the tensor, move to it by
    # averaging all dimensions outside that block. This covers e.g. [S, B, Q]
    # when target is [B] only if there is no exact suffix match.
    shape = tuple(value.shape)
    for start in range(0, len(shape) - len(target_shape) + 1):
        if shape[start : start + len(target_shape)] == target_shape:
            reduce_dims = tuple(i for i in range(value.ndim) if i < start or i >= start + len(target_shape))
            if len(reduce_dims) > 0:
                value = value.mean(dim=reduce_dims)
            if value.shape == target_shape:
                return value

    # Legacy fallback, but avoid reducing the preserved target suffix when present.
    while value.ndim > len(target_shape):
        value = value.mean(dim=0)
        if value.shape == target_shape:
            return value

    if value.numel() == _prod(target_shape):
        return value.reshape(target_shape)
    if value.numel() == 1:
        return value.reshape(()).expand(*target_shape)
    return value


def apply_bayesian_optimization_output_compat() -> None:
    """Patch multiclass BO output finalization helper in-place."""
    _multi_output._finalize_acq_output = _finalize_acq_output


apply_bayesian_optimization_output_compat()


__all__ = ["apply_bayesian_optimization_output_compat"]
