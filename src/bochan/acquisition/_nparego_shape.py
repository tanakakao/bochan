"""Shape normalization shared by multi-output NParEGO acquisitions."""

from __future__ import annotations

import torch
from torch import Tensor


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _matches_trailing_shape(value: Tensor, shape: torch.Size) -> bool:
    """Return whether ``value`` ends with ``shape``.

    An empty t-batch shape always matches. This is useful for ``q=1`` because
    reducing a present singleton q-axis and accepting an already squeezed q-axis
    are numerically equivalent when no t-batch dimensions exist.
    """
    if len(shape) == 0:
        return True
    if value.ndim < len(shape):
        return False
    return tuple(value.shape[-len(shape) :]) == tuple(shape)


def reduce_nparego_sample_and_q_to_tbatch(value: Tensor, X: Tensor) -> Tensor:
    """Reduce NParEGO sample and q dimensions while preserving t-batches.

    Expected normal layout is ``sample_shape x batch_shape x q``. Some
    classification and ordinal posterior/objective paths squeeze the q-axis when
    ``q == 1`` and instead return ``sample_shape x batch_shape``. Sequential
    ``optimize_acqf`` uses exactly this case while evaluating many restart/raw
    sample batches, so the missing singleton q-axis must be accepted without
    confusing the optimizer batch dimension for q.
    """
    batch_shape = X.shape[:-2]
    q = int(X.shape[-2])
    batch_prod = _prod(batch_shape)
    batch_ndim = len(batch_shape)

    # Remove an explicit output singleton only for [..., q, 1].
    if (
        value.ndim >= batch_ndim + 2
        and value.shape[-1] == 1
        and value.shape[-2] == q
    ):
        value = value.squeeze(-1)

    # If q=1 was squeezed by the posterior/objective, the remaining trailing
    # dimensions are the t-batch dimensions. Do not reduce the last batch axis.
    q_axis_is_squeezed = q == 1 and _matches_trailing_shape(value, batch_shape)

    if q_axis_is_squeezed:
        pass
    elif value.ndim >= 1 and value.shape[-1] == q:
        value = value.max(dim=-1).values
    else:
        raise RuntimeError(
            "Expected scalarized NParEGO values to end in q or, for q=1, "
            "to end in the t-batch shape. "
            f"value.shape={tuple(value.shape)}, q={q}, "
            f"batch_shape={tuple(batch_shape)}, X.shape={tuple(X.shape)}."
        )

    # Average all MC / model dimensions preceding the preserved t-batch axes.
    while value.ndim > batch_ndim:
        value = value.mean(dim=0)

    if value.shape == batch_shape:
        return value
    if value.numel() == batch_prod:
        return value.reshape(batch_shape)
    if len(batch_shape) == 0 and value.numel() == 1:
        return value.reshape(batch_shape)

    raise RuntimeError(
        "NParEGO produced an invalid output shape after sample/q reduction. "
        f"value.shape={tuple(value.shape)}, "
        f"expected batch_shape={tuple(batch_shape)}, X.shape={tuple(X.shape)}."
    )


__all__ = ["reduce_nparego_sample_and_q_to_tbatch"]
