from __future__ import annotations

import torch
from torch import Tensor

from . import multi_output as _multi_output


def _align_pointwise_to_reference(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    """Align pointwise acquisition tensors under extra sample / latent axes.

    DeepGP and some posterior wrappers may leave an extra leading sample-like
    dimension on one side only, e.g.

    - value: ``[batch, q, m]``
    - reference: ``[S, batch, q, m]``

    For pointwise active-learning scores that extra leading dimension is not a
    separate candidate dimension. This helper therefore supports suffix matching
    and broadcasts / averages leading dimensions before falling back to the
    original q-like alignment rules.
    """
    if value.ndim >= 1 and value.shape[-1] == 1 and reference.ndim >= 1 and reference.shape[-1] != 1:
        value = value.squeeze(-1)

    # If one tensor has sample / latent leading axes and the other does not,
    # align by suffix. This is the DeepGP failure pattern:
    # value=(128, 1, 2), reference=(10, 128, 1, 2).
    if value.shape == reference.shape:
        return value.to(reference)
    if value.ndim < reference.ndim and tuple(value.shape) == tuple(reference.shape[-value.ndim:]):
        view_shape = (1,) * (reference.ndim - value.ndim) + tuple(value.shape)
        return value.reshape(view_shape).expand_as(reference).to(reference)
    if reference.ndim < value.ndim and tuple(reference.shape) == tuple(value.shape[-reference.ndim:]):
        leading_dims = tuple(range(value.ndim - reference.ndim))
        return value.mean(dim=leading_dims).to(reference)

    while value.ndim > reference.ndim:
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        else:
            value = value.mean(dim=-1)

    if value.shape == reference.shape:
        return value.to(reference)
    if value.shape == reference.shape[:-1]:
        return value.unsqueeze(-1).expand_as(reference).to(reference)

    if value.ndim == reference.ndim and value.shape[:-1] == reference.shape[:-1]:
        q_ref = int(reference.shape[-1])
        q_value = int(value.shape[-1])
        if q_value > 0 and q_ref % q_value == 0:
            return value.repeat_interleave(q_ref // q_value, dim=-1).to(reference)
        if q_ref > 0 and q_value % q_ref == 0:
            return value.reshape(*reference.shape[:-1], q_ref, q_value // q_ref).mean(dim=-1).to(reference)

    # A second chance for suffix matching after q-like reductions.
    if value.ndim < reference.ndim and tuple(value.shape) == tuple(reference.shape[-value.ndim:]):
        view_shape = (1,) * (reference.ndim - value.ndim) + tuple(value.shape)
        return value.reshape(view_shape).expand_as(reference).to(reference)
    if reference.ndim < value.ndim and tuple(reference.shape) == tuple(value.shape[-reference.ndim:]):
        leading_dims = tuple(range(value.ndim - reference.ndim))
        return value.mean(dim=leading_dims).to(reference)

    if value.numel() == reference.numel():
        return value.reshape_as(reference).to(reference)
    if value.numel() == 1:
        return value.reshape(()).expand_as(reference).to(reference)
    raise RuntimeError(
        f"{name}: cannot align value to reference. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


def apply_active_learning_alignment_compat() -> None:
    """Patch active-learning multi-output alignment helper in-place."""
    _multi_output._align_pointwise_to_reference = _align_pointwise_to_reference


apply_active_learning_alignment_compat()


__all__ = ["apply_active_learning_alignment_compat"]
