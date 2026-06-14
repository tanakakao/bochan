from __future__ import annotations

import torch
from torch import Tensor

from . import heteroscedastic as _heteroscedastic


_ORIGINAL_ALIGN_LIKE_ATTR = "_bochan_original_align_like_before_input_perturbation_compat"


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _align_like(t: Tensor, ref: Tensor) -> Tensor:
    """Align a noise tensor to a reference probability tensor.

    Heteroscedastic noise models are often evaluated on the raw candidate batch,
    while the base multiclass posterior may be evaluated after one-to-many input
    transforms such as InputPerturbation. A common shape pair is:

    - noise logvar: ``[1, batch, 1, C]``
    - ref_like:     ``[batch, n_w, 1, C]``

    In that case the raw batch axis must be moved to the reference batch axis and
    the perturbation axis must be broadcast.
    """
    t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)

    if t.shape == ref.shape:
        return t

    # Exact element count: reshape is unambiguous.
    if t.numel() == ref.numel():
        return t.reshape_as(ref)

    # Standard broadcasting may already work.
    try:
        return t.expand_as(ref)
    except RuntimeError:
        pass

    # Remove leading singleton axes, then try suffix broadcast.
    t_work = t
    while t_work.ndim > 0 and t_work.shape[0] == 1 and t_work.ndim >= ref.ndim:
        t_work = t_work.squeeze(0)
    if t_work.ndim <= ref.ndim:
        view_shape = (1,) * (ref.ndim - t_work.ndim) + tuple(t_work.shape)
        try:
            return t_work.reshape(view_shape).expand_as(ref)
        except RuntimeError:
            pass

    # InputPerturbation pattern:
    #   t   = [1, B, 1, C] or [B, 1, C]
    #   ref = [B, W, 1, C]
    if ref.ndim >= 4 and t.shape[-1] == ref.shape[-1]:
        b = int(ref.shape[0])
        c = int(ref.shape[-1])

        # [1, B, 1, C] -> [B, 1, 1, C] -> [B, W, 1, C]
        if t.ndim == ref.ndim and t.shape[0] == 1 and t.shape[1] == b and t.shape[-1] == c:
            moved = t.squeeze(0)
            while moved.ndim < ref.ndim - 1:
                moved = moved.unsqueeze(-2)
            moved = moved.reshape(b, *([1] * (ref.ndim - 2)), c)
            return moved.expand_as(ref)

        # [B, 1, C] -> [B, 1, 1, C] -> [B, W, 1, C]
        if t.ndim == ref.ndim - 1 and t.shape[0] == b and t.shape[-1] == c:
            moved = t
            while moved.ndim < ref.ndim:
                moved = moved.unsqueeze(-2)
            return moved.expand_as(ref)

        # [B, C] -> [B, 1, 1, C] -> [B, W, 1, C]
        if t.ndim == 2 and t.shape[0] == b and t.shape[-1] == c:
            moved = t.reshape(b, *([1] * (ref.ndim - 2)), c)
            return moved.expand_as(ref)

    # If t has extra axes, average them until a broadcastable representation is found.
    t_work = t
    while t_work.ndim > 0:
        if t_work.ndim <= ref.ndim:
            view_shape = (1,) * (ref.ndim - t_work.ndim) + tuple(t_work.shape)
            try:
                return t_work.reshape(view_shape).expand_as(ref)
            except RuntimeError:
                pass
        # Prefer reducing singleton / sample-like leading axes first.
        if t_work.shape[0] == 1:
            t_work = t_work.squeeze(0)
        else:
            t_work = t_work.mean(dim=0)

    if t.numel() == 1:
        return t.reshape(()).expand_as(ref)

    raise RuntimeError(
        "Could not align heteroscedastic noise tensor to reference. "
        f"t.shape={tuple(t.shape)}, ref.shape={tuple(ref.shape)}."
    )


def apply_heteroscedastic_alignment_compat() -> None:
    """Patch heteroscedastic multiclass noise alignment in-place."""
    if not hasattr(_heteroscedastic, _ORIGINAL_ALIGN_LIKE_ATTR):
        setattr(_heteroscedastic, _ORIGINAL_ALIGN_LIKE_ATTR, _heteroscedastic._align_like)
    _heteroscedastic._align_like = _align_like


apply_heteroscedastic_alignment_compat()


__all__ = ["apply_heteroscedastic_alignment_compat"]
