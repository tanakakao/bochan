from __future__ import annotations

import torch
from torch import Tensor

from . import heteroscedastic as _heteroscedastic

_ORIGINAL_ALIGN_LIKE_ATTR = "_bochan_original_align_like_before_input_perturbation"


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _expand_q_like_to_ref(t: Tensor, ref: Tensor) -> Tensor | None:
    """Expand raw q-like noise to an InputPerturbation-expanded reference.

    Examples:
        ``[B, q, C]``     -> ``[B, q * n_w, 1, C]``
        ``[B, q, 1, C]``  -> ``[B, q * n_w, 1, C]``
        ``[1, B, q, C]``  -> ``[B, q * n_w, 1, C]``
    """
    if ref.ndim < 4 or t.shape[-1] != ref.shape[-1]:
        return None

    b = int(ref.shape[0])
    q_ref = int(ref.shape[1])
    c = int(ref.shape[-1])

    work = t
    while work.ndim > 0 and work.shape[0] == 1 and work.ndim > 3:
        work = work.squeeze(0)

    # [B, q, C]
    if work.ndim == 3 and work.shape[0] == b and work.shape[-1] == c:
        q_like = int(work.shape[1])
        if q_like == q_ref:
            return work.unsqueeze(-2).expand_as(ref)
        if q_like > 0 and q_ref % q_like == 0:
            return work.repeat_interleave(q_ref // q_like, dim=1).unsqueeze(-2).expand_as(ref)
        if q_like > q_ref and q_like % q_ref == 0:
            reduced = work.reshape(b, q_ref, q_like // q_ref, c).mean(dim=2)
            return reduced.unsqueeze(-2).expand_as(ref)

    # [B, q, 1, C]
    if work.ndim == 4 and work.shape[0] == b and work.shape[-1] == c and work.shape[-2] == 1:
        q_like = int(work.shape[1])
        if q_like == q_ref:
            return work.expand_as(ref)
        if q_like > 0 and q_ref % q_like == 0:
            return work.repeat_interleave(q_ref // q_like, dim=1).expand_as(ref)
        if q_like > q_ref and q_like % q_ref == 0:
            reduced = work.reshape(b, q_ref, q_like // q_ref, 1, c).mean(dim=2)
            return reduced.expand_as(ref)

    return None


def _align_like(t: Tensor, ref: Tensor) -> Tensor:
    """Align a noise tensor to a reference probability tensor.

    Heteroscedastic noise models are often evaluated on the raw candidate batch,
    while the base multiclass posterior may be evaluated after one-to-many input
    transforms such as InputPerturbation. Common shape pairs are:

    - noise logvar: ``[1, B, 1, C]``
      ref_like:     ``[B, n_w, 1, C]``
    - noise logvar: ``[B, q, C]`` or ``[B, q, 1, C]``
      ref_like:     ``[B, q * n_w, 1, C]``
    """
    t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)

    if t.shape == ref.shape:
        return t

    # Exact element count: reshape is unambiguous.
    if t.numel() == ref.numel():
        return t.reshape_as(ref)

    # Handle q-like raw candidate axis before generic broadcasting, because
    # [B, q, C] must become [B, q * n_w, 1, C], not [1, B, q, C].
    expanded_q = _expand_q_like_to_ref(t, ref)
    if expanded_q is not None:
        return expanded_q

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
        # Try q-like expansion again after dropping leading singleton axes.
        expanded_q = _expand_q_like_to_ref(t_work, ref)
        if expanded_q is not None:
            return expanded_q
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

        # [B, 1, C] -> [B, 1, 1, C] -> [B, W, 1, C].
        # If the middle dimension is not 1, it is q-like and should have been
        # handled by _expand_q_like_to_ref above.
        if t.ndim == ref.ndim - 1 and t.shape[0] == b and t.shape[-1] == c and t.shape[1] == 1:
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
            expanded_q = _expand_q_like_to_ref(t_work, ref)
            if expanded_q is not None:
                return expanded_q
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


def apply_heteroscedastic_alignment() -> None:
    """Patch heteroscedastic multiclass noise alignment in-place."""
    if not hasattr(_heteroscedastic, _ORIGINAL_ALIGN_LIKE_ATTR):
        setattr(_heteroscedastic, _ORIGINAL_ALIGN_LIKE_ATTR, _heteroscedastic._align_like)
    _heteroscedastic._align_like = _align_like


apply_heteroscedastic_alignment()


__all__ = ["apply_heteroscedastic_alignment"]
