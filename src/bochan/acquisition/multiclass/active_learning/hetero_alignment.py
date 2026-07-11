from __future__ import annotations

import torch
from torch import Tensor

from . import hetero_multi_output as _hetero_multi_output

_ORIGINAL_COMBINE_ATTR = "_bochan_original_combine_score_and_weight_before_alignment"


def _align_score_to_weight(score: Tensor, weight: Tensor) -> Tensor:
    """Align hetero score tensor to the noise-weight tensor shape.

    Hetero joint acquisitions may produce per-output score with output axis first,
    e.g. ``[m, batch]``, while noise weights are ``[batch, m]``. This helper
    normalizes such cases before noise weighting.
    """
    score = torch.as_tensor(score, device=weight.device, dtype=weight.dtype)

    if score.shape == weight.shape:
        return score

    # Common failure pattern: score=[m, batch], weight=[batch, m].
    if score.ndim == 2 and weight.ndim == 2 and score.T.shape == weight.shape:
        return score.T

    # More general output-first pattern: score=[m, *batch], weight=[*batch, m].
    if score.ndim == weight.ndim and score.ndim >= 2:
        if score.shape[0] == weight.shape[-1] and tuple(score.shape[1:]) == tuple(weight.shape[:-1]):
            return score.movedim(0, -1)

    # If score lacks leading sample/batch axes, broadcast it to weight.
    if score.ndim < weight.ndim and tuple(score.shape) == tuple(weight.shape[-score.ndim:]):
        view_shape = (1,) * (weight.ndim - score.ndim) + tuple(score.shape)
        return score.reshape(view_shape).expand_as(weight)

    # If score has extra leading axes, average them out if the suffix matches.
    if score.ndim > weight.ndim and tuple(score.shape[-weight.ndim:]) == tuple(weight.shape):
        leading_dims = tuple(range(score.ndim - weight.ndim))
        return score.mean(dim=leading_dims)

    if score.numel() == weight.numel():
        return score.reshape_as(weight)

    if score.numel() == 1:
        return score.reshape(()).expand_as(weight)

    # Last-resort: if score ends with the same output dimension, collapse all
    # non-output axes and broadcast over weight's batch shape.
    if score.ndim >= 1 and score.shape[-1] == weight.shape[-1]:
        while score.ndim > 1:
            score = score.mean(dim=0)
        return score.reshape(*([1] * (weight.ndim - 1)), weight.shape[-1]).expand_as(weight)

    # Last-resort output-first version.
    if score.ndim >= 1 and score.shape[0] == weight.shape[-1]:
        while score.ndim > 1:
            score = score.mean(dim=-1)
        return score.reshape(*([1] * (weight.ndim - 1)), weight.shape[-1]).expand_as(weight)

    return score


def _combine_score_and_weight(self, score: Tensor, weight: Tensor) -> Tensor:
    weight = torch.as_tensor(weight)
    score = _align_score_to_weight(score, weight)
    weight = weight.to(score)
    if self.noise_combine == "multiply":
        return score * weight
    if self.noise_combine in {"subtract", "add"}:
        return score - (1.0 - weight)
    raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")


def apply_hetero_noise_alignment() -> None:
    """Patch hetero multi-output score/noise-weight alignment in-place."""
    cls = _hetero_multi_output._HeteroMultiOutputMulticlassMixin
    if not hasattr(cls, _ORIGINAL_COMBINE_ATTR):
        setattr(cls, _ORIGINAL_COMBINE_ATTR, cls._combine_score_and_weight)
    cls._combine_score_and_weight = _combine_score_and_weight


apply_hetero_noise_alignment()


__all__ = ["apply_hetero_noise_alignment"]
