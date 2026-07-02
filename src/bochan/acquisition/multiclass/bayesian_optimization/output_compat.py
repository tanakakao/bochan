from __future__ import annotations

import torch
from torch import Tensor

from . import multi_output as _multi_output


_ORIGINAL_FORWARD_ATTR = "_bochan_original_forward_before_output_compat"
_ORIGINAL_OBJECTIVE_CANONICALIZE_ATTR = (
    "_bochan_original_canonicalize_probability_samples"
)


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
    ``[raw_samples]``. This helper preserves the target t-batch suffix and
    averages only leading sample-like dimensions.
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

    if len(target_shape) == 0:
        return value.mean()

    # Target shape appears contiguously inside value. Preserve it and average
    # all other axes.
    shape = tuple(value.shape)
    for start in range(0, len(shape) - len(target_shape) + 1):
        if shape[start : start + len(target_shape)] == target_shape:
            reduce_dims = tuple(i for i in range(value.ndim) if i < start or i >= start + len(target_shape))
            if len(reduce_dims) > 0:
                value = value.mean(dim=reduce_dims)
            if value.shape == target_shape:
                return value

    # If the result has only a DeepGP sample / latent axis left, it is safer to
    # average it and broadcast than to return a mismatched length that breaks
    # BoTorch's restart bookkeeping. This is a last-resort fallback for cases
    # where an older forward already collapsed the candidate axis.
    if value.ndim == 1 and len(target_shape) == 1 and value.numel() != target_shape[0]:
        return value.mean().expand(*target_shape)

    while value.ndim > len(target_shape):
        value = value.mean(dim=0)
        if value.shape == target_shape:
            return value

    if value.numel() == _prod(target_shape):
        return value.reshape(target_shape)
    if value.numel() == 1:
        return value.reshape(()).expand(*target_shape)
    return value


def _wrap_forward(cls) -> None:
    if hasattr(cls, _ORIGINAL_FORWARD_ATTR):
        return

    original_forward = cls.forward
    setattr(cls, _ORIGINAL_FORWARD_ATTR, original_forward)

    def _forward(self, X: Tensor) -> Tensor:
        value = original_forward(self, X)
        X_raw = _multi_output.ensure_q_batch(X)
        return _finalize_acq_output(value, X_raw)

    cls.forward = _forward


def _patch_multiclass_probability_objective() -> None:
    """Normalize ``... x q x m x C`` even when ``m == C``.

    GPyTorch single-output posteriors may retain a singleton output axis, so a
    dedicated multiclass wrapper can temporarily produce
    ``... x q x 1 x m x C``. In addition, the original layout detector used
    unequal axis sizes to distinguish ``m`` from ``C`` and therefore failed when
    the number of objectives equalled the number of classes. The patched method
    removes only the singleton between q and m, then detects probabilities by
    their normalization rather than by axis-size inequality.
    """

    cls = _multi_output.MulticlassTargetProbabilityObjective
    if hasattr(cls, _ORIGINAL_OBJECTIVE_CANONICALIZE_ATTR):
        return

    original = cls._canonicalize_probability_samples
    setattr(cls, _ORIGINAL_OBJECTIVE_CANONICALIZE_ATTR, original)

    def _canonicalize_probability_samples(self, samples: Tensor) -> Tensor | None:
        num_outputs = self.num_outputs
        if num_outputs is not None and samples.ndim >= 4:
            m = int(num_outputs)
            normalized = samples

            # Standard wrapper shape should be ... x q x m x C. Remove only
            # GPyTorch's extra singleton output axis: ... x q x 1 x m x C.
            if (
                normalized.ndim >= 5
                and normalized.shape[-3] == 1
                and normalized.shape[-2] == m
            ):
                normalized = normalized.squeeze(-3)

            # Prefer the documented standard layout. This remains unambiguous
            # through probability normalization even when m == C.
            if (
                normalized.shape[-2] == m
                and self._looks_like_probabilities_along_dim(normalized, dim=-1)
            ):
                return normalized

            # Compatibility layout: ... x q x C x m.
            if (
                normalized.shape[-1] == m
                and self._looks_like_probabilities_along_dim(normalized, dim=-2)
            ):
                return normalized.movedim(-1, -2)

        return original(self, samples)

    cls._canonicalize_probability_samples = _canonicalize_probability_samples


def apply_bayesian_optimization_output_compat() -> None:
    """Patch multiclass BO probability and acquisition output shapes in-place."""
    _patch_multiclass_probability_objective()
    _multi_output._finalize_acq_output = _finalize_acq_output
    _wrap_forward(_multi_output.qMultiOutputMulticlassExpectedHypervolumeImprovement)
    _wrap_forward(_multi_output.qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement)


apply_bayesian_optimization_output_compat()


__all__ = ["apply_bayesian_optimization_output_compat"]
