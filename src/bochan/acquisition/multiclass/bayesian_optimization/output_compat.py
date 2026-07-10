from __future__ import annotations

import torch
from torch import Tensor

from . import multi_output as _multi_output


_ORIGINAL_FORWARD_ATTR = "_bochan_original_forward_before_output_compat"
_ORIGINAL_DIRECT_FINALIZE_ATTR = (
    "_bochan_original_finalize_before_output_compat"
)
_ORIGINAL_OBJECTIVE_CANONICALIZE_ATTR = (
    "_bochan_original_canonicalize_probability_samples"
)
_ORIGINAL_OBJECTIVE_FORWARD_ATTR = (
    "_bochan_original_forward_before_q_axis_compat"
)


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _shape_endswith(
    shape: torch.Size | tuple[int, ...],
    suffix: tuple[int, ...],
) -> bool:
    if len(suffix) == 0:
        return True
    if len(shape) < len(suffix):
        return False
    return tuple(shape[-len(suffix) :]) == tuple(suffix)


def _input_batch_shape_and_q(X: Tensor) -> tuple[tuple[int, ...], int]:
    """Return public t-batch shape and q represented by ``X``."""
    X = torch.as_tensor(X)
    if X.ndim == 1:
        return (), 1
    if X.ndim == 2:
        return (), int(X.shape[-2])
    return tuple(int(size) for size in X.shape[:-2]), int(X.shape[-2])


def _restore_multiclass_objective_q_axis(
    value: Tensor,
    X: Tensor | None,
    *,
    num_outputs: int | None,
) -> Tensor:
    """Restore a missing q singleton using the objective call's raw input.

    Correlated multitask posteriors can omit the candidate event dimension when
    ``q == 1`` and return probability samples as
    ``sample x t_batch x m x C``. After class reduction the objective then has
    shape ``sample x t_batch x m``. Hypervolume acquisitions interpret the
    penultimate axis as q, so the t-batch is mistaken for q. The raw ``X`` passed
    to the objective is authoritative and allows the missing singleton to be
    restored immediately before qEHVI / qNEHVI subset enumeration.
    """
    if X is None or num_outputs is None or value.ndim < 1:
        return value

    batch_shape, raw_q = _input_batch_shape_and_q(X)
    m = int(num_outputs)

    expected_suffix = batch_shape + (raw_q, m)
    if _shape_endswith(value.shape, expected_suffix):
        return value

    missing_q_suffix = batch_shape + (m,)
    if raw_q == 1 and _shape_endswith(value.shape, missing_q_suffix):
        return value.unsqueeze(-2)

    return value


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
            reduce_dims = tuple(
                i
                for i in range(value.ndim)
                if i < start or i >= start + len(target_shape)
            )
            if len(reduce_dims) > 0:
                value = value.mean(dim=reduce_dims)
            if value.shape == target_shape:
                return value

    # If the result has only a DeepGP sample / latent axis left, it is safer to
    # average it and broadcast than to return a mismatched length that breaks
    # BoTorch's restart bookkeeping. This is a last-resort fallback for cases
    # where an older forward already collapsed the candidate axis.
    if (
        value.ndim == 1
        and len(target_shape) == 1
        and value.numel() != target_shape[0]
    ):
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


def _patch_direct_multioutput_finalize() -> None:
    """Fallback to the shared DeepGP output aligner for direct acquisitions.

    EI / PI / UCB and the direct active-learning acquisitions use the shared
    ``_DirectMultiOutputMulticlassAcqBase._finalize`` method instead of the
    qEHVI wrappers below. A DeepGP posterior can leave only an extra latent axis,
    e.g. ``value.shape == (10,)`` for a single t-batch point. The original
    finalizer cannot distinguish that axis from t-batch and raises before the
    existing output compatibility helper is reached.
    """
    cls = _multi_output._DirectMultiOutputMulticlassAcqBase
    if hasattr(cls, _ORIGINAL_DIRECT_FINALIZE_ATTR):
        return

    original_finalize = cls._finalize
    setattr(cls, _ORIGINAL_DIRECT_FINALIZE_ATTR, original_finalize)

    def _finalize(
        self,
        value: Tensor,
        X: Tensor,
        *,
        name: str,
    ) -> Tensor:
        try:
            return original_finalize(self, value, X, name=name)
        except RuntimeError:
            aligned = _finalize_acq_output(value, X)
            if aligned.shape == tuple(X.shape[:-2]):
                return aligned
            raise

    cls._finalize = _finalize


def _patch_multiclass_probability_objective() -> None:
    """Normalize probability layout and restore an omitted ``q == 1`` axis.

    GPyTorch single-output posteriors may retain a singleton output axis, so a
    dedicated multiclass wrapper can temporarily produce
    ``... x q x 1 x m x C``. In addition, correlated multitask posteriors can
    omit q when it equals one and produce ``... x batch x m x C``. The patched
    objective first normalizes the probability layout and then uses raw ``X`` to
    restore the missing candidate singleton after class reduction.
    """

    cls = _multi_output.MulticlassTargetProbabilityObjective

    if not hasattr(cls, _ORIGINAL_OBJECTIVE_CANONICALIZE_ATTR):
        original_canonicalize = cls._canonicalize_probability_samples
        setattr(
            cls,
            _ORIGINAL_OBJECTIVE_CANONICALIZE_ATTR,
            original_canonicalize,
        )

        def _canonicalize_probability_samples(
            self,
            samples: Tensor,
        ) -> Tensor | None:
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

                # Prefer the documented standard layout. This remains
                # unambiguous through probability normalization when m == C.
                if (
                    normalized.shape[-2] == m
                    and self._looks_like_probabilities_along_dim(
                        normalized,
                        dim=-1,
                    )
                ):
                    return normalized

                # Compatibility layout: ... x q x C x m.
                if (
                    normalized.shape[-1] == m
                    and self._looks_like_probabilities_along_dim(
                        normalized,
                        dim=-2,
                    )
                ):
                    return normalized.movedim(-1, -2)

            return original_canonicalize(self, samples)

        cls._canonicalize_probability_samples = _canonicalize_probability_samples

    if not hasattr(cls, _ORIGINAL_OBJECTIVE_FORWARD_ATTR):
        original_forward = cls.forward
        setattr(cls, _ORIGINAL_OBJECTIVE_FORWARD_ATTR, original_forward)

        def _objective_forward(
            self,
            samples: Tensor,
            X: Tensor | None = None,
        ) -> Tensor:
            value = original_forward(self, samples, X=X)
            return _restore_multiclass_objective_q_axis(
                value,
                X,
                num_outputs=self.num_outputs,
            )

        cls.forward = _objective_forward


def apply_bayesian_optimization_output_compat() -> None:
    """Patch multiclass BO probability and acquisition output shapes in-place."""
    _patch_multiclass_probability_objective()
    _patch_direct_multioutput_finalize()
    _multi_output._finalize_acq_output = _finalize_acq_output
    _wrap_forward(
        _multi_output.qMultiOutputMulticlassExpectedHypervolumeImprovement
    )
    _wrap_forward(
        _multi_output.qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
    )


apply_bayesian_optimization_output_compat()


__all__ = ["apply_bayesian_optimization_output_compat"]
