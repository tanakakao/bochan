"""Native shape-safe multiclass active-learning acquisitions.

The implementation core is kept in ``_multi_output_core``. This public module
owns the canonical probability / score alignment contract directly through
normal inheritance rather than import-time class or module mutation.
"""

from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from . import _multi_output_core as _core

ReductionType = _core.ReductionType
OutputReductionType = _core.OutputReductionType
OutputModeType = _core.OutputModeType
LargeQStrategy = _core.LargeQStrategy


def _prefix_endswith_batch(
    prefix: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> bool:
    """Return whether a canonical tensor prefix ends with the t-batch shape."""

    if len(batch_shape) == 0:
        return True
    if len(prefix) < len(batch_shape):
        return False
    return tuple(prefix[-len(batch_shape) :]) == tuple(batch_shape)


def _leading_ndim_before_batch(
    prefix: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> int:
    """Return sample / latent axes preceding the t-batch axes."""

    return len(prefix) - len(batch_shape)


def _align_pointwise_to_reference(
    value: Tensor,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
    """Align pointwise tensors while preserving leading sample / latent axes."""

    if (
        value.ndim >= 1
        and value.shape[-1] == 1
        and reference.ndim >= 1
        and reference.shape[-1] != 1
    ):
        value = value.squeeze(-1)

    if value.shape == reference.shape:
        return value.to(reference)
    if (
        value.ndim < reference.ndim
        and tuple(value.shape) == tuple(reference.shape[-value.ndim :])
    ):
        view_shape = (1,) * (reference.ndim - value.ndim) + tuple(value.shape)
        return value.reshape(view_shape).expand_as(reference).to(reference)
    if (
        reference.ndim < value.ndim
        and tuple(reference.shape) == tuple(value.shape[-reference.ndim :])
    ):
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
            return value.reshape(
                *reference.shape[:-1],
                q_ref,
                q_value // q_ref,
            ).mean(dim=-1).to(reference)

    if value.numel() == reference.numel():
        return value.reshape_as(reference).to(reference)
    if value.numel() == 1:
        return value.reshape(()).expand_as(reference).to(reference)
    raise RuntimeError(
        f"{name}: cannot align value to reference. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


class _NativeMulticlassAlignmentMixin:
    """Canonical multiclass probability and score shape normalization."""

    def _coerce_single_output_probs(
        self,
        probs: Tensor,
        X: Tensor,
        *,
        name: str,
    ) -> Tensor:
        X = self._ensure_q_batch(X)
        batch_shape = tuple(X.shape[:-2])
        q = int(X.shape[-2])
        if probs.ndim >= 2:
            q_like = int(probs.shape[-2])
            prefix = tuple(probs.shape[:-2])
            if q > 0 and q_like % q == 0 and _prefix_endswith_batch(prefix, batch_shape):
                return probs.unsqueeze(-2)
        return super()._coerce_single_output_probs(probs, X, name=name)

    def _coerce_explicit_multi_output_probs(
        self,
        probs: Tensor,
        X: Tensor,
        *,
        name: str,
    ) -> Tensor:
        X = self._ensure_q_batch(X)
        batch_shape = tuple(X.shape[:-2])
        q = int(X.shape[-2])
        if probs.ndim >= 3:
            q_like = int(probs.shape[-3])
            prefix = tuple(probs.shape[:-3])
            if q > 0 and q_like % q == 0 and _prefix_endswith_batch(prefix, batch_shape):
                return probs
        return super()._coerce_explicit_multi_output_probs(probs, X, name=name)

    def _align_score_per_output_to_raw_X(
        self,
        score: Tensor,
        raw_X: Tensor,
        *,
        name: str,
    ) -> Tensor:
        raw_X = self._ensure_q_batch(raw_X)
        batch_shape = tuple(raw_X.shape[:-2])
        q = int(raw_X.shape[-2])
        if score.ndim >= 2:
            q_like = int(score.shape[-2])
            m = int(score.shape[-1])
            prefix = tuple(score.shape[:-2])
            if q > 0 and q_like % q == 0 and _prefix_endswith_batch(prefix, batch_shape):
                leading_ndim = _leading_ndim_before_batch(prefix, batch_shape)
                out = score
                if q_like != q:
                    out = out.reshape(*prefix, q, q_like // q, m).mean(dim=-2)
                if leading_ndim > 0:
                    out = out.mean(dim=tuple(range(leading_ndim)))
                return out
        return super()._align_score_per_output_to_raw_X(score, raw_X, name=name)

    def _align_joint_score_per_output_to_raw_X(
        self,
        score: Tensor,
        raw_X: Tensor,
        *,
        name: str,
    ) -> Tensor:
        raw_X = self._ensure_q_batch(raw_X)
        batch_shape = tuple(raw_X.shape[:-2])
        if score.ndim >= 1:
            prefix = tuple(score.shape[:-1])
            if _prefix_endswith_batch(prefix, batch_shape):
                leading_ndim = _leading_ndim_before_batch(prefix, batch_shape)
                if leading_ndim > 0:
                    return score.mean(dim=tuple(range(leading_ndim)))
                return score
        return super()._align_joint_score_per_output_to_raw_X(score, raw_X, name=name)

    def _pointwise_score_to_value(
        self,
        score_per_output: Tensor,
        raw_X: Tensor,
        Xt: Tensor,
    ) -> Tensor:
        score_per_output = self._align_score_per_output_to_raw_X(
            score_per_output,
            raw_X,
            name=f"{self.__class__.__name__}.score_per_output",
        )
        score = self._aggregate_outputs(score_per_output)
        pending = _align_pointwise_to_reference(
            self._pending_penalty_per_point(Xt),
            score,
            name=f"{self.__class__.__name__}.pending",
        )
        observed = _align_pointwise_to_reference(
            self._observed_penalty_per_point(Xt),
            score,
            name=f"{self.__class__.__name__}.observed",
        )
        score = score - pending - observed
        score = self._apply_objective(score, raw_X=raw_X, expanded_X=Xt)
        value = score if score.shape == raw_X.shape[:-2] else self._reduce_q(score)
        return value - self._same_batch_penalty(Xt)


class _DirectMultiOutputMulticlassAcqBase(
    _NativeMulticlassAlignmentMixin,
    _core._DirectMultiOutputMulticlassAcqBase,
):
    """Direct multiclass base with native shape alignment."""


class qMultiOutputMulticlassPredictiveEntropy(
    _DirectMultiOutputMulticlassAcqBase,
    _core.qMultiOutputMulticlassPredictiveEntropy,
):
    pass


class qMultiOutputMulticlassProbabilityVariance(
    _DirectMultiOutputMulticlassAcqBase,
    _core.qMultiOutputMulticlassProbabilityVariance,
):
    pass


class qMultiOutputMulticlassMarginUncertainty(
    _DirectMultiOutputMulticlassAcqBase,
    _core.qMultiOutputMulticlassMarginUncertainty,
):
    pass


class qMultiOutputMulticlassBALD(
    _DirectMultiOutputMulticlassAcqBase,
    _core.qMultiOutputMulticlassBALD,
):
    pass


class qMultiOutputMulticlassJointBALD(
    qMultiOutputMulticlassBALD,
    _core.qMultiOutputMulticlassJointBALD,
):
    pass


class qMultiOutputMulticlassGreedyJointBALD(
    qMultiOutputMulticlassJointBALD,
    _core.qMultiOutputMulticlassGreedyJointBALD,
):
    pass


class qMultiOutputMulticlassIntegratedPosteriorVarianceProxy(
    _DirectMultiOutputMulticlassAcqBase,
    _core.qMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_per_output(raw_X, Xt)
        integrated_score = _align_pointwise_to_reference(
            integrated_score,
            local_score,
            name=f"{self.__class__.__name__}.integrated",
        )
        score_per_output = self.local_weight * local_score + self.integrated_weight * integrated_score
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "ReductionType",
    "OutputReductionType",
    "OutputModeType",
    "LargeQStrategy",
    "_DirectMultiOutputMulticlassAcqBase",
    "qMultiOutputMulticlassPredictiveEntropy",
    "qMultiOutputMulticlassProbabilityVariance",
    "qMultiOutputMulticlassMarginUncertainty",
    "qMultiOutputMulticlassBALD",
    "qMultiOutputMulticlassJointBALD",
    "qMultiOutputMulticlassGreedyJointBALD",
    "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
