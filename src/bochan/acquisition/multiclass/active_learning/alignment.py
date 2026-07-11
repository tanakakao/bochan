from __future__ import annotations

from torch import Tensor

from . import multi_output as _multi_output

_BASE = _multi_output._DirectMultiOutputMulticlassAcqBase
_ORIGINAL_COERCE_SINGLE = getattr(
    _BASE,
    "_bochan_original_coerce_single_output_probs",
    _BASE._coerce_single_output_probs,
)
_ORIGINAL_COERCE_MULTI = getattr(
    _BASE,
    "_bochan_original_coerce_explicit_multi_output_probs",
    _BASE._coerce_explicit_multi_output_probs,
)
_ORIGINAL_ALIGN_SCORE = getattr(
    _BASE,
    "_bochan_original_align_score_per_output_to_raw_X",
    _BASE._align_score_per_output_to_raw_X,
)
_ORIGINAL_ALIGN_JOINT = getattr(
    _BASE,
    "_bochan_original_align_joint_score_per_output_to_raw_X",
    _BASE._align_joint_score_per_output_to_raw_X,
)


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
    """Return the number of sample / latent axes preceding the t-batch axes."""

    return len(prefix) - len(batch_shape)


def _align_pointwise_to_reference(
    value: Tensor,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
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

    if value.numel() == reference.numel():
        return value.reshape_as(reference).to(reference)
    if value.numel() == 1:
        return value.reshape(()).expand_as(reference).to(reference)
    raise RuntimeError(
        f"{name}: cannot align value to reference. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


def _coerce_single_output_probs(
    self,
    probs: Tensor,
    X: Tensor,
    *,
    name: str,
) -> Tensor:
    """Preserve sample and t-batch axes for canonical ``... x q x C`` data."""

    X = self._ensure_q_batch(X)
    batch_shape = tuple(X.shape[:-2])
    q = int(X.shape[-2])

    if probs.ndim >= 2:
        q_like = int(probs.shape[-2])
        prefix = tuple(probs.shape[:-2])
        if (
            q > 0
            and q_like % q == 0
            and _prefix_endswith_batch(prefix, batch_shape)
        ):
            return probs.unsqueeze(-2)

    return _ORIGINAL_COERCE_SINGLE(self, probs, X, name=name)


def _coerce_explicit_multi_output_probs(
    self,
    probs: Tensor,
    X: Tensor,
    *,
    name: str,
) -> Tensor:
    """Preserve canonical ``sample x batch x q x m x C`` probability layout.

    The original subsequence search selected the first matching batch size. If
    ``num_samples == batch_size`` (commonly both 32), it interpreted the Monte
    Carlo sample axis as the t-batch axis and averaged away the actual batch.
    Canonical layouts are unambiguous from the right-hand ``q x m x C`` suffix,
    so the t-batch is resolved as the suffix of the preceding prefix instead.
    """

    X = self._ensure_q_batch(X)
    batch_shape = tuple(X.shape[:-2])
    q = int(X.shape[-2])

    if probs.ndim >= 3:
        q_like = int(probs.shape[-3])
        prefix = tuple(probs.shape[:-3])
        if (
            q > 0
            and q_like % q == 0
            and _prefix_endswith_batch(prefix, batch_shape)
        ):
            return probs

    return _ORIGINAL_COERCE_MULTI(self, probs, X, name=name)


def _align_score_per_output_to_raw_X(
    self,
    score: Tensor,
    raw_X: Tensor,
    *,
    name: str,
) -> Tensor:
    """Align canonical ``leading x batch x q_like x m`` score tensors."""

    raw_X = self._ensure_q_batch(raw_X)
    batch_shape = tuple(raw_X.shape[:-2])
    q = int(raw_X.shape[-2])

    if score.ndim >= 2:
        q_like = int(score.shape[-2])
        m = int(score.shape[-1])
        prefix = tuple(score.shape[:-2])
        if (
            q > 0
            and q_like % q == 0
            and _prefix_endswith_batch(prefix, batch_shape)
        ):
            leading_ndim = _leading_ndim_before_batch(prefix, batch_shape)
            out = score
            if q_like != q:
                out = out.reshape(
                    *prefix,
                    q,
                    q_like // q,
                    m,
                ).mean(dim=-2)
            if leading_ndim > 0:
                out = out.mean(dim=tuple(range(leading_ndim)))
            return out

    return _ORIGINAL_ALIGN_SCORE(self, score, raw_X, name=name)


def _align_joint_score_per_output_to_raw_X(
    self,
    score: Tensor,
    raw_X: Tensor,
    *,
    name: str,
) -> Tensor:
    """Align canonical ``leading x batch x m`` joint-score tensors."""

    raw_X = self._ensure_q_batch(raw_X)
    batch_shape = tuple(raw_X.shape[:-2])

    if score.ndim >= 1:
        prefix = tuple(score.shape[:-1])
        if _prefix_endswith_batch(prefix, batch_shape):
            leading_ndim = _leading_ndim_before_batch(prefix, batch_shape)
            if leading_ndim > 0:
                return score.mean(dim=tuple(range(leading_ndim)))
            return score

    return _ORIGINAL_ALIGN_JOINT(self, score, raw_X, name=name)


def apply_active_learning_alignment() -> None:
    """Patch active-learning probability and score alignment in-place."""

    _multi_output._align_pointwise_to_reference = _align_pointwise_to_reference

    if not hasattr(_BASE, "_bochan_original_coerce_single_output_probs"):
        _BASE._bochan_original_coerce_single_output_probs = _ORIGINAL_COERCE_SINGLE
    if not hasattr(_BASE, "_bochan_original_coerce_explicit_multi_output_probs"):
        _BASE._bochan_original_coerce_explicit_multi_output_probs = _ORIGINAL_COERCE_MULTI
    if not hasattr(_BASE, "_bochan_original_align_score_per_output_to_raw_X"):
        _BASE._bochan_original_align_score_per_output_to_raw_X = _ORIGINAL_ALIGN_SCORE
    if not hasattr(_BASE, "_bochan_original_align_joint_score_per_output_to_raw_X"):
        _BASE._bochan_original_align_joint_score_per_output_to_raw_X = _ORIGINAL_ALIGN_JOINT

    _BASE._coerce_single_output_probs = _coerce_single_output_probs
    _BASE._coerce_explicit_multi_output_probs = _coerce_explicit_multi_output_probs
    _BASE._align_score_per_output_to_raw_X = _align_score_per_output_to_raw_X
    _BASE._align_joint_score_per_output_to_raw_X = (
        _align_joint_score_per_output_to_raw_X
    )


apply_active_learning_alignment()


__all__ = ["apply_active_learning_alignment"]
