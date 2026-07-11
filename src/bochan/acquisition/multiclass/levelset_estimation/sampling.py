from __future__ import annotations

from torch import Tensor

from . import multi_output as _multi_output

_ORIGINAL_SAMPLE_TARGET_ATTR = "_bochan_original_sample_target_probs_before_q_like_support"


def _sample_target_probs(self, X: Tensor) -> Tensor:
    """Sample target probabilities with robust q_like -> raw q alignment.

    Some heteroscedastic / wrapper multiclass models collapse a concatenated
    pending+candidate q-batch and return ``q_like=1`` even when raw ``q > 1``.
    For joint level-set acquisition we need a ``... x q x m`` tensor. If the
    returned q_like is 1, broadcast it across raw q. If q_like is a multiple of
    q, average perturbation-like repeats. If raw q is a multiple of q_like,
    repeat-interleave the available values.
    """
    Xq = self._ensure_q_batch(X)
    posterior = self._get_multiclass_probability_posterior(Xq)
    samples = posterior.rsample(self.sampler.sample_shape)
    target = self._target_prob_per_output(samples)

    q = int(Xq.shape[-2])
    q_like = int(target.shape[-2])
    if q_like == q:
        return target

    m = int(target.shape[-1])
    if q_like == 1 and q > 1:
        return target.expand(*target.shape[:-2], q, m)

    if q_like > q and q_like % q == 0:
        return target.reshape(*target.shape[:-2], q, q_like // q, m).mean(dim=-2)

    if q > q_like and q % q_like == 0:
        return target.repeat_interleave(q // q_like, dim=-2)

    raise RuntimeError(
        f"{self.__class__.__name__}: cannot align q_like={q_like} to raw q={q}. "
        f"target.shape={tuple(target.shape)}."
    )


def configure_levelset_sampling() -> None:
    """Patch joint multiclass level-set q_like alignment in-place."""
    cls = _multi_output.qMultiOutputMulticlassJointLatentStraddleAcquisition
    if not hasattr(cls, _ORIGINAL_SAMPLE_TARGET_ATTR):
        setattr(cls, _ORIGINAL_SAMPLE_TARGET_ATTR, cls._sample_target_probs)
    cls._sample_target_probs = _sample_target_probs


configure_levelset_sampling()


__all__ = ["configure_levelset_sampling"]
