from __future__ import annotations

"""q-like alignment helpers for multiclass level-set sampling."""

from torch import Tensor


def align_levelset_q_like(
    target: Tensor,
    *,
    raw_q: int,
    name: str,
) -> Tensor:
    """Align sampled target probabilities from ``q_like`` to the raw q axis.

    Wrapper / heteroscedastic models may collapse the q axis to one value, while
    InputPerturbation-like transforms may expand it. The alignment contract is:

    - identical q: return unchanged;
    - ``q_like == 1``: broadcast across raw q;
    - ``q_like`` is a multiple of raw q: average repeated perturbation slots;
    - raw q is a multiple of ``q_like``: repeat-interleave the available slots.
    """
    q = int(raw_q)
    if q <= 0:
        raise ValueError(f"raw_q must be positive. Got {raw_q}.")
    if target.ndim < 2:
        raise RuntimeError(
            f"{name}: target probabilities must contain q and output axes. "
            f"Got target.shape={tuple(target.shape)}."
        )

    q_like = int(target.shape[-2])
    if q_like == q:
        return target

    m = int(target.shape[-1])
    if q_like == 1 and q > 1:
        return target.expand(*target.shape[:-2], q, m)

    if q_like > q and q_like % q == 0:
        return target.reshape(
            *target.shape[:-2],
            q,
            q_like // q,
            m,
        ).mean(dim=-2)

    if q > q_like and q % q_like == 0:
        return target.repeat_interleave(q // q_like, dim=-2)

    raise RuntimeError(
        f"{name}: cannot align q_like={q_like} to raw q={q}. "
        f"target.shape={tuple(target.shape)}."
    )


__all__ = ["align_levelset_q_like"]
