"""Normalize optimized candidate outputs at the high-level API boundary."""

from __future__ import annotations

from math import prod
from typing import Any

import torch
from torch import Tensor


def _as_tensor(value: Any, *, reference: Tensor | None = None) -> Tensor:
    if torch.is_tensor(value):
        return value
    kwargs = {}
    if reference is not None:
        kwargs = {"device": reference.device, "dtype": reference.dtype}
    return torch.as_tensor(value, **kwargs)


def _candidate_set_scores(
    *,
    candidates: Tensor,
    acq_value: Any,
    acqf: Any | None,
) -> Tensor:
    """Return one scalar score for every leading candidate-set batch."""

    n_sets = int(prod(candidates.shape[:-2]))
    scores = _as_tensor(acq_value, reference=candidates).detach().reshape(-1)
    if scores.numel() == n_sets:
        return scores
    if scores.numel() == 1 and n_sets == 1:
        return scores

    if acqf is not None:
        with torch.no_grad():
            evaluated = _as_tensor(
                acqf(candidates),
                reference=candidates,
            ).detach().reshape(-1)
        if evaluated.numel() == n_sets:
            return evaluated

    raise RuntimeError(
        "Could not align acquisition values with candidate-set batches. "
        f"candidates.shape={tuple(candidates.shape)}, "
        f"acq_value.shape={tuple(scores.shape)}, n_sets={n_sets}."
    )


def select_best_candidate_set(
    candidates: Any,
    acq_value: Any,
    *,
    q: int,
    return_best_only: bool,
    acqf: Any | None = None,
) -> tuple[Any, Any]:
    """Collapse stray model/sample batch axes to the best ``q x d`` set.

    Some posterior implementations expose an internal sample or latent batch axis
    to the candidate optimizer. The high-level API promises one candidate set
    when ``return_best_only=True``; this helper enforces that contract before
    tabular or HTTP adapters see the result.
    """

    if not return_best_only:
        return candidates, acq_value

    candidate_tensor = _as_tensor(candidates)
    if candidate_tensor.ndim <= 2:
        return candidates, acq_value
    if candidate_tensor.shape[-2] != int(q):
        raise RuntimeError(
            "Unexpected optimized candidate shape. Expected the penultimate "
            f"dimension to equal q={q}, got {tuple(candidate_tensor.shape)}."
        )

    scores = _candidate_set_scores(
        candidates=candidate_tensor,
        acq_value=acq_value,
        acqf=acqf,
    )
    candidate_sets = candidate_tensor.reshape(
        -1,
        candidate_tensor.shape[-2],
        candidate_tensor.shape[-1],
    )
    best_index = int(torch.argmax(scores).item())
    return candidate_sets[best_index], scores[best_index]


__all__ = ["select_best_candidate_set"]
