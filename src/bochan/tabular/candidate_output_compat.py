from __future__ import annotations

from functools import wraps
from math import prod
from typing import Any

import torch
from torch import Tensor

from bochan.api import BayesianOptimizer, CandidateResult


_ORIGINAL_CANDIDATE_ATTR = "_bochan_candidate_before_tabular_output_compat"


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
            evaluated = _as_tensor(acqf(candidates), reference=candidates).detach().reshape(-1)
        if evaluated.numel() == n_sets:
            return evaluated

    raise RuntimeError(
        "Could not align acquisition values with candidate-set batches. "
        f"candidates.shape={tuple(candidates.shape)}, "
        f"acq_value.shape={tuple(scores.shape)}, n_sets={n_sets}."
    )


def _select_best_candidate_set(
    candidates: Any,
    acq_value: Any,
    *,
    q: int,
    return_best_only: bool,
    acqf: Any | None = None,
) -> tuple[Any, Any]:
    """Collapse stray model/sample batch axes to the best ``q x d`` candidate set.

    DeepGP classification acquisitions can expose an internal sample or latent
    batch axis to an optimizer. Even with ``return_best_only=True``, this may
    produce ``batch x q x d`` candidates. The public high-level API promises one
    candidate set, so select the set with the largest acquisition value.
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
    candidate_sets = candidate_tensor.reshape(-1, candidate_tensor.shape[-2], candidate_tensor.shape[-1])
    best_index = int(torch.argmax(scores).item())
    best_candidates = candidate_sets[best_index]
    best_value = scores[best_index]
    return best_candidates, best_value


def apply_tabular_candidate_output_compat() -> None:
    """Patch the core candidate API before tabular DataFrame conversion."""
    cls = BayesianOptimizer
    if hasattr(cls, _ORIGINAL_CANDIDATE_ATTR):
        return

    original_candidate = cls.candidate
    setattr(cls, _ORIGINAL_CANDIDATE_ATTR, original_candidate)

    @wraps(original_candidate)
    def _candidate(self, acq_config, opt_config, **kwargs):
        return_result = bool(kwargs.get("return_result", False))
        result = original_candidate(self, acq_config, opt_config, **kwargs)

        if return_result:
            if not isinstance(result, CandidateResult):
                return result
            candidates, acq_value = _select_best_candidate_set(
                result.candidates,
                result.acq_value,
                q=opt_config.q,
                return_best_only=opt_config.return_best_only,
                acqf=result.acqf,
            )
            result.candidates = candidates
            result.acq_value = acq_value
            return result

        candidates, acq_value = result
        acqf = None
        if getattr(self, "history", None):
            acqf = getattr(self.history[-1], "acqf", None)
        candidates, acq_value = _select_best_candidate_set(
            candidates,
            acq_value,
            q=opt_config.q,
            return_best_only=opt_config.return_best_only,
            acqf=acqf,
        )
        if getattr(self, "history", None):
            self.history[-1].candidates = candidates
            self.history[-1].acq_value = acq_value
        return candidates, acq_value

    cls.candidate = _candidate


__all__ = [
    "_select_best_candidate_set",
    "apply_tabular_candidate_output_compat",
]
