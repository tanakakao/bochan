"""Support-selection strategies for sparse candidate repair.

Support selection is intentionally separated from value repair.  Local
strategies such as ``topk`` and ``sample`` can select a support from one
candidate tensor directly.  Acquisition-aware strategies such as
``best_subset`` require optimizer-level search across supports and therefore
must be resolved before the k-sparse repair layer is called.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

ScoreMode = Literal["abs", "value"]
SupportSelection = Literal["topk", "sample", "best_subset"]


def sample_k_without_replacement(
    scores: Tensor,
    k: int,
    *,
    tau: float = 0.2,
    eps: float = 0.05,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``k`` indices without replacement from score-derived probabilities."""
    if k < 1:
        raise ValueError("k must be >= 1.")
    d = scores.shape[-1]
    k_eff = min(k, d)

    scaled = scores / max(float(tau), 1e-12)
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scaled, dim=-1)

    if eps > 0:
        probs = (1.0 - eps) * probs + eps / d

    flat = probs.reshape(-1, d)
    idx = torch.multinomial(flat, num_samples=k_eff, replacement=False, generator=generator)
    return idx.reshape(scores.shape[:-1] + (k_eff,))


def select_support_mask(
    group: Tensor,
    *,
    k: int,
    score: ScoreMode,
    support_selection: SupportSelection,
    sample_tau: float,
    sample_eps: float,
    generator: torch.Generator | None,
) -> Tensor:
    """Return a boolean support mask for one local support-selection strategy.

    ``best_subset`` is deliberately not implemented here.  Its score depends on
    optimizing the acquisition function under each candidate support, so it
    belongs to the optimizer layer rather than candidate repair.
    """
    if support_selection == "best_subset":
        raise NotImplementedError(
            "support_selection='best_subset' requires acquisition-aware support "
            "search and must be resolved before k-sparse repair."
        )
    if support_selection not in {"topk", "sample"}:
        raise ValueError(f"Unknown support_selection: {support_selection}")

    m = group.shape[-1]
    if k <= 0:
        return torch.zeros_like(group, dtype=torch.bool)
    k_eff = min(k, m)
    scores = group.abs() if score == "abs" else group

    if support_selection == "topk":
        idx = scores.topk(k_eff, dim=-1).indices
    else:
        idx = sample_k_without_replacement(
            scores=scores,
            k=k_eff,
            tau=sample_tau,
            eps=sample_eps,
            generator=generator,
        )
    return torch.zeros_like(group, dtype=torch.bool).scatter(-1, idx, True)
