"""ALIGNN-FF wrapper around the backend-neutral relaxation ranker."""

from __future__ import annotations

from typing import Any

from .alignn_ff_relaxation import ALIGNNFFStructureRelaxer
from .relax_rank import (
    MaterialRelaxationRanker,
    ModelFactory,
    RankingCriterion,
    RankingDirection,
    RelaxedStructureRank,
    RelaxedStructureRankingResult,
)


class ALIGNNFFRelaxationRanker(MaterialRelaxationRanker):
    """Relax with ALIGNN-FF and rank through the common relaxation layer."""

    def __init__(self, *, relaxer: ALIGNNFFStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = ALIGNNFFStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "ALIGNNFFRelaxationRanker",
    "ModelFactory",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
]
