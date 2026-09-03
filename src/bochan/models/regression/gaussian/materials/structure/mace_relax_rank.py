"""Backward-compatible MACE wrapper around the backend-neutral relaxation ranker."""

from __future__ import annotations

from typing import Any

from .mace_relaxation import MACEStructureRelaxer
from .relax_rank import (
    MaterialRelaxationRanker,
    ModelFactory,
    RankingCriterion,
    RankingDirection,
    RelaxedStructureRank,
    RelaxedStructureRankingResult,
)


class MACERelaxationRanker(MaterialRelaxationRanker):
    """Relax with MACE and rank through the common material relaxation layer."""

    def __init__(self, *, relaxer: MACEStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = MACEStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "MACERelaxationRanker",
    "ModelFactory",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
]
