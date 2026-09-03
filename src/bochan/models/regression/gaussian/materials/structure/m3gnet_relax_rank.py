"""M3GNet wrapper around the backend-neutral relaxation ranker."""

from __future__ import annotations

from typing import Any

from .m3gnet_relaxation import M3GNetStructureRelaxer
from .relax_rank import (
    MaterialRelaxationRanker,
    ModelFactory,
    RankingCriterion,
    RankingDirection,
    RelaxedStructureRank,
    RelaxedStructureRankingResult,
)


class M3GNetRelaxationRanker(MaterialRelaxationRanker):
    """Relax with M3GNet and rank through the common material relaxation layer."""

    def __init__(self, *, relaxer: M3GNetStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = M3GNetStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "M3GNetRelaxationRanker",
    "ModelFactory",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
]
