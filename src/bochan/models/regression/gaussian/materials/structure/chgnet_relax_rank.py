"""CHGNet wrapper around the backend-neutral relaxation ranker."""

from __future__ import annotations

from typing import Any

from .chgnet_relaxation import CHGNetStructureRelaxer
from .relax_rank import (
    MaterialRelaxationRanker,
    ModelFactory,
    RankingCriterion,
    RankingDirection,
    RelaxedStructureRank,
    RelaxedStructureRankingResult,
)


class CHGNetRelaxationRanker(MaterialRelaxationRanker):
    """Relax with CHGNet and rank through the common material relaxation layer."""

    def __init__(self, *, relaxer: CHGNetStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = CHGNetStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "CHGNetRelaxationRanker",
    "ModelFactory",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
]
