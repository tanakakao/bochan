"""Candidate scoring for repaired element-constrained compositions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from ...data.conversion import dataframe_to_tensors


class CompositionElementConstraintCandidateReranker:
    """Score repaired candidates independently from the optimizer facade."""

    @staticmethod
    def requested_q(opt_config: Any, kwargs: Mapping[str, Any]) -> int:
        direct = kwargs.get("q")
        if isinstance(direct, int):
            return max(1, direct)
        if isinstance(opt_config, Mapping) and isinstance(opt_config.get("q"), int):
            return max(1, int(opt_config["q"]))
        configured = getattr(opt_config, "q", None)
        return max(1, int(configured)) if isinstance(configured, int) else 1

    @staticmethod
    def rerank(
        candidates: Any,
        acqf: Any,
        requested_q: int,
        *,
        transform_compositions: Callable[[Any], Any],
        data_config: Any,
        feature_names: Sequence[Any],
    ) -> tuple[Any, Any]:
        unique = candidates.drop_duplicates().reset_index(drop=True)
        transformed = transform_compositions(unique)
        scoring_config = replace(
            data_config,
            input_cols=feature_names,
            target_cols=None,
        )
        X = dataframe_to_tensors(transformed, scoring_config).X
        with torch.no_grad():
            try:
                scores = acqf(X.unsqueeze(-2))
            except (RuntimeError, ValueError, TypeError):
                scores = acqf(X)
        scores = scores.detach().reshape(-1)
        if scores.numel() != len(unique):
            raise ValueError(
                "The acquisition function did not return one score per repaired candidate."
            )
        order = torch.argsort(scores, descending=True)[:requested_q]
        indices = order.detach().cpu().numpy().tolist()
        return unique.iloc[indices].reset_index(drop=True), scores[order]


__all__ = ["CompositionElementConstraintCandidateReranker"]
