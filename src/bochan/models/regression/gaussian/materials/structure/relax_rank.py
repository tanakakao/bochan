"""Relax structures with any material backend and rank them with a posterior."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from ..common.relaxation import (
    MaterialStructureRelaxer,
    StructureRelaxationResult,
    validate_structure_relaxer,
)

RankingDirection = Literal["minimize", "maximize"]
RankingCriterion = Literal["posterior_mean", "ucb"]
ModelFactory = Callable[[tuple[dict[str, Any], ...]], Any]


@dataclass(frozen=True)
class RelaxedStructureRank:
    """One relaxed candidate and its posterior ranking diagnostics."""

    rank: int
    source_index: int
    relaxation: StructureRelaxationResult
    posterior_mean: float
    posterior_std: float
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "source_index": self.source_index,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "score": self.score,
            "relaxation": self.relaxation.as_dict(),
        }


@dataclass(frozen=True)
class RelaxedStructureRankingResult:
    """Serializable ranking result for a batch of relaxed structures."""

    candidates: tuple[RelaxedStructureRank, ...]
    direction: RankingDirection
    criterion: RankingCriterion
    beta: float

    @property
    def best(self) -> RelaxedStructureRank:
        if not self.candidates:
            raise ValueError("No ranked candidates are available.")
        return self.candidates[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "criterion": self.criterion,
            "beta": self.beta,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _validate_structures(structures: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
        raise TypeError("structures must be a non-empty sequence.")
    resolved = tuple(structures)
    if not resolved:
        raise ValueError("structures must contain at least one structure.")
    return resolved


def _resolve_process_X(process_X: Tensor | None, *, n: int) -> Tensor:
    if process_X is None:
        return torch.empty(n, 0, dtype=torch.double)
    if not torch.is_tensor(process_X):
        raise TypeError("process_X must be a Tensor when provided.")
    if process_X.ndim != 2 or process_X.shape[0] != n:
        raise ValueError(f"process_X must have shape [{n}, process_dim].")
    if not torch.isfinite(process_X).all():
        raise ValueError("process_X must contain only finite values.")
    return process_X


def _posterior_statistics(model: Any, X: Tensor) -> tuple[Tensor, Tensor]:
    posterior_fn = getattr(model, "posterior", None)
    if not callable(posterior_fn):
        raise TypeError("model_factory must return an object exposing posterior(X).")
    posterior = posterior_fn(X)
    mean = getattr(posterior, "mean", None)
    variance = getattr(posterior, "variance", None)
    if not torch.is_tensor(mean) or not torch.is_tensor(variance):
        raise TypeError("posterior must expose Tensor mean and variance.")
    if mean.ndim == 1:
        mean = mean.unsqueeze(-1)
    if variance.ndim == 1:
        variance = variance.unsqueeze(-1)
    if mean.shape != (X.shape[0], 1) or variance.shape != (X.shape[0], 1):
        raise ValueError(
            "Relax-and-rank currently requires a scalar posterior with shape [n_candidates, 1]."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise FloatingPointError("Posterior mean/variance must be finite.")
    if torch.any(variance < 0):
        raise FloatingPointError("Posterior variance must be non-negative.")
    return mean, variance.sqrt()


def _ranking_score(
    mean: float,
    std: float,
    *,
    direction: RankingDirection,
    criterion: RankingCriterion,
    beta: float,
) -> float:
    sign = 1.0 if direction == "maximize" else -1.0
    utility = sign * mean
    if criterion == "ucb":
        utility += beta * std
    return utility


class MaterialRelaxationRanker:
    """Relax a candidate bank with any compatible MLIP and rank its posterior."""

    def __init__(self, *, relaxer: MaterialStructureRelaxer) -> None:
        self.relaxer = validate_structure_relaxer(relaxer)

    def run(
        self,
        structures: Sequence[Any],
        *,
        model_factory: ModelFactory,
        process_X: Tensor | None = None,
        direction: RankingDirection = "minimize",
        criterion: RankingCriterion = "posterior_mean",
        beta: float = 2.0,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 200,
        relax_cell: bool = False,
    ) -> RelaxedStructureRankingResult:
        resolved_structures = _validate_structures(structures)
        if not callable(model_factory):
            raise TypeError("model_factory must be callable.")
        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        if criterion not in {"posterior_mean", "ucb"}:
            raise ValueError("criterion must be 'posterior_mean' or 'ucb'.")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) < 0:
            raise ValueError("beta must be a non-negative number.")

        relaxations = tuple(
            self.relaxer.relax(
                structure,
                optimizer=optimizer,
                fmax=fmax,
                max_steps=max_steps,
                relax_cell=relax_cell,
            )
            for structure in resolved_structures
        )
        relaxed_structures = tuple(result.structure for result in relaxations)
        model = model_factory(relaxed_structures)

        process = _resolve_process_X(process_X, n=len(relaxed_structures))
        structure_ids = torch.arange(
            len(relaxed_structures), device=process.device, dtype=process.dtype
        ).unsqueeze(-1)
        X = torch.cat([structure_ids, process], dim=-1)
        mean, std = _posterior_statistics(model, X)

        rows: list[tuple[int, StructureRelaxationResult, float, float, float]] = []
        for index, relaxation in enumerate(relaxations):
            mean_value = float(mean[index, 0].item())
            std_value = float(std[index, 0].item())
            score = _ranking_score(
                mean_value,
                std_value,
                direction=direction,
                criterion=criterion,
                beta=float(beta),
            )
            rows.append((index, relaxation, mean_value, std_value, score))

        rows.sort(key=lambda row: row[4], reverse=True)
        ranked = tuple(
            RelaxedStructureRank(
                rank=rank,
                source_index=source_index,
                relaxation=relaxation,
                posterior_mean=mean_value,
                posterior_std=std_value,
                score=score,
            )
            for rank, (source_index, relaxation, mean_value, std_value, score) in enumerate(rows, start=1)
        )
        return RelaxedStructureRankingResult(
            candidates=ranked,
            direction=direction,
            criterion=criterion,
            beta=float(beta),
        )


__all__ = [
    "MaterialRelaxationRanker",
    "ModelFactory",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
]
