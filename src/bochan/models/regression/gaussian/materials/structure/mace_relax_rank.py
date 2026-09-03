"""Relax multiple structures with MACE and rank them with bochan / BoTorch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor

from ..common.relaxation import StructureRelaxationResult
from .mace_relaxation import MACEStructureRelaxer, OptimizerName

RankingDirection = Literal["minimize", "maximize"]
RankingCriterion = Literal["posterior_mean", "ucb", "acquisition"]
ModelFactory = Callable[[tuple[dict[str, Any], ...]], Any]

_SUPPORTED_DISCRETE_ACQUISITIONS = {
    "ei",
    "qei",
    "expectedimprovement",
    "qexpectedimprovement",
    "logei",
    "qlogei",
    "logexpectedimprovement",
    "qlogexpectedimprovement",
    "pi",
    "qpi",
    "probabilityofimprovement",
    "qprobabilityofimprovement",
    "ucb",
    "qucb",
    "upperconfidencebound",
    "qupperconfidencebound",
    "nei",
    "qnei",
    "noisyexpectedimprovement",
    "qnoisyexpectedimprovement",
    "lognei",
    "qlognei",
    "lognoisyexpectedimprovement",
    "qlognoisyexpectedimprovement",
}


@dataclass(frozen=True)
class RelaxedStructureRank:
    """One relaxed candidate and its ranking diagnostics."""

    rank: int
    source_index: int
    relaxation: StructureRelaxationResult
    posterior_mean: float
    posterior_std: float
    score: float
    acquisition_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "source_index": self.source_index,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "score": self.score,
            "acquisition_value": self.acquisition_value,
            "relaxation": self.relaxation.as_dict(),
        }


@dataclass(frozen=True)
class RelaxedStructureRankingResult:
    """Serializable ranking result for a batch of relaxed structures."""

    candidates: tuple[RelaxedStructureRank, ...]
    direction: RankingDirection
    criterion: RankingCriterion
    beta: float
    acquisition_name: str | None = None

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
            "acquisition_name": self.acquisition_name,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


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


def _model_from_factory_result(value: Any) -> Any:
    model = getattr(value, "model", value)
    if not callable(getattr(model, "posterior", None)):
        raise TypeError("model_factory must return a model or ModelBundle exposing posterior(X).")
    return model


def _posterior_statistics(model: Any, X: Tensor) -> tuple[Tensor, Tensor]:
    posterior = model.posterior(X)
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


def _default_acquisition_context(bundle: Any, config: Any, *, direction: RankingDirection) -> Any:
    from bochan.api.configs import DataContext

    train_X = getattr(bundle, "train_X", None)
    train_Y = getattr(bundle, "train_Y", None)
    context = DataContext(X_baseline=train_X, Y_baseline=train_Y)
    if torch.is_tensor(train_Y) and train_Y.numel() > 0:
        flattened = train_Y.reshape(-1)
        sign = -1.0 if direction == "minimize" else 1.0
        weight = 1.0
        objective_config = getattr(config, "objective_config", None)
        if objective_config is not None:
            weight = float(getattr(objective_config, "weight", 1.0))
        context.best_f = (sign * weight * flattened).max()
    return context


def _build_discrete_acquisition(
    bundle: Any,
    config: Any,
    *,
    direction: RankingDirection,
    data_context: Any | None,
) -> tuple[Any, Any]:
    from bochan.api.acquisition import build_acquisition
    from bochan.api.configs import ObjectiveConfig
    from bochan.api.registry.acquisition import resolve_acqf_cls

    required = ("model", "train_X", "train_Y", "task_type", "model_type")
    missing = [name for name in required if not hasattr(bundle, name)]
    if missing:
        raise TypeError(
            "Acquisition ranking requires model_factory to return a ModelBundle-like object; "
            f"missing attributes: {missing}."
        )

    name = _normalize_name(getattr(config, "name", ""))
    if name not in _SUPPORTED_DISCRETE_ACQUISITIONS:
        raise ValueError(
            "Relaxed-structure acquisition ranking currently supports EI/logEI/PI/UCB/NEI/logNEI "
            "and their q aliases."
        )

    resolved = config
    if getattr(resolved, "objective", None) is None and getattr(resolved, "objective_config", None) is None:
        resolved = replace(resolved, objective_config=ObjectiveConfig(direction=direction))
    if getattr(resolved, "acqf_cls", None) is None and getattr(resolved, "acqf_factory", None) is None:
        acqf_cls = resolve_acqf_cls(
            resolved.name,
            task_type=str(bundle.task_type),
            model_type=str(bundle.model_type),
            multi_output=False,
        )
        resolved = replace(resolved, acqf_cls=acqf_cls)

    context = data_context or _default_acquisition_context(bundle, resolved, direction=direction)
    acqf = build_acquisition(bundle, resolved, data_context=context)
    return acqf, resolved


def _evaluate_discrete_acquisition(acqf: Any, X: Tensor) -> Tensor:
    values = acqf(X.unsqueeze(-2))
    if not torch.is_tensor(values):
        raise TypeError("Acquisition function must return a Tensor.")
    values = values.reshape(-1)
    if values.numel() != X.shape[0]:
        raise ValueError(
            "Acquisition ranking requires one scalar value per q=1 candidate; "
            f"got {values.numel()} values for {X.shape[0]} candidates."
        )
    if not torch.isfinite(values).all():
        raise FloatingPointError("Acquisition values must be finite.")
    return values


class MACERelaxationRanker:
    """Relax structures with MACE, then rank the rebuilt structure bank with bochan."""

    def __init__(self, *, relaxer: MACEStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        self.relaxer = MACEStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        if not callable(getattr(self.relaxer, "relax", None)):
            raise TypeError("relaxer must expose relax(structure, ...).")

    def _prepare(
        self,
        structures: Sequence[Any],
        *,
        model_factory: ModelFactory,
        process_X: Tensor | None,
        optimizer: OptimizerName,
        fmax: float,
        max_steps: int,
        relax_cell: bool,
    ) -> tuple[tuple[StructureRelaxationResult, ...], Any, Any, Tensor, Tensor, Tensor]:
        resolved_structures = _validate_structures(structures)
        if not callable(model_factory):
            raise TypeError("model_factory must be callable.")
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
        factory_result = model_factory(relaxed_structures)
        model = _model_from_factory_result(factory_result)
        process = _resolve_process_X(process_X, n=len(relaxed_structures))
        structure_ids = torch.arange(
            len(relaxed_structures),
            device=process.device,
            dtype=process.dtype,
        ).unsqueeze(-1)
        X = torch.cat([structure_ids, process], dim=-1)
        mean, std = _posterior_statistics(model, X)
        return relaxations, factory_result, model, X, mean, std

    def run(
        self,
        structures: Sequence[Any],
        *,
        model_factory: ModelFactory,
        process_X: Tensor | None = None,
        direction: RankingDirection = "minimize",
        criterion: RankingCriterion = "posterior_mean",
        beta: float = 2.0,
        optimizer: OptimizerName = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 200,
        relax_cell: bool = False,
    ) -> RelaxedStructureRankingResult:
        """Relax every structure and rank the relaxed bank by posterior diagnostics."""

        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        if criterion not in {"posterior_mean", "ucb"}:
            raise ValueError("run() criterion must be 'posterior_mean' or 'ucb'.")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) < 0:
            raise ValueError("beta must be a non-negative number.")

        relaxations, _, _, _, mean, std = self._prepare(
            structures,
            model_factory=model_factory,
            process_X=process_X,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=max_steps,
            relax_cell=relax_cell,
        )
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

    def run_acquisition(
        self,
        structures: Sequence[Any],
        *,
        model_factory: ModelFactory,
        acquisition_config: Any,
        data_context: Any | None = None,
        process_X: Tensor | None = None,
        direction: RankingDirection = "minimize",
        optimizer: OptimizerName = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 200,
        relax_cell: bool = False,
    ) -> RelaxedStructureRankingResult:
        """Relax structures and rank q=1 candidates by a bochan acquisition function.

        The model factory must return a ``ModelBundle``-like object so bochan's
        standard acquisition builder can resolve the model, training baseline,
        objective direction, and acquisition kwargs. Acquisition values are
        always ranked in descending order; minimization is represented through
        the scalar objective rather than by negating acquisition values.
        """

        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        if acquisition_config is None:
            raise TypeError("acquisition_config is required.")

        relaxations, bundle, _, X, mean, std = self._prepare(
            structures,
            model_factory=model_factory,
            process_X=process_X,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=max_steps,
            relax_cell=relax_cell,
        )
        acqf, resolved_config = _build_discrete_acquisition(
            bundle,
            acquisition_config,
            direction=direction,
            data_context=data_context,
        )
        acquisition_values = _evaluate_discrete_acquisition(acqf, X)

        rows: list[tuple[int, StructureRelaxationResult, float, float, float]] = []
        for index, relaxation in enumerate(relaxations):
            rows.append(
                (
                    index,
                    relaxation,
                    float(mean[index, 0].item()),
                    float(std[index, 0].item()),
                    float(acquisition_values[index].item()),
                )
            )
        rows.sort(key=lambda row: row[4], reverse=True)
        ranked = tuple(
            RelaxedStructureRank(
                rank=rank,
                source_index=source_index,
                relaxation=relaxation,
                posterior_mean=mean_value,
                posterior_std=std_value,
                score=acquisition_value,
                acquisition_value=acquisition_value,
            )
            for rank, (source_index, relaxation, mean_value, std_value, acquisition_value) in enumerate(
                rows, start=1
            )
        )
        return RelaxedStructureRankingResult(
            candidates=ranked,
            direction=direction,
            criterion="acquisition",
            beta=0.0,
            acquisition_name=str(resolved_config.name),
        )


__all__ = [
    "MACERelaxationRanker",
    "ModelFactory",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
]
