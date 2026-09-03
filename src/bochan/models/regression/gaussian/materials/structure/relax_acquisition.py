"""Select relaxed structures with bochan acquisition functions, independent of MLIP backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from botorch.optim.optimize import optimize_acqf_discrete
from torch import Tensor

from bochan.api.acquisition.service import build_acquisition
from bochan.api.configs import AcquisitionConfig, DataContext, ModelBundle
from bochan.api.registry.acquisition import resolve_acqf_cls

from ..common.relaxation import (
    MaterialStructureRelaxer,
    StructureRelaxationResult,
    validate_structure_relaxer,
)

BundleFactory = Callable[[tuple[dict[str, Any], ...]], ModelBundle]

_DEFAULT_UCB_BETA = 3.0
_SUPPORTED_BO_ACQUISITION_NAMES = {
    "ei", "qei", "expectedimprovement", "qexpectedimprovement",
    "logei", "qlogei", "logexpectedimprovement", "qlogexpectedimprovement",
    "pi", "qpi", "probabilityofimprovement", "qprobabilityofimprovement",
    "ucb", "qucb", "upperconfidencebound", "qupperconfidencebound",
    "nei", "qnei", "noisyexpectedimprovement", "qnoisyexpectedimprovement",
    "lognei", "qlognei", "lognoisyexpectedimprovement", "qlognoisyexpectedimprovement",
}
_SUPPORTED_ACTIVE_LEARNING_NAMES = {
    "variance", "posteriorvariance", "qregressionposteriorvariance",
    "predictiveentropy", "entropy", "qregressionpredictiveentropy",
    "bald", "qregressionbald",
    "nipv", "qnipv", "negintegratedposteriorvariance", "qnegintegratedposteriorvariance",
    "negativeintegratedposteriorvariance", "qnegativeintegratedposteriorvariance",
}
_SUPPORTED_ACQUISITION_NAMES = _SUPPORTED_BO_ACQUISITION_NAMES | _SUPPORTED_ACTIVE_LEARNING_NAMES
_NIPV_NAMES = {
    "nipv", "qnipv", "negintegratedposteriorvariance", "qnegintegratedposteriorvariance",
    "negativeintegratedposteriorvariance", "qnegativeintegratedposteriorvariance",
}
_UCB_NAMES = {"ucb", "qucb", "upperconfidencebound", "qupperconfidencebound"}


@dataclass(frozen=True)
class RelaxedStructureAcquisitionCandidate:
    """One relaxed structure selected by an acquisition function."""

    selection_order: int
    source_index: int
    relaxation: StructureRelaxationResult
    individual_acquisition_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection_order": self.selection_order,
            "source_index": self.source_index,
            "individual_acquisition_value": self.individual_acquisition_value,
            "relaxation": self.relaxation.as_dict(),
        }


@dataclass(frozen=True)
class RelaxedStructureAcquisitionResult:
    """Serializable result of discrete acquisition selection after relaxation."""

    candidates: tuple[RelaxedStructureAcquisitionCandidate, ...]
    acquisition_name: str
    q: int
    acquisition_value: tuple[float, ...]

    @property
    def best(self) -> RelaxedStructureAcquisitionCandidate:
        if not self.candidates:
            raise ValueError("No acquisition-selected candidates are available.")
        return self.candidates[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "acquisition_name": self.acquisition_name,
            "q": self.q,
            "acquisition_value": list(self.acquisition_value),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _normalize_acquisition_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _is_active_learning_acquisition(config: AcquisitionConfig) -> bool:
    return _normalize_acquisition_name(config.name) in _SUPPORTED_ACTIVE_LEARNING_NAMES


def _is_nipv_acquisition(config: AcquisitionConfig) -> bool:
    return _normalize_acquisition_name(config.name) in _NIPV_NAMES


def _validate_acquisition_name(config: AcquisitionConfig) -> None:
    if _normalize_acquisition_name(config.name) not in _SUPPORTED_ACQUISITION_NAMES:
        raise ValueError(
            "Relaxed-structure selection currently supports EI/logEI/PI/UCB/NEI/logNEI "
            "plus active-learning variance/predictive-entropy/BALD/NIPV acquisitions. "
            "KG, MES, JES, lookahead, and multi-objective acquisitions require specialized "
            "discrete-selection semantics and are not enabled in this phase."
        )


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


def _candidate_X(process_X: Tensor, *, n: int) -> Tensor:
    structure_ids = torch.arange(n, device=process_X.device, dtype=process_X.dtype).unsqueeze(-1)
    return torch.cat([structure_ids, process_X], dim=-1)


def _is_multi_output(bundle: ModelBundle) -> bool:
    if bool(bundle.metadata.get("multi_output", False)):
        return True
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _resolve_acquisition_config(bundle: ModelBundle, config: AcquisitionConfig) -> AcquisitionConfig:
    resolved = config
    if config.acqf_cls is None and config.acqf_factory is None:
        acqf_cls = resolve_acqf_cls(
            config.name,
            task_type=str(bundle.task_type),
            model_type=str(bundle.model_type),
            multi_output=_is_multi_output(bundle),
        )
        resolved = replace(config, acqf_cls=acqf_cls)
    if _normalize_acquisition_name(resolved.name) in _UCB_NAMES and "beta" not in resolved.acqf_kwargs:
        kwargs = dict(resolved.acqf_kwargs)
        kwargs["beta"] = _DEFAULT_UCB_BETA
        resolved = replace(resolved, acqf_kwargs=kwargs)
    return resolved


def _default_data_context(bundle: ModelBundle, config: AcquisitionConfig, *, choices: Tensor) -> DataContext:
    train_X = bundle.train_X
    train_Y = bundle.train_Y
    if not torch.is_tensor(train_X):
        raise TypeError("bundle.train_X must be a Tensor for relaxed acquisition selection.")
    if not torch.is_tensor(train_Y):
        raise TypeError("bundle.train_Y must be a Tensor for relaxed acquisition selection.")
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("Automatic acquisition context currently requires scalar train_Y.")
    if _is_active_learning_acquisition(config):
        return DataContext(mc_points=choices if _is_nipv_acquisition(config) else None)
    objective_config = config.objective_config
    direction = "maximize" if objective_config is None else str(objective_config.direction)
    best_f = -train_Y.min() if direction == "minimize" else train_Y.max()
    return DataContext(X_baseline=train_X, best_f=best_f)


def _sanitize_active_learning_context(context: DataContext, *, choices: Tensor, config: AcquisitionConfig) -> DataContext:
    return DataContext(
        X_pending=context.X_pending,
        mc_points=choices if _is_nipv_acquisition(config) and context.mc_points is None else context.mc_points,
    )


def _acquisition_value_tuple(value: Tensor) -> tuple[float, ...]:
    if not torch.is_tensor(value):
        raise TypeError("Discrete acquisition optimizer must return a Tensor acquisition value.")
    if not torch.isfinite(value).all():
        raise FloatingPointError("Acquisition value must be finite.")
    return tuple(float(item) for item in value.detach().reshape(-1).cpu().tolist())


def _selected_indices(candidates: Tensor, *, n: int) -> tuple[int, ...]:
    if candidates.ndim != 2 or candidates.shape[-1] < 1:
        raise ValueError("Selected candidates must have shape [q, d].")
    raw = candidates[:, 0]
    rounded = raw.round()
    if not torch.equal(raw, rounded):
        raise ValueError("Selected structure indices must be integer-valued.")
    indices = tuple(int(value) for value in rounded.detach().cpu().tolist())
    if len(set(indices)) != len(indices):
        raise ValueError("Discrete acquisition selection returned duplicate structures.")
    if any(index < 0 or index >= n for index in indices):
        raise ValueError("Discrete acquisition selection returned an invalid structure index.")
    return indices


def _individual_acquisition_value(acqf: Any, X: Tensor) -> float:
    value = acqf(X.reshape(1, 1, -1))
    if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError("Acquisition must return one finite value for an individual candidate.")
    return float(value.detach().reshape(-1)[0].cpu().item())


class MaterialRelaxationAcquisitionSelector:
    """Relax structures with any compatible MLIP and select a discrete BO/AL batch."""

    def __init__(self, *, relaxer: MaterialStructureRelaxer) -> None:
        self.relaxer = validate_structure_relaxer(relaxer)

    def run(
        self,
        structures: Sequence[Any],
        *,
        bundle_factory: BundleFactory,
        acquisition_config: AcquisitionConfig,
        data_context: DataContext | None = None,
        process_X: Tensor | None = None,
        q: int = 1,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 200,
        relax_cell: bool = False,
    ) -> RelaxedStructureAcquisitionResult:
        resolved_structures = _validate_structures(structures)
        if not callable(bundle_factory):
            raise TypeError("bundle_factory must be callable.")
        if not isinstance(acquisition_config, AcquisitionConfig):
            raise TypeError("acquisition_config must be an AcquisitionConfig.")
        _validate_acquisition_name(acquisition_config)
        if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
            raise ValueError("q must be a positive integer.")
        if q > len(resolved_structures):
            raise ValueError("q cannot exceed the number of relaxed candidate structures.")

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
        bundle = bundle_factory(relaxed_structures)
        if not isinstance(bundle, ModelBundle):
            raise TypeError("bundle_factory must return a ModelBundle.")

        process = _resolve_process_X(process_X, n=len(relaxed_structures))
        choices = _candidate_X(process, n=len(relaxed_structures))
        config = _resolve_acquisition_config(bundle, acquisition_config)
        context = _default_data_context(bundle, config, choices=choices) if data_context is None else data_context
        if _is_active_learning_acquisition(config):
            context = _sanitize_active_learning_context(context, choices=choices, config=config)
        acqf = build_acquisition(bundle=bundle, config=config, data_context=context)

        selected, acq_value = optimize_acqf_discrete(
            acq_function=acqf,
            q=q,
            choices=choices,
            unique=True,
        )
        indices = _selected_indices(selected, n=len(relaxed_structures))
        selected_rows = tuple(
            RelaxedStructureAcquisitionCandidate(
                selection_order=order,
                source_index=index,
                relaxation=relaxations[index],
                individual_acquisition_value=_individual_acquisition_value(acqf, choices[index]),
            )
            for order, index in enumerate(indices, start=1)
        )
        return RelaxedStructureAcquisitionResult(
            candidates=selected_rows,
            acquisition_name=config.name,
            q=q,
            acquisition_value=_acquisition_value_tuple(acq_value),
        )


__all__ = [
    "BundleFactory",
    "MaterialRelaxationAcquisitionSelector",
    "RelaxedStructureAcquisitionCandidate",
    "RelaxedStructureAcquisitionResult",
]
