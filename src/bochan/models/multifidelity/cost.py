"""Cost models and cost-aware utilities for multi-fidelity BO."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

FidelityCostKind = Literal["affine", "fixed", "callable", "learned_gp"]
FidelityCostCallable = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FidelityCostConfig:
    """Configuration for known or learned evaluation cost.

    ``kind='affine'`` preserves the historical bochan contract: ``fixed_cost``
    is the fidelity-independent intercept and ``fidelity_weights`` defines the
    linear fidelity contribution. ``kind='fixed'`` treats ``fixed_cost`` as a
    constant cost for every evaluation. ``kind='callable'`` accepts a Python
    callable mapping candidate ``X`` to cost.

    ``kind='learned_gp'`` fits a ``SingleTaskGP`` to observed costs supplied by
    ``train_X`` and ``train_cost``. By default the GP models ``log(cost)`` so
    posterior predictions can be mapped back to strictly positive costs. A
    pre-built ``cost_model`` can be supplied instead of training data for the
    Python API. ``use_mean=False`` propagates learned-cost posterior uncertainty
    through ``InverseCostWeightedUtility`` for acquisition functions that pass a
    sampler to the utility.

    ``min_cost`` is applied through the cost objective so every public mode
    remains strictly positive for inverse-cost weighting.
    """

    kind: FidelityCostKind | str = "affine"
    fixed_cost: float = 1.0
    fidelity_weights: Mapping[int, float] | None = None
    min_cost: float = 1e-2
    cost_callable: FidelityCostCallable | None = None
    train_X: Any | None = None
    train_cost: Any | None = None
    cost_model: Any | None = None
    log_cost: bool = True
    use_mean: bool = True
    fit_model: bool = True

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        supported = {"affine", "fixed", "callable", "learned_gp"}
        if kind not in supported:
            raise ValueError(
                "FidelityCostConfig.kind must be 'affine', 'fixed', 'callable', "
                "or 'learned_gp'."
            )
        fixed_cost = float(self.fixed_cost)
        min_cost = float(self.min_cost)
        if not math.isfinite(fixed_cost) or fixed_cost < 0:
            raise ValueError("fixed_cost must be finite and non-negative.")
        if not math.isfinite(min_cost) or min_cost <= 0:
            raise ValueError("min_cost must be finite and positive.")

        weights = self.fidelity_weights
        if weights is not None:
            normalized = {int(index): float(value) for index, value in weights.items()}
            if not normalized:
                raise ValueError("fidelity_weights must not be empty when supplied.")
            if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
                raise ValueError("fidelity_weights must contain finite non-negative values.")
            weights = normalized

        if kind == "affine":
            if self.cost_callable is not None:
                raise ValueError("cost_callable is only valid for kind='callable'.")
            self._reject_learned_fields()
        elif kind == "fixed":
            if weights is not None:
                raise ValueError("fidelity_weights is only valid for kind='affine'.")
            if self.cost_callable is not None:
                raise ValueError("cost_callable is only valid for kind='callable'.")
            self._reject_learned_fields()
        elif kind == "callable":
            if weights is not None:
                raise ValueError("fidelity_weights is only valid for kind='affine'.")
            if self.cost_callable is None or not callable(self.cost_callable):
                raise ValueError("kind='callable' requires a callable cost_callable.")
            self._reject_learned_fields()
        else:
            if weights is not None:
                raise ValueError("fidelity_weights is only valid for kind='affine'.")
            if self.cost_callable is not None:
                raise ValueError("cost_callable is only valid for kind='callable'.")
            has_model = self.cost_model is not None
            has_X = self.train_X is not None
            has_cost = self.train_cost is not None
            if has_X != has_cost:
                raise ValueError("learned_gp requires both train_X and train_cost when either is supplied.")
            if has_model and has_X:
                raise ValueError("learned_gp accepts either cost_model or train_X/train_cost, not both.")
            if not has_model and not has_X:
                raise ValueError("learned_gp requires cost_model or train_X/train_cost.")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fixed_cost", fixed_cost)
        object.__setattr__(self, "fidelity_weights", weights)
        object.__setattr__(self, "min_cost", min_cost)
        object.__setattr__(self, "log_cost", bool(self.log_cost))
        object.__setattr__(self, "use_mean", bool(self.use_mean))
        object.__setattr__(self, "fit_model", bool(self.fit_model))

    def _reject_learned_fields(self) -> None:
        if self.train_X is not None or self.train_cost is not None or self.cost_model is not None:
            raise ValueError(
                "train_X, train_cost, and cost_model are only valid for kind='learned_gp'."
            )


def _resolve_index(index: int, *, d: int | None) -> int:
    index = int(index)
    if index >= 0:
        return index
    if d is None:
        raise ValueError(
            "Negative fidelity cost indices require the model dimension d."
        )
    resolved = d + index
    if resolved < 0 or resolved >= d:
        raise ValueError(
            f"Fidelity cost feature index {index} is out of range for d={d}."
        )
    return resolved


def _resolved_weights(
    config: FidelityCostConfig,
    *,
    fidelity_features: tuple[int, ...],
    d: int | None = None,
) -> dict[int, float]:
    features = tuple(int(index) for index in fidelity_features)
    if config.fidelity_weights is None:
        if len(features) != 1:
            raise ValueError(
                "Default fidelity cost weights require exactly one fidelity feature. "
                "Provide fidelity_weights explicitly for multiple fidelities."
            )
        return {features[0]: 1.0}

    weights: dict[int, float] = {}
    for raw_index, value in config.fidelity_weights.items():
        index = _resolve_index(int(raw_index), d=d)
        if index in weights:
            raise ValueError(
                "fidelity_weights contains duplicate indices after negative-index resolution: "
                f"feature {index}."
            )
        weights[index] = float(value)

    unknown = set(weights) - set(features)
    if unknown:
        raise ValueError(
            "fidelity_weights may reference only model fidelity features; "
            f"unknown indices: {sorted(unknown)}."
        )
    return weights


def _minimum_cost_objective(min_cost: float) -> GenericMCObjective:
    """Return a cost objective that guarantees strictly positive costs."""

    return GenericMCObjective(
        lambda samples, X=None: samples.squeeze(-1).clamp_min(min_cost)
    )


def _log_cost_objective(min_cost: float) -> GenericMCObjective:
    """Map a log-cost posterior back to strictly positive cost."""

    return GenericMCObjective(
        lambda samples, X=None: samples.squeeze(-1).exp().clamp_min(min_cost)
    )


def _normalize_callable_cost(
    value: Any,
    *,
    X: torch.Tensor,
) -> torch.Tensor:
    """Normalize a known Python cost callable to ``batch x q x 1``."""

    cost = torch.as_tensor(value, dtype=X.dtype, device=X.device)
    target_shape = X.shape[:-1]
    if cost.ndim == 0:
        cost = cost.expand(target_shape).unsqueeze(-1)
    elif cost.shape == target_shape:
        cost = cost.unsqueeze(-1)
    elif cost.shape != (*target_shape, 1):
        raise ValueError(
            "cost_callable must return a scalar, batch_shape x q, or "
            "batch_shape x q x 1 tensor."
        )
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("cost_callable must return only finite costs.")
    return cost


def _learned_training_data(
    config: FidelityCostConfig,
    *,
    d: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.as_tensor(config.train_X)
    if not train_X.is_floating_point():
        train_X = train_X.to(dtype=torch.get_default_dtype())
    train_cost = torch.as_tensor(
        config.train_cost,
        dtype=train_X.dtype,
        device=train_X.device,
    )
    if train_X.ndim != 2:
        raise ValueError("learned_gp train_X must have shape n x d.")
    if d is not None and int(train_X.shape[-1]) != int(d):
        raise ValueError(
            f"learned_gp train_X has d={train_X.shape[-1]}, expected d={d}."
        )
    if train_cost.ndim == 1:
        train_cost = train_cost.unsqueeze(-1)
    if train_cost.ndim != 2 or train_cost.shape[-1] != 1:
        raise ValueError("learned_gp train_cost must have shape n or n x 1.")
    if train_cost.shape[0] != train_X.shape[0]:
        raise ValueError("learned_gp train_X and train_cost must have the same number of rows.")
    if train_X.shape[0] < 2:
        raise ValueError("learned_gp requires at least two cost observations.")
    if not bool(torch.isfinite(train_X).all()) or not bool(torch.isfinite(train_cost).all()):
        raise ValueError("learned_gp training data must be finite.")
    if not bool((train_cost > 0).all()):
        raise ValueError("learned_gp train_cost must be strictly positive.")
    return train_X, train_cost


def build_learned_fidelity_cost_model(
    config: FidelityCostConfig,
    *,
    d: int | None = None,
) -> Any:
    """Build or return the GP used to model observed evaluation cost."""

    if config.kind != "learned_gp":
        raise ValueError("build_learned_fidelity_cost_model requires kind='learned_gp'.")
    if config.cost_model is not None:
        model = config.cost_model
        if not hasattr(model, "posterior"):
            raise TypeError("learned_gp cost_model must implement posterior(X).")
        return model

    train_X, train_cost = _learned_training_data(config, d=d)
    train_Y = train_cost.log() if config.log_cost else train_cost
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=1),
    )
    if config.fit_model:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    model.eval()
    return model


def evaluate_fidelity_cost_mean(
    cost_model: Any,
    utility: InverseCostWeightedUtility,
    X: torch.Tensor,
    *,
    min_cost: float,
) -> torch.Tensor:
    """Evaluate posterior-mean cost with the utility's public cost objective."""

    posterior = cost_model.posterior(X)
    mean = posterior.mean
    cost = utility.cost_objective(mean, X=X)
    cost = torch.as_tensor(cost, dtype=X.dtype, device=X.device)
    if cost.shape == X.shape[:-1]:
        cost = cost.unsqueeze(-1)
    elif cost.shape != (*X.shape[:-1], 1):
        raise ValueError("Cost model posterior mean must resolve to batch_shape x q x 1 cost.")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("Cost model posterior mean produced non-finite cost.")
    return cost.clamp_min(float(min_cost))


def build_fidelity_cost_utility(
    config: FidelityCostConfig,
    *,
    fidelity_features: tuple[int, ...],
    d: int | None = None,
) -> tuple[Any, InverseCostWeightedUtility]:
    """Build a known or learned cost model and inverse-cost MF utility."""

    if not isinstance(config, FidelityCostConfig):
        raise TypeError("cost_config must be a FidelityCostConfig instance.")

    if config.kind == "affine":
        weights = _resolved_weights(
            config,
            fidelity_features=fidelity_features,
            d=d,
        )
        cost_model: Any = AffineFidelityCostModel(
            fidelity_weights=weights,
            fixed_cost=config.fixed_cost,
        )
        cost_objective = _minimum_cost_objective(config.min_cost)
        use_mean = True
    elif config.kind == "fixed":
        fixed_cost = config.fixed_cost
        cost_model = GenericDeterministicModel(
            f=lambda X: X[..., :1] * 0.0 + fixed_cost,
            num_outputs=1,
        )
        cost_objective = _minimum_cost_objective(config.min_cost)
        use_mean = True
    elif config.kind == "callable":
        cost_callable = config.cost_callable
        if cost_callable is None:  # guarded by the dataclass, defensive for typing
            raise ValueError("kind='callable' requires cost_callable.")

        def known_cost(X: torch.Tensor) -> torch.Tensor:
            return _normalize_callable_cost(cost_callable(X), X=X)

        cost_model = GenericDeterministicModel(f=known_cost, num_outputs=1)
        cost_objective = _minimum_cost_objective(config.min_cost)
        use_mean = True
    else:
        cost_model = build_learned_fidelity_cost_model(config, d=d)
        cost_objective = (
            _log_cost_objective(config.min_cost)
            if config.log_cost
            else _minimum_cost_objective(config.min_cost)
        )
        use_mean = config.use_mean

    utility = InverseCostWeightedUtility(
        cost_model=cost_model,
        use_mean=use_mean,
        cost_objective=cost_objective,
    )
    return cost_model, utility


__all__ = [
    "FidelityCostCallable",
    "FidelityCostConfig",
    "FidelityCostKind",
    "build_fidelity_cost_utility",
    "build_learned_fidelity_cost_model",
    "evaluate_fidelity_cost_mean",
]
