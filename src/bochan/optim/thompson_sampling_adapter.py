"""Adapters for finite-pool Thompson sampling with multi-output objectives.

BoTorch's :class:`~botorch.generation.MaxPosteriorSampling` expects an
``MCAcquisitionObjective`` that returns one scalar score per candidate. Some
multi-objective acquisition functions expose an ``MCMultiOutputObjective`` or
an objective marked as multi-output even when its forward result is already
scalar. Passing that objective directly triggers q-batch validation errors.

This module wraps the acquisition objective for Thompson sampling only. It
preserves scalar objectives, scalarizes true multi-output values, and applies
outcome constraints using constrained posterior-sampling semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.generation import MaxPosteriorSampling

from . import thompson_sampling as _base


Constraint = Callable[[Tensor], Tensor]


def _call_objective_forward(
    objective: Any | None,
    samples: Tensor,
    X: Tensor | None,
) -> Tensor:
    """Evaluate an objective without its q-batch shape verification wrapper."""

    if objective is None:
        return samples

    forward = getattr(objective, "forward", None)
    if callable(forward):
        try:
            return forward(samples, X=X)
        except TypeError:
            return forward(samples)

    try:
        return objective(samples, X=X)
    except TypeError:
        return objective(samples)


def _normalize_multi_output_values(values: Tensor, n_candidates: int) -> Tensor:
    """Convert objective values to ``... x N x m`` or scalar ``... x N``."""

    if values.ndim < 2:
        raise RuntimeError(
            "Thompson objective must return at least sample and candidate dimensions. "
            f"Got shape={tuple(values.shape)}."
        )

    # Scalar objectives, including multi-output objective classes that return a
    # scalar score, conventionally end in the candidate dimension.
    if values.shape[-1] == n_candidates:
        return values

    # True multi-output objectives conventionally use ``... x N x m``.
    if values.shape[-2] == n_candidates:
        if values.shape[-1] == 1:
            return values.squeeze(-1)
        return values

    raise RuntimeError(
        "Could not identify the Thompson candidate dimension in objective values. "
        f"Expected N={n_candidates}, got shape={tuple(values.shape)}."
    )


def _random_scalarize(values: Tensor) -> Tensor:
    """Randomly scalarize ``... x N x m`` values after per-sample scaling."""

    if values.ndim < 3:
        return values

    lower = values.amin(dim=-2, keepdim=True)
    scale = (values.amax(dim=-2, keepdim=True) - lower).clamp_min(1e-12)
    normalized = (values - lower) / scale

    weight_shape = (*values.shape[:-2], 1, values.shape[-1])
    weights = torch.rand(weight_shape, dtype=values.dtype, device=values.device)
    weights = weights.clamp_min(1e-12)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (normalized * weights).sum(dim=-1)


def _normalize_constraint_value(value: Tensor, n_candidates: int) -> Tensor:
    """Normalize one constraint to ``... x N`` where feasible means ``<= 0``."""

    if value.ndim >= 2 and value.shape[-2] == n_candidates:
        if value.shape[-1] == 1:
            return value.squeeze(-1)
        # A single callable returning several constraint values requires all of
        # them to be feasible, so its maximum is the relevant violation score.
        return value.amax(dim=-1)

    if value.ndim >= 1 and value.shape[-1] == n_candidates:
        return value

    raise RuntimeError(
        "Could not identify the candidate dimension in outcome-constraint values. "
        f"Expected N={n_candidates}, got shape={tuple(value.shape)}."
    )


def _apply_outcome_constraints(
    scores: Tensor,
    samples: Tensor,
    constraints: Sequence[Constraint],
    *,
    n_candidates: int,
) -> Tensor:
    """Prefer feasible points and fall back to minimum total violation."""

    if not constraints:
        return scores

    violations = torch.stack(
        [
            _normalize_constraint_value(constraint(samples), n_candidates)
            for constraint in constraints
        ],
        dim=-1,
    )
    feasible = (violations <= 0).all(dim=-1)
    has_feasible = feasible.any(dim=-1, keepdim=True)

    feasible_scores = scores.masked_fill(~feasible, -torch.inf)
    fallback_scores = -violations.clamp_min(0).sum(dim=-1)
    return torch.where(has_feasible, feasible_scores, fallback_scores)


class ThompsonScalarizedObjective(MCAcquisitionObjective):
    """Return one Thompson score per finite-pool candidate.

    Scalar objective outputs are preserved. True multi-output values are
    independently scalarized for each posterior sample, which lets a q-sized
    Thompson batch explore different Pareto trade-offs.
    """

    _is_mo = False

    def __init__(
        self,
        objective: Any | None = None,
        constraints: Sequence[Constraint] | None = None,
    ) -> None:
        super().__init__()
        self.objective = objective
        self.constraints = list(constraints or [])

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if X is None:
            raise ValueError("X is required for finite-pool Thompson sampling.")

        n_candidates = int(X.shape[-2])
        values = _call_objective_forward(self.objective, samples, X)
        values = _normalize_multi_output_values(values, n_candidates)
        scores = _random_scalarize(values)
        if scores.shape[-1] != n_candidates:
            raise RuntimeError(
                "Thompson scalarization did not preserve the candidate dimension. "
                f"Expected N={n_candidates}, got shape={tuple(scores.shape)}."
            )
        return _apply_outcome_constraints(
            scores,
            samples,
            self.constraints,
            n_candidates=n_candidates,
        )


def _select_with_scalarized_max_posterior_sampling(
    *,
    acq_function: Any,
    X_candidates: Tensor,
    q: int,
    replacement: bool,
    observation_noise: bool | Tensor,
) -> tuple[Tensor, Tensor]:
    """Select candidates with an objective compatible with posterior sampling."""

    model = _base._resolve_model(acq_function)
    objective = ThompsonScalarizedObjective(
        objective=getattr(acq_function, "objective", None),
        constraints=getattr(acq_function, "constraints", None),
    )
    posterior_transform = getattr(acq_function, "posterior_transform", None)

    model.eval()
    likelihood = getattr(model, "likelihood", None)
    if likelihood is not None and hasattr(likelihood, "eval"):
        likelihood.eval()

    strategy = MaxPosteriorSampling(
        model=model,
        objective=objective,
        posterior_transform=posterior_transform,
        replacement=replacement,
    )

    with torch.no_grad():
        candidates = strategy(
            X_candidates,
            num_samples=q,
            observation_noise=observation_noise,
        )
        posterior = model.posterior(
            candidates,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
        values = posterior.mean
        if values.ndim >= 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
    return candidates, values


# Keep the public functions and signatures from thompson_sampling.py while
# replacing their internal finite-pool selection step.
_base._select_with_max_posterior_sampling = (
    _select_with_scalarized_max_posterior_sampling
)
optimize_thompson_sampling = _base.optimize_thompson_sampling
optimize_thompson_sampling_mixed = _base.optimize_thompson_sampling_mixed


__all__ = [
    "ThompsonScalarizedObjective",
    "optimize_thompson_sampling",
    "optimize_thompson_sampling_mixed",
]
