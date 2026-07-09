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


def _coerce_outcome_constraints(value: Any) -> list[Constraint]:
    """Normalize an explicitly supplied outcome-constraint sequence."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "Outcome constraints must be a sequence of callables or None. "
            f"Got {type(value)}."
        )

    constraints = list(value)
    invalid = [type(constraint) for constraint in constraints if not callable(constraint)]
    if invalid:
        raise TypeError(
            "Every outcome constraint must be callable. "
            f"Got invalid element types: {invalid}."
        )
    return constraints


def _try_coerce_stored_constraints(value: Any) -> list[Constraint] | None:
    """Return stored callable constraints, or ``None`` for unrelated internals."""

    if value is None:
        return []
    if callable(value) or isinstance(value, (str, bytes)):
        return None
    if not isinstance(value, Sequence):
        return None

    constraints = list(value)
    if not all(callable(constraint) for constraint in constraints):
        return None
    return constraints


def _resolve_outcome_constraints(acq_function: Any) -> list[Constraint]:
    """Read configured constraints without mistaking framework internals for data.

    Only public storage names are considered. In particular, ``_constraints``
    is deliberately ignored because PyTorch / GPyTorch modules may use it as an
    internal dictionary unrelated to outcome constraints.
    """

    names = ("constraints", "outcome_constraints")
    namespace = getattr(acq_function, "__dict__", {})
    for name in names:
        if name not in namespace:
            continue
        constraints = _try_coerce_stored_constraints(namespace[name])
        if constraints is not None:
            return constraints

    # Fallback for wrappers exposing constraints through a property. Bound
    # methods and non-sequence framework state are ignored.
    for name in names:
        constraints = _try_coerce_stored_constraints(
            getattr(acq_function, name, None)
        )
        if constraints is not None:
            return constraints
    return []


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


def _reduce_expanded_candidate_axis(
    values: Tensor,
    n_candidates: int,
    *,
    candidate_dim: int,
    reduction: str,
) -> Tensor:
    """Reduce an InputPerturbation-expanded candidate axis back to ``N``.

    BoTorch ``InputPerturbation`` evaluates a finite pool of ``N`` candidates as
    ``N * n_w`` perturbed points, while ``MaxPosteriorSampling`` still passes the
    original finite pool ``X`` to the objective. Thompson sampling therefore has
    to aggregate the expanded axis before BoTorch's q-batch validation sees it.
    """

    if n_candidates <= 0:
        raise RuntimeError(f"n_candidates must be positive. Got {n_candidates}.")

    candidate_dim = candidate_dim if candidate_dim >= 0 else values.ndim + candidate_dim
    q_expanded = int(values.shape[candidate_dim])
    if q_expanded == n_candidates:
        return values
    if q_expanded % n_candidates != 0:
        return values

    n_w = q_expanded // n_candidates
    if n_w <= 1:
        return values

    values_w = values.reshape(
        *values.shape[:candidate_dim],
        n_candidates,
        n_w,
        *values.shape[candidate_dim + 1 :],
    )
    risk_dim = candidate_dim + 1
    if reduction == "mean":
        return values_w.mean(dim=risk_dim)
    if reduction == "max":
        return values_w.amax(dim=risk_dim)
    raise ValueError(f"Unknown reduction: {reduction!r}.")


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

    # InputPerturbation expands the finite pool from N to N * n_w. Default to
    # mean aggregation, matching the default bochan regression objective when no
    # explicit risk objective is supplied.
    if values.shape[-1] % n_candidates == 0:
        return _reduce_expanded_candidate_axis(
            values,
            n_candidates,
            candidate_dim=-1,
            reduction="mean",
        )

    if values.ndim >= 2 and values.shape[-2] % n_candidates == 0:
        values = _reduce_expanded_candidate_axis(
            values,
            n_candidates,
            candidate_dim=-2,
            reduction="mean",
        )
        if values.shape[-1] == 1:
            return values.squeeze(-1)
        return values

    raise RuntimeError(
        "Could not identify the Thompson candidate dimension in objective values. "
        f"Expected N={n_candidates}, got shape={tuple(values.shape)}."
    )


def _random_scalarize(values: Tensor, *, n_candidates: int) -> Tensor:
    """Randomly scalarize ``... x N x m`` values after per-sample scaling."""

    # Scalar objectives may include one or more model batch dimensions, e.g.
    # ``sample x model_batch x N``. Candidate-axis position, not ndim, is the
    # reliable discriminator.
    if values.shape[-1] == n_candidates:
        return values

    if values.ndim < 3 or values.shape[-2] != n_candidates:
        raise RuntimeError(
            "Expected scalar values ending in N or multi-output values ending in N x m. "
            f"Expected N={n_candidates}, got shape={tuple(values.shape)}."
        )

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

    if value.ndim >= 2 and value.shape[-2] % n_candidates == 0:
        value = _reduce_expanded_candidate_axis(
            value,
            n_candidates,
            candidate_dim=-2,
            reduction="max",
        )
        if value.shape[-1] == 1:
            return value.squeeze(-1)
        return value.amax(dim=-1)

    if value.ndim >= 1 and value.shape[-1] % n_candidates == 0:
        return _reduce_expanded_candidate_axis(
            value,
            n_candidates,
            candidate_dim=-1,
            reduction="max",
        )

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


def _has_posterior(value: Any) -> bool:
    """Return whether ``value`` exposes a BoTorch-compatible posterior method."""

    return callable(getattr(value, "posterior", None))


def _configured_thompson_sampling_model(acq_function: Any) -> Any | None:
    """Return a public posterior model stored by the high-level API."""

    for name in ("_bochan_thompson_model", "_thompson_sampling_model"):
        model = getattr(acq_function, name, None)
        if model is not None and _has_posterior(model):
            return model
    return None


def _resolve_sampling_model(acq_function: Any) -> Any | None:
    """Resolve a model for MaxPosteriorSampling, if one is safely available."""

    configured = _configured_thompson_sampling_model(acq_function)
    if configured is not None:
        return configured

    if _has_posterior(acq_function):
        return acq_function

    model = getattr(acq_function, "model", None)
    if model is not None and _has_posterior(model):
        return model

    try:
        model = _base._resolve_model(acq_function)
    except ValueError:
        return None
    if _has_posterior(model):
        return model
    return None


def _normalize_acquisition_scores(scores: Tensor, n_candidates: int) -> Tensor:
    """Normalize finite-pool acquisition values to one score per candidate."""

    scores = torch.as_tensor(scores)
    if scores.ndim > 0 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    if scores.ndim > 0 and scores.shape[0] == n_candidates:
        while scores.ndim > 1:
            scores = scores.mean(dim=-1)
        return scores
    if scores.numel() == n_candidates:
        return scores.reshape(n_candidates)
    raise RuntimeError(
        "Could not normalize finite-pool acquisition scores. "
        f"Expected N={n_candidates}, got shape={tuple(scores.shape)}."
    )


def _select_with_acquisition_scores(
    *,
    acq_function: Any,
    X_candidates: Tensor,
    q: int,
) -> tuple[Tensor, Tensor]:
    """Fallback for acquisitions whose internal model is not a posterior model."""

    if not callable(acq_function):
        raise AttributeError(
            f"{type(acq_function).__name__} does not expose a posterior model and is not callable."
        )

    with torch.no_grad():
        try:
            raw_scores = acq_function(X_candidates.unsqueeze(-2))
        except Exception:
            raw_scores = acq_function(X_candidates)
        scores = _normalize_acquisition_scores(raw_scores, int(X_candidates.shape[0]))
        topk = torch.topk(scores, k=int(q), largest=True).indices
        return X_candidates.index_select(0, topk), scores.index_select(0, topk)


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
        self.outcome_constraints = _coerce_outcome_constraints(constraints)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if X is None:
            raise ValueError("X is required for finite-pool Thompson sampling.")

        n_candidates = int(X.shape[-2])
        values = _call_objective_forward(self.objective, samples, X)
        values = _normalize_multi_output_values(values, n_candidates)
        scores = _random_scalarize(values, n_candidates=n_candidates)
        if scores.shape[-1] != n_candidates:
            raise RuntimeError(
                "Thompson scalarization did not preserve the candidate dimension. "
                f"Expected N={n_candidates}, got shape={tuple(scores.shape)}."
            )
        return _apply_outcome_constraints(
            scores,
            samples,
            self.outcome_constraints,
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

    model = _resolve_sampling_model(acq_function)
    if model is None:
        return _select_with_acquisition_scores(
            acq_function=acq_function,
            X_candidates=X_candidates,
            q=q,
        )

    objective = ThompsonScalarizedObjective(
        objective=getattr(acq_function, "objective", None),
        constraints=_resolve_outcome_constraints(acq_function),
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
        values = _normalize_multi_output_values(
            posterior.mean,
            n_candidates=int(candidates.shape[-2]),
        )
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
