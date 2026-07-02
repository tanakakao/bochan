"""Version-compatible NSGA-II adapter for high-level optimizer dispatch.

The public BoTorch ``optimize_with_nsgaii`` signature expanded over time.
Bochan supports older BoTorch releases, so this module filters unsupported
keywords at runtime and adapts scalar multi-objective acquisitions such as EHVI
to the multi-output posterior-mean acquisition expected by NSGA-II.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from . import nsgaii as _base


LinearConstraint = _base.LinearConstraint
OutcomeConstraint = _base.OutcomeConstraint


def _callable_sequence(value: Any) -> list[Callable[[Tensor], Tensor]] | None:
    """Return a callable sequence, ignoring methods and framework registries."""

    if value is None:
        return None
    if callable(value) or isinstance(value, (str, bytes, dict)):
        return None
    if not isinstance(value, Sequence):
        return None
    values = list(value)
    if not all(callable(item) for item in values):
        return None
    return values


def _resolve_acquisition_constraints(acq_function: Any) -> list[OutcomeConstraint] | None:
    """Extract explicitly stored outcome constraints from an acquisition."""

    namespace = getattr(acq_function, "__dict__", {})
    for name in ("constraints", "outcome_constraints"):
        if name in namespace:
            constraints = _callable_sequence(namespace[name])
            if constraints is not None:
                return constraints

    for name in ("constraints", "outcome_constraints"):
        constraints = _callable_sequence(getattr(acq_function, name, None))
        if constraints is not None:
            return constraints
    return None


def _model_num_outputs(model: Any) -> int | None:
    """Resolve a model's output count without assuming a concrete model class."""

    for name in ("num_outputs", "_num_outputs", "num_objectives"):
        value = getattr(model, name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value is not None:
            return int(value)
    return None


def _resolve_nsgaii_target(acq_function: Any) -> Any:
    """Return a true multi-output acquisition suitable for NSGA-II.

    EHVI and NEHVI return a scalar hypervolume-improvement value. NSGA-II should
    instead optimize the underlying model's vector posterior mean while reusing
    the EHVI objective transform, constraints, and reference point.
    """

    try:
        from botorch.acquisition.multioutput_acquisition import (
            MultiOutputAcquisitionFunction,
            MultiOutputPosteriorMean,
        )
    except ImportError:  # pragma: no cover - old BoTorch import layout
        return acq_function

    if isinstance(acq_function, MultiOutputAcquisitionFunction):
        return acq_function

    model = getattr(acq_function, "model", None)
    if model is None:
        return acq_function
    num_outputs = _model_num_outputs(model)
    if num_outputs is None or num_outputs < 2:
        return acq_function
    return MultiOutputPosteriorMean(model=model)


def _apply_discrete_choices(
    X: Tensor,
    discrete_choices: dict[int, Sequence[float] | Tensor] | None,
) -> Tensor:
    """Apply nearest-choice rounding for legacy BoTorch releases."""

    if not discrete_choices:
        return X
    repaired = X.clone()
    for dim, choices in discrete_choices.items():
        choices_t = torch.as_tensor(
            choices,
            dtype=X.dtype,
            device=X.device,
        ).reshape(-1)
        if choices_t.numel() == 0:
            raise ValueError(f"discrete_choices[{dim}] must not be empty.")
        values = repaired[..., int(dim)].unsqueeze(-1)
        nearest = (values - choices_t).abs().argmin(dim=-1)
        repaired[..., int(dim)] = choices_t[nearest]
    return repaired


def _evaluate_objectives(
    *,
    acq_function: Any,
    X: Tensor,
    objective: Any | None,
) -> Tensor:
    """Re-evaluate NSGA-II objective values after legacy post-processing."""

    X_eval = X.unsqueeze(-2)
    with torch.no_grad():
        values = acq_function(X=X_eval)
        if objective is not None:
            try:
                values = objective(values, X=X_eval)
            except TypeError:
                values = objective(values)

    if values.ndim >= 3 and values.shape[-2] == 1:
        values = values.squeeze(-2)
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    return values


def _accepted_parameters(function: Callable[..., Any]) -> tuple[set[str], bool]:
    """Return named parameters and whether ``**kwargs`` is accepted."""

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return set(), True
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return set(parameters), accepts_var_kwargs


def _make_version_compatible_optimizer(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap BoTorch's NSGA-II utility with runtime signature adaptation."""

    original = getattr(function, "_bochan_original", function)

    def compatible_optimize_with_nsgaii(*args: Any, **kwargs: Any):
        accepted, accepts_var_kwargs = _accepted_parameters(original)

        def supports(name: str) -> bool:
            return accepts_var_kwargs or name in accepted

        inequality_constraints = kwargs.get("inequality_constraints")
        if inequality_constraints and not supports("inequality_constraints"):
            raise NotImplementedError(
                "The installed BoTorch optimize_with_nsgaii does not support "
                "inequality_constraints. Upgrade BoTorch to a release whose "
                "optimize_with_nsgaii signature includes this parameter."
            )

        legacy_discrete_choices = None
        if kwargs.get("discrete_choices") and not supports("discrete_choices"):
            legacy_discrete_choices = kwargs["discrete_choices"]

        legacy_post_processing = None
        if kwargs.get("post_processing_func") is not None and not supports(
            "post_processing_func"
        ):
            legacy_post_processing = kwargs["post_processing_func"]

        filtered_kwargs = {
            name: value
            for name, value in kwargs.items()
            if supports(name)
        }
        X, Y = original(*args, **filtered_kwargs)

        changed = False
        if legacy_discrete_choices:
            X = _apply_discrete_choices(X, legacy_discrete_choices)
            changed = True
        if legacy_post_processing is not None:
            X = legacy_post_processing(X)
            changed = True
        if changed:
            Y = _evaluate_objectives(
                acq_function=kwargs["acq_function"],
                X=X,
                objective=kwargs.get("objective"),
            )
        return X, Y

    compatible_optimize_with_nsgaii._bochan_original = original  # type: ignore[attr-defined]
    return compatible_optimize_with_nsgaii


# Patch the function referenced by nsgaii.py. Its public wrapper remains the
# single source of validation, objective-count inference, and equality merging.
_base.optimize_with_nsgaii = _make_version_compatible_optimizer(
    _base.optimize_with_nsgaii
)


def optimize_acqf_nsgaii(
    acq_function: Any,
    bounds: Tensor,
    *,
    q: int | None = 10,
    num_objectives: int | None = None,
    ref_point: Tensor | Sequence[float] | None = None,
    objective: Any | None = None,
    constraints: Sequence[OutcomeConstraint] | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    equality_tol: float = 1e-6,
    fixed_features: dict[int, float] | None = None,
    discrete_choices: dict[int, Sequence[float] | Tensor] | None = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    population_size: int = 250,
    max_gen: int | None = 200,
    seed: int | None = None,
    max_attempts: int = 2,
    validate_output: bool = True,
    validate_discrete: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Optimize a model's vector objective with version-compatible NSGA-II."""

    target = _resolve_nsgaii_target(acq_function)
    if target is not acq_function:
        if objective is None:
            candidate_objective = getattr(acq_function, "objective", None)
            if callable(candidate_objective):
                objective = candidate_objective
        if constraints is None:
            constraints = _resolve_acquisition_constraints(acq_function)
        if ref_point is None:
            candidate_ref_point = getattr(acq_function, "ref_point", None)
            if candidate_ref_point is not None and not callable(candidate_ref_point):
                ref_point = candidate_ref_point

    return _base.optimize_acqf_nsgaii(
        acq_function=target,
        bounds=bounds,
        q=q,
        num_objectives=num_objectives,
        ref_point=ref_point,
        objective=objective,
        constraints=constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        equality_tol=equality_tol,
        fixed_features=fixed_features,
        discrete_choices=discrete_choices,
        post_processing_func=post_processing_func,
        population_size=population_size,
        max_gen=max_gen,
        seed=seed,
        max_attempts=max_attempts,
        validate_output=validate_output,
        validate_discrete=validate_discrete,
        sequential=sequential,
        **kwargs,
    )


equality_constraints_to_inequality_constraints = (
    _base.equality_constraints_to_inequality_constraints
)
validate_discrete_choices = _base.validate_discrete_choices


__all__ = [
    "equality_constraints_to_inequality_constraints",
    "optimize_acqf_nsgaii",
    "validate_discrete_choices",
]
