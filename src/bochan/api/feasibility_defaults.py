"""Core outcome-constraint composition for acquisition construction.

This module keeps feasibility routing in the normal high-level API call graph.
It intentionally does not replace functions in :mod:`bochan.api.factory` or
:mod:`bochan.api.engine` at runtime.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class FeasibilityBuildPlan:
    """Deferred wrapper settings for one acquisition build."""

    constraints: tuple[Any, ...]
    config: Any


def _output_names(bundle: Any | None) -> list[str] | None:
    if bundle is None:
        return None
    names = getattr(bundle.model, "output_names", None)
    if callable(names):
        names = names()
    return None if names is None else list(names)


def _explicitly_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    """Return whether a callable or one of its class bases declares ``keyword``.

    A variadic ``**kwargs`` alone is intentionally not treated as support. Some
    bochan acquisitions forward ``**kwargs`` to a parent that does not accept
    BoTorch's ``constraints`` argument.
    """

    if callable_obj is None:
        return False

    callables: list[Any] = []
    if isinstance(callable_obj, type):
        for cls in callable_obj.__mro__:
            init = cls.__dict__.get("__init__")
            if init is not None:
                callables.append(init)
    else:
        callables.append(callable_obj)

    for candidate in callables:
        try:
            signature = inspect.signature(candidate)
        except (TypeError, ValueError):
            continue
        if keyword in signature.parameters:
            return True
    return False


def _constraint_specs(config: Any) -> list[Any]:
    """Convert a high-level outcome constraint config to wrapper specs."""

    if config.has_spec_constraints():
        return list(config.constraints or [])

    from bochan.acquisition.feasible import FeasibilityConstraintSpec

    specs = []
    for output, operator, threshold in zip(
        config.output_indices,
        config.operators,
        config.thresholds,
        strict=True,
    ):
        normalized_operator = str(operator).lower()
        sense = "ge" if normalized_operator in {"ge", "gt"} else "le"
        specs.append(
            FeasibilityConstraintSpec(
                output=int(output),
                threshold=float(threshold),
                sense=sense,
            )
        )
    return specs


def resolve_outcome_constraint_config(*, bundle: Any | None, config: Any) -> Any:
    """Resolve deferred named numeric constraints once model outputs are known."""

    constraint_config = getattr(config, "outcome_constraint_config", None)
    if constraint_config is None:
        return config
    if isinstance(constraint_config, dict):
        from .acquisition_config import OutcomeConstraintConfig

        constraint_config = OutcomeConstraintConfig(**constraint_config)
        config.outcome_constraint_config = constraint_config

    # Class-probability and ordinal-rank constraints need model access and are
    # applied by FeasibilityWeightedAcquisition rather than sample callables.
    if constraint_config.wrapper_constraints():
        kwargs = dict(config.acqf_kwargs)
        kwargs.pop("constraints", None)
        config.constraints = None
        config.acqf_kwargs = kwargs
        return config

    if config.constraints is None:
        built_constraints = constraint_config.build(output_names=_output_names(bundle))
        if built_constraints:
            kwargs = dict(config.acqf_kwargs)
            kwargs["constraints"] = built_constraints
            config.constraints = built_constraints
            config.acqf_kwargs = kwargs
    return config


def prepare_feasibility_build(*, bundle: Any, config: Any) -> tuple[Any, FeasibilityBuildPlan | None]:
    """Resolve native constraints or prepare a model-aware feasibility wrapper."""

    config = resolve_outcome_constraint_config(bundle=bundle, config=config)
    constraint_config = getattr(config, "outcome_constraint_config", None)
    if constraint_config is None:
        return config, None

    wrapper_constraints = _constraint_specs(constraint_config)
    has_model_dependent_constraints = constraint_config.has_model_dependent_constraints()
    needs_non_native_wrapper = bool(
        config.constraints
        and not _explicitly_accepts_keyword(config.acqf_cls, "constraints")
    )
    should_wrap = bool(
        wrapper_constraints
        and (has_model_dependent_constraints or needs_non_native_wrapper)
    )
    if not should_wrap:
        base_config = replace(
            config,
            acqf_factory=None,
            outcome_constraint_config=None,
        )
        return base_config, None

    base_kwargs = dict(config.acqf_kwargs)
    base_kwargs.pop("constraints", None)
    base_config = replace(
        config,
        acqf_factory=None,
        constraints=None,
        outcome_constraint_config=None,
        acqf_kwargs=base_kwargs,
    )
    return base_config, FeasibilityBuildPlan(
        constraints=tuple(wrapper_constraints),
        config=constraint_config,
    )


def apply_feasibility_build_plan(*, acqf: Any, model: Any, plan: FeasibilityBuildPlan | None) -> Any:
    """Apply a prepared feasibility wrapper after the base acquisition is built."""

    if plan is None:
        return acqf

    from bochan.acquisition.feasible import FeasibilityWeightedAcquisition

    config = plan.config
    return FeasibilityWeightedAcquisition(
        acqf=acqf,
        model=model,
        constraints=plan.constraints,
        eta=config.eta,
        posterior_mode=config.posterior_mode,
        reduce_constraints=config.reduce_constraints,
        reduce_q=config.reduce_q,
        min_feasibility=config.min_feasibility,
        detach_feasibility=config.detach_feasibility,
    )


def build_outcome_constrained_acquisition(
    *,
    bundle: Any,
    config: Any,
    data_context: Any | None = None,
) -> Any:
    """Build an acquisition with high-level outcome constraints natively.

    Numeric sample constraints are passed to acquisition classes that explicitly
    support BoTorch's ``constraints`` keyword. Model-dependent class/rank
    constraints, and numeric constraints for acquisitions without native support,
    are composed through :class:`FeasibilityWeightedAcquisition`.
    """

    from .factory import build_acquisition

    base_config, plan = prepare_feasibility_build(bundle=bundle, config=config)
    acqf = build_acquisition(
        bundle=bundle,
        config=base_config,
        data_context=data_context,
    )
    return apply_feasibility_build_plan(
        acqf=acqf,
        model=bundle.model,
        plan=plan,
    )


__all__ = [
    "FeasibilityBuildPlan",
    "_constraint_specs",
    "_explicitly_accepts_keyword",
    "apply_feasibility_build_plan",
    "build_outcome_constrained_acquisition",
    "prepare_feasibility_build",
    "resolve_outcome_constraint_config",
]
