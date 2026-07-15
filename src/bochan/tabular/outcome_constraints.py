"""Outcome-constraint support for tabular acquisition functions."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

from bochan.api import engine as _engine
from bochan.api import factory as _factory
from bochan.api.acquisition_config import OutcomeConstraintConfig

_APPLIED = False
_BASE_BUILD_ACQUISITION = _factory.build_acquisition


def _explicitly_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    """Return whether a callable or one of its class bases declares ``keyword``.

    A ``**kwargs`` parameter alone is intentionally not treated as support. Some
    bochan acquisition classes forward ``**kwargs`` to a base class that does not
    accept BoTorch's ``constraints`` argument, which is the source of the reported
    BALD failure.
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


def _constraint_specs(config: OutcomeConstraintConfig) -> list[Any]:
    """Convert a high-level constraint config to feasibility-wrapper specs."""

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


class _TabularFeasibilityWeightedAcquisition:
    """Factory namespace for a posterior-supported feasibility wrapper."""

    @staticmethod
    def build(*, acqf: Any, model: Any, constraints: list[Any], config: OutcomeConstraintConfig) -> Any:
        from bochan.acquisition.feasible import FeasibilityWeightedAcquisition

        class PosteriorSupportedFeasibilityWeightedAcquisition(FeasibilityWeightedAcquisition):
            def _posterior(self, X):
                try:
                    return super()._posterior(X)
                except TypeError as exc:
                    if "output_mode" not in str(exc):
                        raise
                    return self.model.posterior(X)

        return PosteriorSupportedFeasibilityWeightedAcquisition(
            acqf=acqf,
            model=model,
            constraints=constraints,
            eta=config.eta,
            posterior_mode=config.posterior_mode,
            reduce_constraints=config.reduce_constraints,
            reduce_q=config.reduce_q,
            min_feasibility=config.min_feasibility,
            detach_feasibility=config.detach_feasibility,
        )


def _build_acquisition(bundle: Any, config: Any, data_context: Any | None = None) -> Any:
    """Wrap tabular acquisitions when constraints require wrapper evaluation.

    Model-dependent class-probability and ordinal-rank constraints cannot be
    represented by BoTorch sample-constraint callables. ``AcquisitionConfig``
    therefore intentionally leaves ``config.constraints`` as ``None`` for these
    constraints. They must always be applied through
    ``FeasibilityWeightedAcquisition``, even when the underlying acquisition
    class natively accepts a ``constraints`` keyword.
    """

    constraint_config = getattr(config, "outcome_constraint_config", None)
    if isinstance(constraint_config, dict):
        constraint_config = OutcomeConstraintConfig(**constraint_config)

    wrapper_constraints = (
        _constraint_specs(constraint_config)
        if constraint_config is not None
        else []
    )
    has_model_dependent_constraints = bool(
        constraint_config is not None
        and constraint_config.has_model_dependent_constraints()
    )
    needs_non_native_wrapper = bool(
        config.constraints
        and not _explicitly_accepts_keyword(config.acqf_cls, "constraints")
    )
    should_wrap = bool(
        wrapper_constraints
        and config.acqf_factory is None
        and (has_model_dependent_constraints or needs_non_native_wrapper)
    )
    if not should_wrap:
        return _BASE_BUILD_ACQUISITION(
            bundle=bundle,
            config=config,
            data_context=data_context,
        )

    base_kwargs = dict(config.acqf_kwargs)
    base_kwargs.pop("constraints", None)
    base_config = replace(
        config,
        constraints=None,
        outcome_constraint_config=None,
        acqf_kwargs=base_kwargs,
    )
    base_acqf = _BASE_BUILD_ACQUISITION(
        bundle=bundle,
        config=base_config,
        data_context=data_context,
    )
    return _TabularFeasibilityWeightedAcquisition.build(
        acqf=base_acqf,
        model=bundle.model,
        constraints=wrapper_constraints,
        config=constraint_config,
    )


def apply_tabular_outcome_constraints() -> None:
    """Install tabular support after the core API defaults are registered."""

    global _APPLIED
    if _APPLIED:
        return

    _factory.build_acquisition = _build_acquisition
    _engine.build_acquisition = _build_acquisition
    _APPLIED = True


__all__ = ["apply_tabular_outcome_constraints"]
