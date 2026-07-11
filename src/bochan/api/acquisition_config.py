"""Public acquisition configuration with common defaults."""

# ruff: noqa: I001

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .configs import AcquisitionConfig as _BaseAcquisitionConfig


_DEFAULT_UCB_BETA = 3.0
_UCB_NAMES = {"ucb", "qucb", "upperconfidencebound", "qupperconfidencebound"}
ConstraintOperator = Literal["ge", "gt", "le", "lt"]
_MISSING = object()


def _normalize_acquisition_name(name: str) -> str:
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


def _coerce_constraint_spec(value: Any) -> Any:
    """Coerce a serializable outcome-constraint mapping to a spec object."""

    if not isinstance(value, dict):
        return value

    from bochan.acquisition.feasible import (
        FeasibilityConstraintSpec,
        OrdinalRankConstraintSpec,
    )

    kind = str(
        value.get("kind")
        or value.get("type")
        or value.get("constraint_type")
        or "feasibility"
    ).lower()
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"kind", "type", "constraint_type"}
    }
    if kind in {"ordinal", "ordinal_rank", "ordinalrank", "rank"}:
        return OrdinalRankConstraintSpec(**payload)
    return FeasibilityConstraintSpec(**payload)


@dataclass
class OutcomeConstraintConfig:
    """Serializable, user-facing configuration for outcome constraints.

    Prefer this high-level config in notebooks and apps. It keeps the constraint
    intent visible as data and lets bochan convert it to either BoTorch sample
    constraints or a feasibility wrapper when model-dependent class probabilities
    are required.

    ``constraints`` accepts ``FeasibilityConstraintSpec`` /
    ``OrdinalRankConstraintSpec`` objects or equivalent dictionaries. The old
    ``output_indices`` / ``operators`` / ``thresholds`` fields are still supported
    for numeric threshold constraints.
    """

    constraints: Sequence[Any] | None = None
    output_indices: Sequence[int] = field(default_factory=list)
    operators: Sequence[ConstraintOperator] = field(default_factory=list)
    thresholds: Sequence[float] = field(default_factory=list)

    # Settings used only when constraints need model access and are therefore
    # applied through FeasibilityWeightedAcquisition.
    eta: float = 1e-3
    reduce_constraints: str = "prod"
    reduce_q: str = "mean"
    posterior_mode: str = "objective"
    min_feasibility: float = 0.0
    detach_feasibility: bool = False

    def __post_init__(self) -> None:
        self.output_indices = list(self.output_indices)
        self.operators = list(self.operators)
        self.thresholds = list(self.thresholds)
        self.constraints = (
            None
            if self.constraints is None
            else [_coerce_constraint_spec(item) for item in self.constraints]
        )

        lengths = {
            "output_indices": len(self.output_indices),
            "operators": len(self.operators),
            "thresholds": len(self.thresholds),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "output_indices, operators, and thresholds must have the same "
                f"length. Got: {lengths}"
            )
        if any(int(index) < 0 for index in self.output_indices):
            raise ValueError("output_indices must contain non-negative integers.")
        invalid = [
            operator
            for operator in self.operators
            if str(operator).lower() not in {"ge", "gt", "le", "lt"}
        ]
        if invalid:
            raise ValueError(
                "operators must contain only 'ge', 'gt', 'le', or 'lt'. "
                f"Got invalid values: {invalid}"
            )
        if float(self.eta) <= 0.0:
            raise ValueError("eta must be positive.")

    def has_spec_constraints(self) -> bool:
        return bool(self.constraints)

    def has_named_outputs(self) -> bool:
        """Return whether spec constraints need model output names to build."""

        for spec in self.constraints or []:
            if isinstance(getattr(spec, "output", None), str):
                return True
        return False

    def has_model_dependent_constraints(self) -> bool:
        """Return whether constraints require model access, not just samples."""

        from bochan.acquisition.feasible import (
            FeasibilityConstraintSpec,
            OrdinalRankConstraintSpec,
        )

        for spec in self.constraints or []:
            if isinstance(spec, OrdinalRankConstraintSpec):
                return True
            if isinstance(spec, FeasibilityConstraintSpec) and spec.has_target_classes:
                return True
        return False

    def build(self, *, output_names: Sequence[str] | None = None) -> list[Any]:
        """Build BoTorch-supported sample constraint callables.

        Model-dependent class / rank probability constraints are intentionally
        not converted here; they should be applied through
        ``FeasibilityWeightedAcquisition`` so the model can provide
        ``class_probs_list``.
        """

        if self.has_spec_constraints():
            if self.has_model_dependent_constraints():
                return []
            if output_names is None and self.has_named_outputs():
                return []
            from bochan.acquisition.feasible import make_sample_constraints

            return make_sample_constraints(self.constraints or [], output_names=output_names)

        from bochan.acquisition.objective import make_outcome_constraints

        return make_outcome_constraints(
            output_indices=self.output_indices,
            operators=self.operators,
            thresholds=self.thresholds,
        )

    def wrapper_constraints(self) -> list[Any]:
        """Return constraints that should be evaluated with model access."""

        if not self.has_model_dependent_constraints():
            return []
        return list(self.constraints or [])


@dataclass
class AcquisitionConfig(_BaseAcquisitionConfig):
    """High-level acquisition configuration.

    Args:
        outcome_constraint_config: User-facing constraint config. Prefer this
            for normal use, especially when specifying classification classes.
        constraints: Explicit BoTorch sample-constraint callables for advanced
            Python use.

    Notes:
        ``constraints`` and ``outcome_constraint_config`` are mutually exclusive
        user inputs. ``constraints`` is the low-level BoTorch-facing escape hatch;
        ``outcome_constraint_config`` is the high-level, serializable API.
    """

    constraints: list[Any] | None = None
    outcome_constraint_config: OutcomeConstraintConfig | None = None

    def __post_init__(self) -> None:
        kwargs = dict(self.acqf_kwargs)

        kwargs_constraints = kwargs.pop("constraints", _MISSING)
        replaying_internal_constraints = (
            kwargs_constraints is not _MISSING
            and self.constraints is not None
            and kwargs_constraints is self.constraints
        )
        if kwargs_constraints is not _MISSING and not replaying_internal_constraints:
            raise ValueError(
                "Pass outcome constraints through AcquisitionConfig.constraints "
                "or AcquisitionConfig.outcome_constraint_config, not "
                "acqf_kwargs['constraints']."
            )

        if self.constraints is not None and self.outcome_constraint_config is not None:
            if replaying_internal_constraints:
                self.constraints = None
            else:
                raise ValueError(
                    "Specify either constraints or outcome_constraint_config, not both. "
                    "Use outcome_constraint_config for user-facing specs and "
                    "constraints only for explicit BoTorch callables."
                )

        if self.outcome_constraint_config is not None:
            constraint_config = self.outcome_constraint_config
            if isinstance(constraint_config, dict):
                constraint_config = OutcomeConstraintConfig(**constraint_config)
                self.outcome_constraint_config = constraint_config
            built_constraints = constraint_config.build()
            self.constraints = built_constraints if len(built_constraints) > 0 else None

        if self.constraints is not None:
            kwargs["constraints"] = self.constraints

        if (
            _normalize_acquisition_name(self.name) in _UCB_NAMES
            and "beta" not in kwargs
        ):
            kwargs["beta"] = _DEFAULT_UCB_BETA

        self.acqf_kwargs = kwargs


def _install_nan_multiobjective() -> None:
    from .nan_multiobjective import apply_nan_multiobjective

    apply_nan_multiobjective()


_install_nan_multiobjective()


__all__ = ["AcquisitionConfig", "ConstraintOperator", "OutcomeConstraintConfig"]
