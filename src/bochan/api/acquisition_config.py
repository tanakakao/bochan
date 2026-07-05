"""Public acquisition configuration with common defaults."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .configs import AcquisitionConfig as _BaseAcquisitionConfig
from .nan_multiobjective_compat import apply_nan_multiobjective_compat


_DEFAULT_UCB_BETA = 3.0
_UCB_NAMES = {"ucb", "qucb", "upperconfidencebound", "qupperconfidencebound"}
ConstraintOperator = Literal["ge", "gt", "le", "lt"]
_MISSING = object()


def _normalize_acquisition_name(name: str) -> str:
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


@dataclass
class OutcomeConstraintConfig:
    """Serializable configuration for threshold-based outcome constraints.

    The values at the same position define one constraint. For example,
    ``output_indices=[1, 2]``, ``operators=["ge", "le"]``, and
    ``thresholds=[0.5, 1.2]`` represent ``y[1] >= 0.5`` and ``y[2] <= 1.2``.

    BoTorch considers a generated constraint feasible when its callable returns
    a value less than or equal to zero.
    """

    output_indices: Sequence[int] = field(default_factory=list)
    operators: Sequence[ConstraintOperator] = field(default_factory=list)
    thresholds: Sequence[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_indices = list(self.output_indices)
        self.operators = list(self.operators)
        self.thresholds = list(self.thresholds)

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

    def build(self) -> list[Any]:
        """Build BoTorch-compatible constraint callables."""

        from bochan.acquisition.objective import make_outcome_constraints

        return make_outcome_constraints(
            output_indices=self.output_indices,
            operators=self.operators,
            thresholds=self.thresholds,
        )


@dataclass
class AcquisitionConfig(_BaseAcquisitionConfig):
    """High-level acquisition configuration.

    Args:
        constraints: Explicit BoTorch outcome-constraint callables for advanced
            Python use.
        outcome_constraint_config: Serializable threshold constraints. These are
            converted internally with ``make_outcome_constraints``.

    Notes:
        ``constraints`` and ``outcome_constraint_config`` are mutually exclusive.
        Both are first-class acquisition settings parallel to ``objective``.

        ``__post_init__`` is intentionally idempotent because the optimizer uses
        ``dataclasses.replace`` while resolving acquisition classes and defaults.

        UCB aliases use ``beta=3.0`` when ``acqf_kwargs`` does not explicitly
        provide a beta value. Explicit user configuration always takes priority.
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
                # ``dataclasses.replace`` replays the fields after the first
                # ``__post_init__`` generated ``constraints`` from the config.
                # Clear that derived value and rebuild it below.
                self.constraints = None
            else:
                raise ValueError(
                    "Specify either constraints or outcome_constraint_config, not both."
                )

        if self.outcome_constraint_config is not None:
            constraint_config = self.outcome_constraint_config
            if isinstance(constraint_config, dict):
                constraint_config = OutcomeConstraintConfig(**constraint_config)
                self.outcome_constraint_config = constraint_config
            self.constraints = constraint_config.build()

        if self.constraints is not None:
            kwargs["constraints"] = self.constraints

        if (
            _normalize_acquisition_name(self.name) in _UCB_NAMES
            and "beta" not in kwargs
        ):
            kwargs["beta"] = _DEFAULT_UCB_BETA

        self.acqf_kwargs = kwargs


apply_nan_multiobjective_compat()


__all__ = ["AcquisitionConfig", "ConstraintOperator", "OutcomeConstraintConfig"]
