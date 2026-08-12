"""Regression objective for one-to-many input perturbation transforms."""

from __future__ import annotations

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

from .regression import (
    MultiOutputRegressionInputPerturbationObjective as _BasePerturbationObjective,
)
from .regression import RiskType


class MultiOutputRegressionInputPerturbationObjective(_BasePerturbationObjective):
    """Aggregate expanded ``q * n_w`` values before final q-shape validation.

    A one-to-many input transform expands each public candidate into ``n_w``
    internal points. The inner multi-output objective therefore intentionally
    produces a ``q * n_w`` axis. BoTorch's generic objective validation would
    reject that intermediate shape before this wrapper can aggregate it back to
    the public q-batch. The inner objective is owned by this wrapper, so its
    intermediate shape validation is disabled at construction time while the
    outer objective keeps BoTorch's final output-shape validation enabled.
    """

    def __init__(
        self,
        inner_objective: MCMultiOutputObjective,
        n_w: int | None = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
    ) -> None:
        if hasattr(inner_objective, "_verify_output_shape"):
            inner_objective._verify_output_shape = False
        super().__init__(
            inner_objective=inner_objective,
            n_w=n_w,
            risk_type=risk_type,
            alpha=alpha,
            maximize=maximize,
            aggregate_mean_when_no_risk=aggregate_mean_when_no_risk,
            allow_unexpanded=allow_unexpanded,
        )


__all__ = ["MultiOutputRegressionInputPerturbationObjective"]
