from __future__ import annotations

import inspect

from bochan.acquisition.ordinal.bayesian_optimization.utility_acquisitions import (
    qOrdinalExpectedImprovement,
    qOrdinalExpectedUtility,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)


def test_ordinal_utility_bo_defaults_to_joint_q_semantics() -> None:
    for cls in (
        qOrdinalExpectedUtility,
        qOrdinalExpectedImprovement,
        qOrdinalProbabilityOfImprovement,
        qOrdinalUpperConfidenceBound,
    ):
        signature = inspect.signature(cls.__init__)
        assert signature.parameters["q_mode"].default == "joint"
        assert signature.parameters["reduction"].default == "mean"
