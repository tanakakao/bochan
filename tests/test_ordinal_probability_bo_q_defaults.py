from __future__ import annotations

import inspect

from bochan.acquisition.ordinal.bayesian_optimization.utility_acquisitions import (
    _OrdinalPointwiseUtilityBOBase,
)


def test_ordinal_utility_bo_base_defaults_to_joint_q_semantics() -> None:
    signature = inspect.signature(_OrdinalPointwiseUtilityBOBase.__init__)

    assert signature.parameters["q_mode"].default == "joint"
    assert signature.parameters["reduction"].default == "mean"
