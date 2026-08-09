from __future__ import annotations

import torch

from bochan.acquisition.feasible.wrapper import combine_acquisition_with_feasibility
from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api import factory as api_factory
from bochan.tabular.outcome_constraints import apply_tabular_outcome_constraints


def test_signed_feasibility_weighting_never_rewards_lower_feasibility() -> None:
    base = torch.tensor([2.0, -2.0], dtype=torch.double)
    high_pf = torch.tensor([1.0, 1.0], dtype=torch.double)
    low_pf = torch.tensor([0.1, 0.1], dtype=torch.double)

    high = combine_acquisition_with_feasibility(base, high_pf)
    low = combine_acquisition_with_feasibility(base, low_pf)

    torch.testing.assert_close(high, base)
    assert low[0] < high[0]
    assert low[1] < high[1]
    torch.testing.assert_close(low, torch.tensor([0.2, -3.8], dtype=torch.double))


def test_signed_feasibility_weighting_has_finite_gradients() -> None:
    base = torch.tensor([-0.2, 0.3], dtype=torch.double, requires_grad=True)
    feasibility = torch.tensor([1e-12, 0.4], dtype=torch.double, requires_grad=True)

    value = combine_acquisition_with_feasibility(base, feasibility).sum()
    value.backward()

    assert base.grad is not None
    assert feasibility.grad is not None
    assert torch.isfinite(base.grad).all()
    assert torch.isfinite(feasibility.grad).all()


def test_tabular_outcome_constraint_entrypoint_does_not_patch_factory() -> None:
    before = api_factory.build_acquisition

    apply_tabular_outcome_constraints()

    assert api_factory.build_acquisition is before


def test_repair_fallback_inequality_sense_is_resolved_in_optimize_config() -> None:
    config = OptimizeConfig(
        inequality_constraints=[("dummy",)],
        repair_config=CandidateRepairConfig(),
    )

    assert config.repair_config is not None
    assert config.repair_config.inequality_sense == "ge"


def test_repair_local_inequality_sense_is_preserved() -> None:
    local_constraints = [("local",)]
    config = OptimizeConfig(
        inequality_constraints=[("top",)],
        repair_config=CandidateRepairConfig(
            inequality_constraints=local_constraints,
            inequality_sense="le",
        ),
    )

    assert config.repair_config is not None
    assert config.repair_config.inequality_constraints is local_constraints
    assert config.repair_config.inequality_sense == "le"
