from __future__ import annotations

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api import factory as api_factory
from bochan.tabular.outcome_constraints import apply_tabular_outcome_constraints


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
