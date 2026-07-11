from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from bochan.acquisition.objective import (
    make_outcome_constraint,
    make_outcome_constraints,
)
from bochan.api import (
    AcquisitionConfig,
    ModelBundle,
    ModelConfig,
    OutcomeConstraintConfig,
    build_acquisition,
)


class _AcquisitionWithConstraints:
    def __init__(self, model, constraints=None, objective=None) -> None:
        self.model = model
        self.constraints = constraints
        self.objective = objective


def _make_bundle() -> ModelBundle:
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=None,
        train_Y=None,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        task_type="regression",
        model_type="base",
    )


def test_acquisition_config_forwards_single_constraint() -> None:
    constraints = [make_outcome_constraint(1, "ge", 0.5)]
    objective = object()
    config = AcquisitionConfig(
        name="qei",
        acqf_cls=_AcquisitionWithConstraints,
        objective=objective,
        constraints=constraints,
    )

    acqf = build_acquisition(bundle=_make_bundle(), config=config)

    assert acqf.constraints is constraints
    assert acqf.objective is objective


def test_acquisition_config_forwards_parallel_list_constraints() -> None:
    constraints = make_outcome_constraints(
        output_indices=[1, 2],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    )
    config = AcquisitionConfig(
        name="qei",
        acqf_cls=_AcquisitionWithConstraints,
        constraints=constraints,
    )

    acqf = build_acquisition(bundle=_make_bundle(), config=config)

    assert acqf.constraints is constraints
    assert len(acqf.constraints) == 2


def test_outcome_constraint_config_builds_and_forwards_constraints() -> None:
    constraint_config = OutcomeConstraintConfig(
        output_indices=[1, 2],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    )
    config = AcquisitionConfig(
        name="qei",
        acqf_cls=_AcquisitionWithConstraints,
        outcome_constraint_config=constraint_config,
    )

    acqf = build_acquisition(bundle=_make_bundle(), config=config)
    samples = torch.tensor([[[[0.0, 0.7, 1.0]]]], dtype=torch.double)

    assert config.outcome_constraint_config is constraint_config
    assert config.constraints is not None
    assert len(config.constraints) == 2
    assert acqf.constraints is config.constraints
    torch.testing.assert_close(
        acqf.constraints[0](samples),
        torch.tensor([[[-0.2]]], dtype=torch.double),
    )
    torch.testing.assert_close(
        acqf.constraints[1](samples),
        torch.tensor([[[-0.2]]], dtype=torch.double),
    )


def test_outcome_constraint_config_survives_dataclasses_replace() -> None:
    config = AcquisitionConfig(
        name="nparego",
        outcome_constraint_config=OutcomeConstraintConfig(
            output_indices=[1],
            operators=["ge"],
            thresholds=[0.5],
        ),
    )

    resolved = replace(config, acqf_cls=_AcquisitionWithConstraints)
    acqf = build_acquisition(bundle=_make_bundle(), config=resolved)
    samples = torch.tensor([[[[0.0, 0.7]]]], dtype=torch.double)

    assert resolved.outcome_constraint_config is config.outcome_constraint_config
    assert resolved.constraints is not config.constraints
    assert resolved.acqf_kwargs["constraints"] is resolved.constraints
    torch.testing.assert_close(
        acqf.constraints[0](samples),
        torch.tensor([[[-0.2]]], dtype=torch.double),
    )


def test_direct_constraints_survive_dataclasses_replace() -> None:
    constraints = [make_outcome_constraint(0, "le", 1.0)]
    config = AcquisitionConfig(name="qei", constraints=constraints)

    resolved = replace(config, acqf_cls=_AcquisitionWithConstraints)

    assert resolved.constraints is constraints
    assert resolved.acqf_kwargs["constraints"] is constraints


def test_outcome_constraint_config_accepts_dictionary_input() -> None:
    config = AcquisitionConfig(
        name="qei",
        acqf_cls=_AcquisitionWithConstraints,
        outcome_constraint_config={
            "output_indices": [0],
            "operators": ["ge"],
            "thresholds": [0.25],
        },
    )

    assert isinstance(config.outcome_constraint_config, OutcomeConstraintConfig)
    assert config.constraints is not None
    assert len(config.constraints) == 1


def test_outcome_constraint_config_validates_parallel_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        OutcomeConstraintConfig(
            output_indices=[0, 1],
            operators=["ge"],
            thresholds=[0.5, 1.0],
        )


def test_constraints_and_constraint_config_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either constraints"):
        AcquisitionConfig(
            name="qei",
            constraints=[make_outcome_constraint(0, "ge", 0.0)],
            outcome_constraint_config=OutcomeConstraintConfig(
                output_indices=[0],
                operators=["ge"],
                thresholds=[0.0],
            ),
        )


def test_constraints_are_parallel_to_objective_in_config() -> None:
    constraint = make_outcome_constraint(0, "le", 1.0)
    config = AcquisitionConfig(
        name="qei",
        constraints=[constraint],
    )

    assert config.constraints == [constraint]
    assert config.acqf_kwargs["constraints"] == [constraint]


def test_old_constraints_inside_acqf_kwargs_are_rejected() -> None:
    with pytest.raises(ValueError, match="AcquisitionConfig.constraints"):
        AcquisitionConfig(
            name="qei",
            acqf_kwargs={"constraints": [lambda samples: samples[..., 0]]},
        )
