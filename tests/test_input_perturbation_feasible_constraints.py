from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bochan.acquisition.feasible import (
    FeasibilityConstraintSpec,
    make_sample_constraints,
    wrap_sample_constraints_for_input_perturbation,
)
from bochan.api.classification_perturbation_defaults import (
    _build_objective,
    _keep_constrained_perturbation_q_expanded,
)
from bochan.api.configs import ObjectiveConfig
from bochan.api.configs.acquisition import AcquisitionConfig


@dataclass
class _InputTransformConfig:
    perturbation: bool = True
    n_w: int = 16


@dataclass
class _ModelConfig:
    input_transform_config: _InputTransformConfig = field(default_factory=_InputTransformConfig)


class _Model:
    output_names = ["strength", "defect"]
    num_outputs = 2


@dataclass
class _Bundle:
    model: _Model = field(default_factory=_Model)
    model_config: _ModelConfig = field(default_factory=_ModelConfig)
    task_type: str = "hybrid"
    metadata: dict = field(default_factory=dict)


def test_make_sample_constraints_can_explicitly_reduce_input_perturbation_dimension() -> None:
    constraints = make_sample_constraints(
        [FeasibilityConstraintSpec(output=1, threshold=0.2, sense="le")],
        input_perturbation_n_w=16,
    )
    samples = torch.zeros(4, 5, 48, 2, dtype=torch.double)
    samples[..., 1] = 0.1

    value = constraints[0](samples)

    assert value.shape == torch.Size([4, 5, 3])
    assert torch.all(value <= 0.0)


def test_make_sample_constraints_default_preserves_botorch_constraint_shape() -> None:
    constraints = make_sample_constraints(
        [FeasibilityConstraintSpec(output=1, threshold=0.2, sense="le")]
    )
    samples = torch.zeros(4, 5, 48, 2, dtype=torch.double)
    samples[..., 1] = 0.1

    value = constraints[0](samples)

    assert value.shape == torch.Size([4, 5, 48])
    assert torch.all(value <= 0.0)


def test_wrap_sample_constraints_can_reduce_existing_constraint() -> None:
    def raw_constraint(samples: torch.Tensor) -> torch.Tensor:
        return samples[..., 1] - 0.2

    wrapped = wrap_sample_constraints_for_input_perturbation(
        [raw_constraint],
        n_w=16,
    )[0]
    samples = torch.zeros(4, 5, 48, 2, dtype=torch.double)
    samples[..., 1] = 0.1

    value = wrapped(samples)

    assert value.shape == torch.Size([4, 5, 3])
    assert torch.all(value <= 0.0)


def test_high_level_constrained_config_keeps_objective_q_expanded() -> None:
    raw_constraints = make_sample_constraints(
        [FeasibilityConstraintSpec(output=1, threshold=0.2, sense="le")]
    )
    config = AcquisitionConfig(
        name="ei",
        objective_config=ObjectiveConfig(
            mode="scalar",
            output=0,
            n_w=16,
            aggregate_mean_when_no_risk=True,
        ),
        constraints=raw_constraints,
    )

    resolved = _keep_constrained_perturbation_q_expanded(
        bundle=_Bundle(),
        config=config,
    )

    assert resolved.constraints is raw_constraints
    assert resolved.acqf_kwargs["constraints"] is raw_constraints
    assert resolved.objective_config.n_w == 16
    assert resolved.objective_config.aggregate_mean_when_no_risk is False


def test_expanded_objective_disables_botorch_q_shape_check() -> None:
    config = AcquisitionConfig(
        name="ei",
        objective_config=ObjectiveConfig(
            mode="scalar",
            output="strength",
            n_w=16,
            aggregate_mean_when_no_risk=False,
        ),
        constraints=make_sample_constraints(
            [FeasibilityConstraintSpec(output="defect", threshold=0.2, sense="le")],
            output_names=["strength", "defect"],
        ),
    )

    objective = _build_objective(bundle=_Bundle(), config=config)

    assert objective._verify_output_shape is False

    samples = torch.zeros(4, 5, 48, 2, dtype=torch.double)
    X = torch.zeros(5, 3, 2, dtype=torch.double)
    value = objective(samples=samples, X=X)

    assert value.shape == torch.Size([4, 5, 48])
