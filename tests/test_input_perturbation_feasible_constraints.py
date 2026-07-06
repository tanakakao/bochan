from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bochan.acquisition.feasible import (
    FeasibilityConstraintSpec,
    make_sample_constraints,
    wrap_sample_constraints_for_input_perturbation,
)
from bochan.api.acquisition_config import AcquisitionConfig
from bochan.api.classification_perturbation_defaults import _wrap_input_perturbation_constraints


@dataclass
class _InputTransformConfig:
    perturbation: bool = True
    n_w: int = 16


@dataclass
class _ModelConfig:
    input_transform_config: _InputTransformConfig = field(default_factory=_InputTransformConfig)


@dataclass
class _Bundle:
    model_config: _ModelConfig = field(default_factory=_ModelConfig)
    metadata: dict = field(default_factory=dict)


def test_make_sample_constraints_reduces_input_perturbation_dimension() -> None:
    constraints = make_sample_constraints(
        [FeasibilityConstraintSpec(output=1, threshold=0.2, sense="le")],
        input_perturbation_n_w=16,
    )
    samples = torch.zeros(4, 5, 48, 2, dtype=torch.double)
    samples[..., 1] = 0.1

    value = constraints[0](samples)

    assert value.shape == torch.Size([4, 5, 3])
    assert torch.all(value <= 0.0)


def test_wrap_sample_constraints_reduces_existing_constraint() -> None:
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


def test_high_level_config_wraps_constraints_from_bundle_n_w() -> None:
    raw_constraints = make_sample_constraints(
        [FeasibilityConstraintSpec(output=1, threshold=0.2, sense="le")]
    )
    config = AcquisitionConfig(name="ei", constraints=raw_constraints)

    resolved = _wrap_input_perturbation_constraints(
        bundle=_Bundle(),
        config=config,
    )

    assert resolved.constraints is not raw_constraints
    assert resolved.acqf_kwargs["constraints"] is resolved.constraints

    samples = torch.zeros(4, 5, 48, 2, dtype=torch.double)
    samples[..., 1] = 0.1
    value = resolved.constraints[0](samples)

    assert value.shape == torch.Size([4, 5, 3])
    assert torch.all(value <= 0.0)
