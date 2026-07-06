from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bochan.acquisition.feasible import FeasibilityConstraintSpec, FeasibilityWeightedAcquisition
from bochan.api.acquisition_config import AcquisitionConfig, OutcomeConstraintConfig
from bochan.api.classification_perturbation_defaults import (
    _build_acquisition,
    _resolve_outcome_constraint_config,
)
from bochan.api.configs import ModelBundle, ObjectiveConfig


class _BaseAcq:
    X_pending = None

    def __call__(self, X):
        return torch.ones(X.shape[:-2], dtype=X.dtype, device=X.device)

    def set_X_pending(self, X_pending=None):
        self.X_pending = X_pending


class _Model:
    output_names = ["strength", "quality_class"]
    num_outputs = 2

    def class_probs_list(self, X, output_indices=None):
        probs = torch.zeros(*X.shape[:-1], 3, dtype=X.dtype, device=X.device)
        probs[..., 2] = 0.9
        return [probs]

    def posterior(self, X, output_mode="objective", **kwargs):
        raise AssertionError("base acquisition stub should not call posterior")


@dataclass
class _Bundle:
    model: _Model = field(default_factory=_Model)
    train_Y: torch.Tensor = field(default_factory=lambda: torch.zeros(4, 2))
    model_config: object | None = None
    task_type: str = "hybrid"
    model_type: str = "base"
    metadata: dict = field(default_factory=dict)


def test_outcome_constraint_config_is_user_facing_spec_api() -> None:
    config = AcquisitionConfig(
        name="ei",
        objective_config=ObjectiveConfig(mode="scalar", output="strength"),
        outcome_constraint_config=OutcomeConstraintConfig(
            constraints=[
                FeasibilityConstraintSpec(
                    output="quality_class",
                    target_class=2,
                    threshold=0.7,
                    sense="ge",
                )
            ]
        ),
    )

    assert config.constraints is None
    assert config.acqf_kwargs.get("constraints") is None
    assert config.outcome_constraint_config.wrapper_constraints()[0].target_class == 2


def test_named_numeric_outcome_constraints_are_resolved_with_model_outputs() -> None:
    config = AcquisitionConfig(
        name="ei",
        outcome_constraint_config=OutcomeConstraintConfig(
            constraints=[
                FeasibilityConstraintSpec(
                    output="strength",
                    threshold=0.5,
                    sense="ge",
                )
            ]
        ),
    )

    resolved = _resolve_outcome_constraint_config(bundle=_Bundle(), config=config)

    assert resolved.constraints is not None
    assert resolved.acqf_kwargs["constraints"] is resolved.constraints
    samples = torch.tensor([[[0.6, 0.0], [0.4, 0.0]]], dtype=torch.double)
    values = resolved.constraints[0](samples)
    assert values.shape == torch.Size([1, 2])
    assert values[0, 0] <= 0.0
    assert values[0, 1] > 0.0


def test_target_class_outcome_constraints_wrap_base_acquisition() -> None:
    config = AcquisitionConfig(
        name="ei",
        acqf_factory=lambda bundle, config, data_context=None: _BaseAcq(),
        objective_config=ObjectiveConfig(mode="scalar", output="strength"),
        outcome_constraint_config=OutcomeConstraintConfig(
            constraints=[
                FeasibilityConstraintSpec(
                    output="quality_class",
                    target_class=2,
                    threshold=0.7,
                    sense="ge",
                )
            ],
            eta=0.05,
        ),
    )

    acqf = _build_acquisition(bundle=_Bundle(), config=config)

    assert isinstance(acqf, FeasibilityWeightedAcquisition)
    X = torch.zeros(5, 3, 2, dtype=torch.double)
    values = acqf(X)
    assert values.shape == torch.Size([5])
    assert torch.all(values > 0.5)
