from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.models.regression.gaussian.materials.common import (
    DirectMaterialPredictor,
    MaterialBaselineSpec,
    MaterialPropertyContract,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)


class _ZeroPredictor(DirectMaterialPredictor):
    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*X.shape[:-1], 1), dtype=X.dtype, device=X.device)


def test_property_contract_rejects_unit_mismatch() -> None:
    baseline = MaterialPropertyContract("energy", "eV", "total")
    target = MaterialPropertyContract("energy", "eV/atom", "per_atom")

    with pytest.raises(ValueError, match="Unit mismatch"):
        baseline.assert_compatible(target)


def test_property_contract_rejects_aggregation_mismatch() -> None:
    baseline = MaterialPropertyContract("energy", "eV", "total")
    target = MaterialPropertyContract("energy", "eV", "per_atom")

    with pytest.raises(ValueError, match="Aggregation mismatch"):
        baseline.assert_compatible(target)


def test_unspecified_aggregation_does_not_invent_conversion() -> None:
    baseline = MaterialPropertyContract("band_gap", "eV", "unspecified")
    target = MaterialPropertyContract("BAND_GAP", "eV", "intensive")

    baseline.assert_compatible(target)


def test_baseline_spec_has_stable_metadata() -> None:
    spec = MaterialBaselineSpec(
        family="MACE",
        property=MaterialPropertyContract("energy", "eV", "total"),
        output_name="energy",
        model_name="medium-mpa-0",
    )

    assert spec.family == "mace"
    assert spec.as_dict() == {
        "family": "mace",
        "property": {
            "quantity": "energy",
            "unit": "eV",
            "aggregation": "total",
        },
        "output_name": "energy",
        "output_index": None,
        "model_name": "medium-mpa-0",
        "enabled": True,
    }


def test_baseline_spec_rejects_ambiguous_output_selector() -> None:
    with pytest.raises(ValueError, match="at most one"):
        MaterialBaselineSpec(
            family="mace",
            property=MaterialPropertyContract("energy", "eV"),
            output_name="energy",
            output_index=0,
        )


def test_residual_targets_validate_physical_contract_before_subtraction() -> None:
    dtype = torch.double
    X = torch.linspace(0.0, 1.0, 4, dtype=dtype).unsqueeze(-1)
    Y = torch.zeros(4, 1, dtype=dtype)
    predictor = _ZeroPredictor()
    baseline_spec = MaterialBaselineSpec(
        family="mace",
        property=MaterialPropertyContract("energy", "eV", "total"),
    )

    with pytest.raises(ValueError, match="Unit mismatch"):
        compute_material_residual_targets(
            X,
            Y,
            predictor,
            baseline_spec=baseline_spec,
            target_contract=MaterialPropertyContract("energy", "eV/atom", "per_atom"),
        )


def test_residual_model_exposes_and_preserves_baseline_contract() -> None:
    dtype = torch.double
    X = torch.linspace(0.0, 1.0, 5, dtype=dtype).unsqueeze(-1)
    Y = torch.sin(X)
    predictor = _ZeroPredictor()
    residual_gp = SingleTaskGP(X, Y)
    baseline_spec = MaterialBaselineSpec(
        family="mace",
        property=MaterialPropertyContract("energy", "eV", "total"),
        output_index=0,
        model_name="medium-mpa-0",
    )
    model = ResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_gp,
        baseline_spec=baseline_spec,
    )

    assert model.baseline_spec is baseline_spec
    assert model.baseline_metadata == baseline_spec.as_dict()
    model.validate_target_contract(MaterialPropertyContract("energy", "eV", "total"))
