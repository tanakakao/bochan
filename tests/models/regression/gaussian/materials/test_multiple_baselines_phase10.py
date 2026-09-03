from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.models.regression.gaussian.materials.common import (
    DirectMaterialPredictor,
    MaterialBaselinePlan,
    MaterialBaselineSpec,
    MaterialPropertyContract,
    MultipleBaselineModelListGP,
    ResidualMaterialGPModel,
)


class _ConstantPredictor(DirectMaterialPredictor):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.full((*X.shape[:-1], 1), self.value, dtype=X.dtype, device=X.device)


def _baseline(
    family: str,
    output_name: str,
    *,
    quantity: str,
    unit: str,
) -> MaterialBaselineSpec:
    return MaterialBaselineSpec(
        family=family,
        output_name=output_name,
        property=MaterialPropertyContract(quantity=quantity, unit=unit),
    )


def _residual_model(spec: MaterialBaselineSpec, value: float) -> ResidualMaterialGPModel:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    observed = torch.tensor([[value], [value + 0.1], [value - 0.1]], dtype=torch.double)
    predictor = _ConstantPredictor(value)
    residual_Y = observed - predictor(train_X)
    return ResidualMaterialGPModel(
        predictor=predictor,
        residual_model=SingleTaskGP(train_X, residual_Y),
        baseline_spec=spec,
    )


def test_material_baseline_plan_resolves_multiple_families() -> None:
    specs = [
        _baseline("mace", "energy", quantity="energy", unit="eV"),
        _baseline("chgnet", "band_gap", quantity="band_gap", unit="eV"),
    ]

    plan = MaterialBaselinePlan.resolve(
        output_names=["energy", "band_gap", "strength"],
        baseline_specs=specs,
    )

    assert plan.baseline_output_indices == (0, 1)
    assert plan.ordinary_output_indices == (2,)
    assert [item.spec.family for item in plan.assignments] == ["mace", "chgnet"]


def test_material_baseline_plan_rejects_duplicate_output_assignment() -> None:
    specs = [
        _baseline("mace", "energy", quantity="energy", unit="eV"),
        _baseline("chgnet", "energy", quantity="energy", unit="eV"),
    ]

    with pytest.raises(ValueError, match="Multiple enabled baselines"):
        MaterialBaselinePlan.resolve(output_names=["energy", "strength"], baseline_specs=specs)


def test_multiple_baseline_model_list_accepts_residual_and_ordinary_outputs() -> None:
    energy_spec = _baseline("mace", "energy", quantity="energy", unit="eV")
    gap_spec = _baseline("chgnet", "band_gap", quantity="band_gap", unit="eV")
    energy_model = _residual_model(energy_spec, -1.0)
    gap_model = _residual_model(gap_spec, 1.5)

    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    strength_model = SingleTaskGP(
        train_X,
        torch.tensor([[2.0], [2.2], [2.1]], dtype=torch.double),
    )

    model = MultipleBaselineModelListGP(
        energy_model,
        gap_model,
        strength_model,
        output_names=["energy", "band_gap", "strength"],
        baseline_specs=[energy_spec, gap_spec],
    )

    assert model.num_outputs == 3
    assert model.baseline_plan.baseline_output_indices == (0, 1)
    assert model.baseline_metadata["ordinary_output_indices"] == [2]

    posterior = model.posterior(torch.tensor([[0.25]], dtype=torch.double))
    assert posterior.mean.shape[-1] == 3


def test_multiple_baseline_model_list_rejects_missing_residual_wrapper() -> None:
    energy_spec = _baseline("mace", "energy", quantity="energy", unit="eV")
    train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    ordinary = SingleTaskGP(train_X, torch.tensor([[0.0], [1.0]], dtype=torch.double))

    with pytest.raises(TypeError, match="not ResidualMaterialGPModel"):
        MultipleBaselineModelListGP(
            ordinary,
            output_names=["energy"],
            baseline_specs=[energy_spec],
        )
