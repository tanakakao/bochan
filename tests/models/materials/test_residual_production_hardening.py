from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from torch import nn

from bochan.models.regression.gaussian.materials.common import (
    MaterialBaselineSpec,
    MaterialPropertyContract,
    MultipleBaselineModelListGP,
    ResidualMaterialGPModel,
    assert_residual_posterior_equivalent,
    shared_parameter_aliases,
    validate_residual_production_model,
)
from bochan.models.regression.gaussian.materials.common.residual import DirectMaterialPredictor


class _ConstantPredictor(DirectMaterialPredictor):
    def __init__(self, value: float, shared: nn.Module | None = None) -> None:
        super().__init__()
        self.value = float(value)
        self.shared = shared

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.full((*X.shape[:-1], 1), self.value, dtype=X.dtype, device=X.device)


def _residual(value: float, *, spec: MaterialBaselineSpec | None = None) -> ResidualMaterialGPModel:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    predictor = _ConstantPredictor(value)
    train_Y = torch.tensor([[value], [value + 0.2], [value - 0.1]], dtype=torch.double)
    return ResidualMaterialGPModel(
        predictor=predictor,
        residual_model=SingleTaskGP(train_X, train_Y - predictor(train_X)),
        baseline_spec=spec,
    )


def test_validate_scalar_residual_model() -> None:
    model = _residual(1.0)
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    report = validate_residual_production_model(model, X, expected_num_outputs=1)

    assert report.num_outputs == 1
    assert report.baseline_output_indices == (0,)
    assert report.posterior_shape == (2, 1)


def test_validate_multiple_baseline_model_list() -> None:
    energy_spec = MaterialBaselineSpec(
        family="mace",
        output_name="energy",
        property=MaterialPropertyContract("energy", "eV", "total"),
    )
    gap_spec = MaterialBaselineSpec(
        family="m3gnet",
        output_name="gap",
        property=MaterialPropertyContract("band_gap", "eV", "intensive"),
    )
    energy = _residual(-1.0, spec=energy_spec)
    gap = _residual(1.2, spec=gap_spec)
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    ordinary = SingleTaskGP(X, torch.tensor([[2.0], [2.2], [2.1]], dtype=torch.double))
    model = MultipleBaselineModelListGP(
        energy,
        gap,
        ordinary,
        output_names=["energy", "gap", "strength"],
        baseline_specs=[energy_spec, gap_spec],
    )

    report = validate_residual_production_model(
        model,
        torch.tensor([[0.25], [0.75]], dtype=torch.double),
        expected_num_outputs=3,
    )

    assert report.baseline_output_indices == (0, 1)
    assert report.posterior_shape == (2, 3)


def test_shared_parameter_aliases_reports_duplicate_ownership() -> None:
    shared = nn.Linear(1, 1)
    module = nn.Module()
    module.left = shared
    module.right = shared

    aliases = shared_parameter_aliases(module)

    assert ("left.weight", "right.weight") in aliases
    assert ("left.bias", "right.bias") in aliases


def test_posterior_equivalence_helper_accepts_roundtrip_copy(tmp_path) -> None:
    model = _residual(0.5)
    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    path = tmp_path / "residual.pt"
    torch.save(model, path)
    restored = torch.load(path, weights_only=False)

    assert_residual_posterior_equivalent(model, restored, X)
