"""Contracts for correlated multi-output material residual GPs."""

from __future__ import annotations

import torch
from torch import Tensor

from bochan.models.regression.gaussian.materials.common.registry import get_material_family
from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    RoutedDirectMaterialPredictor,
    compute_material_residual_targets,
)
from bochan.models.regression.gaussian.materials.structure import (
    CHGNetMultiTaskResidualGPModel,
    M3GNetMultiTaskResidualGPModel,
    MACEMultiTaskResidualGPModel,
)


class _ScalarPredictor(DirectMaterialPredictor):
    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: Tensor) -> Tensor:
        return X[..., :1] + 10.0


def test_routed_predictor_places_scalar_baseline_on_selected_output() -> None:
    X = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    predictor = RoutedDirectMaterialPredictor(
        _ScalarPredictor(),
        output_dim=3,
        output_index=1,
    )

    baseline = predictor(X)

    assert baseline.shape == torch.Size([2, 3])
    assert torch.allclose(
        baseline,
        torch.tensor([[0.0, 11.0, 0.0], [0.0, 12.0, 0.0]]),
    )


def test_routed_predictor_supports_negative_output_index() -> None:
    predictor = RoutedDirectMaterialPredictor(
        _ScalarPredictor(),
        output_dim=3,
        output_index=-1,
    )
    assert predictor.output_index == 2


def test_routed_residual_targets_preserve_unrouted_outputs_and_nan() -> None:
    X = torch.tensor([[1.0], [2.0]])
    Y = torch.tensor([[3.0, 20.0, 7.0], [float("nan"), 25.0, 8.0]])
    predictor = RoutedDirectMaterialPredictor(
        _ScalarPredictor(),
        output_dim=3,
        output_index=1,
    )

    residual = compute_material_residual_targets(X, Y, predictor)

    assert torch.allclose(residual[0], torch.tensor([3.0, 9.0, 7.0]))
    assert torch.isnan(residual[1, 0])
    assert torch.allclose(residual[1, 1:], torch.tensor([13.0, 8.0]))


def test_routed_predictor_rejects_invalid_contracts() -> None:
    for output_dim, output_index in ((1, 0), (3, 3), (3, -4)):
        try:
            RoutedDirectMaterialPredictor(
                _ScalarPredictor(),
                output_dim=output_dim,
                output_index=output_index,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid routed predictor contract was accepted")


def test_registry_exposes_multitask_residual_variants() -> None:
    expected = {
        "chgnet": CHGNetMultiTaskResidualGPModel,
        "m3gnet": M3GNetMultiTaskResidualGPModel,
        "mace": MACEMultiTaskResidualGPModel,
    }
    for family, model_class in expected.items():
        registration = get_material_family(family)
        assert registration.supports("multitask_residual_gp")
        assert registration.resolve_model_class("multitask_residual_gp") is model_class


def test_non_direct_families_do_not_claim_multitask_residual_support() -> None:
    for family in ("crabnet", "roost", "alignn"):
        assert not get_material_family(family).supports("multitask_residual_gp")
