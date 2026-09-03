"""Contracts for correlated multi-output material residual GPs."""

from __future__ import annotations

import torch
from torch import Tensor

from bochan.models.regression.gaussian.materials.common import (
    DirectMaterialPredictor,
    SingleOutputBaselineAdapter,
    compute_material_residual_targets,
    get_material_family,
)


class _ScalarPredictor(DirectMaterialPredictor):
    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: Tensor) -> Tensor:
        return X[..., :1] + 10.0


def test_single_output_baseline_adapter_places_pretrained_value_in_selected_column() -> None:
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    adapter = SingleOutputBaselineAdapter(
        _ScalarPredictor(),
        output_dim=3,
        output_index=1,
    )

    baseline = adapter(X)

    assert baseline.shape == torch.Size([2, 3])
    assert torch.allclose(
        baseline,
        torch.tensor([[0.0, 11.0, 0.0], [0.0, 13.0, 0.0]]),
    )


def test_single_output_baseline_adapter_supports_negative_index() -> None:
    adapter = SingleOutputBaselineAdapter(
        _ScalarPredictor(),
        output_dim=3,
        output_index=-1,
    )
    assert adapter.output_index == 2


def test_multitask_residual_targets_only_subtract_selected_pretrained_output() -> None:
    X = torch.tensor([[1.0], [2.0]])
    Y = torch.tensor([[100.0, 11.5, 5.0], [200.0, 13.0, 7.0]])
    adapter = SingleOutputBaselineAdapter(
        _ScalarPredictor(),
        output_dim=3,
        output_index=1,
    )

    residual = compute_material_residual_targets(X, Y, adapter)

    assert torch.allclose(
        residual,
        torch.tensor([[100.0, 0.5, 5.0], [200.0, 1.0, 7.0]]),
    )


def test_multitask_residual_preserves_partial_observations() -> None:
    X = torch.tensor([[1.0], [2.0]])
    Y = torch.tensor([[100.0, float("nan")], [200.0, 12.5]])
    adapter = SingleOutputBaselineAdapter(
        _ScalarPredictor(),
        output_dim=2,
        output_index=1,
    )

    residual = compute_material_residual_targets(X, Y, adapter)

    assert torch.isnan(residual[0, 1])
    assert residual[0, 0].item() == 100.0
    assert residual[1, 0].item() == 200.0


def test_residual_ready_structure_families_register_multitask_variants() -> None:
    for family in ("chgnet", "m3gnet", "mace"):
        registration = get_material_family(family)
        assert registration.supports("multitask_residual_gp")
        assert registration.supports("mixed_multitask_residual_gp")
        assert registration.resolve_model_class("multitask_residual_gp").__name__ == (
            {"chgnet": "CHGNet", "m3gnet": "M3GNet", "mace": "MACE"}[family]
            + "MultiTaskResidualGPModel"
        )
        assert registration.resolve_model_class("mixed_multitask_residual_gp").__name__ == (
            {"chgnet": "CHGNet", "m3gnet": "M3GNet", "mace": "MACE"}[family]
            + "MixedMultiTaskResidualGPModel"
        )


def test_non_residual_families_do_not_advertise_multitask_residual() -> None:
    for family in ("crabnet", "roost", "alignn"):
        registration = get_material_family(family)
        assert not registration.supports("multitask_residual_gp")
        assert not registration.supports("mixed_multitask_residual_gp")
