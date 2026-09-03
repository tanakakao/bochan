"""Contracts for CHGNet direct-energy residual GP support."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from bochan.composition.encoders.chgnet import CHGNetEncoder
from bochan.models.regression.gaussian.materials import get_material_family
from bochan.models.regression.gaussian.materials.structure import (
    CHGNetDirectEnergyPredictor,
    CHGNetResidualGPModel,
)


class _RawCHGNet(nn.Module):
    def forward(
        self,
        graphs: list[Any],
        *,
        task: str,
        return_crystal_feas: bool,
    ) -> dict[str, Tensor]:
        assert task == "e"
        assert return_crystal_feas is False
        values = torch.tensor(
            [float(graph) + 0.5 for graph in graphs],
            dtype=torch.float32,
        )
        return {"e": values}


class _FakeCHGNetEncoder(CHGNetEncoder):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.encoder = _RawCHGNet()
        self._output_dim = 1

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _floating_reference(self) -> Tensor | None:
        return torch.empty((), dtype=torch.float32)

    def _prepare_graph(
        self,
        value: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        del device, dtype
        return value


def test_chgnet_direct_energy_predictor_maps_structure_indices_and_ignores_process() -> None:
    predictor = CHGNetDirectEnergyPredictor(_FakeCHGNetEncoder(), [10, 20, 30])
    X = torch.tensor(
        [
            [2.0, 100.0],
            [0.0, -5.0],
            [2.0, 999.0],
        ],
        dtype=torch.float64,
    )

    prediction = predictor(X)

    assert prediction.dtype == torch.float64
    assert prediction.shape == torch.Size([3, 1])
    assert torch.allclose(prediction[:, 0], torch.tensor([30.5, 10.5, 30.5]))


def test_chgnet_direct_energy_predictor_preserves_leading_batch_dimensions() -> None:
    predictor = CHGNetDirectEnergyPredictor(_FakeCHGNetEncoder(), [1, 2])
    X = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]])

    prediction = predictor(X)

    assert prediction.shape == torch.Size([2, 2, 1])
    assert torch.allclose(
        prediction[..., 0],
        torch.tensor([[1.5, 2.5], [2.5, 1.5]]),
    )


@pytest.mark.parametrize(
    "index",
    [0.25, -1.0, 2.0, float("nan")],
)
def test_chgnet_direct_energy_predictor_rejects_invalid_structure_indices(index: float) -> None:
    predictor = CHGNetDirectEnergyPredictor(_FakeCHGNetEncoder(), [1, 2])
    X = torch.tensor([[index]])

    with pytest.raises(ValueError):
        predictor(X)


def test_chgnet_registry_advertises_verified_residual_energy_support() -> None:
    registration = get_material_family("chgnet")

    assert registration.supports("residual_gp") is True
    assert registration.resolve_model_class("residual_gp") is CHGNetResidualGPModel
    assert registration.pretrained.capabilities.direct_prediction is True
    assert registration.pretrained.capabilities.residual_gp is True
    assert registration.pretrained.supports_residual_gp is True


def test_other_material_families_remain_conservative_until_adapters_exist() -> None:
    for family in ("crabnet", "roost", "alignn", "m3gnet", "mace"):
        registration = get_material_family(family)
        assert registration.supports("residual_gp") is False
        assert registration.pretrained.capabilities.residual_gp is False
