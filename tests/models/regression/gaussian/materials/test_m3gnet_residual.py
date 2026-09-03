"""Contract tests for M3GNet residual-GP support without MatGL dependency."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bochan.composition.encoders.m3gnet import M3GNetEncoder
from bochan.models.regression.gaussian.materials import get_material_family
from bochan.models.regression.gaussian.materials.structure.m3gnet_residual import (
    M3GNetDirectPredictor,
)


class _FakeGraph:
    def to(self, device):
        return self


class _FakeM3GNet(nn.Module):
    output_dim = 2
    is_intensive = True
    element_types = ("H",)
    cutoff = 5.0

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.final_layer = nn.Linear(2, 1)

    def forward(self, *, g, state_attr=None):
        value = getattr(g, "value", 0.0)
        return self.weight.new_tensor(float(value)).reshape(1)


class _FakeAdapter:
    pass


def _encoder() -> M3GNetEncoder:
    encoder = object.__new__(M3GNetEncoder)
    nn.Module.__init__(encoder)
    encoder.encoder = _FakeM3GNet()
    encoder._output_dim = 2
    encoder._model_name = "fake"
    encoder._initialization = "injected"
    encoder._representation_mode = "readout"
    encoder.adapter = _FakeAdapter()
    encoder.graph_converter = object()
    encoder.register_buffer("_output_reference", torch.empty(0), persistent=False)

    def prepare_graph(structure):
        graph = _FakeGraph()
        graph.value = float(structure)
        return graph, None

    encoder._prepare_graph = prepare_graph  # type: ignore[method-assign]
    return encoder


def test_m3gnet_direct_predictor_uses_structure_index_and_ignores_process_columns() -> None:
    predictor = M3GNetDirectPredictor(_encoder(), [1.5, 2.5, 4.0])
    X = torch.tensor([[2.0, 100.0], [0.0, -3.0], [2.0, 999.0]])

    result = predictor(X)

    assert result.shape == (3, 1)
    assert torch.allclose(result[:, 0], torch.tensor([4.0, 1.5, 4.0]))


def test_m3gnet_direct_predictor_preserves_leading_dimensions() -> None:
    predictor = M3GNetDirectPredictor(_encoder(), [1.0, 3.0])
    X = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]])

    result = predictor(X)

    assert result.shape == (2, 2, 1)
    assert torch.allclose(result[..., 0], torch.tensor([[1.0, 3.0], [3.0, 1.0]]))


def test_m3gnet_direct_predictor_rejects_non_integer_and_out_of_range_indices() -> None:
    predictor = M3GNetDirectPredictor(_encoder(), [1.0, 2.0])

    try:
        predictor(torch.tensor([[0.5]]))
    except ValueError as error:
        assert "integer-valued" in str(error)
    else:
        raise AssertionError("Expected non-integer structure index to fail.")

    try:
        predictor(torch.tensor([[2.0]]))
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("Expected out-of-range structure index to fail.")


def test_m3gnet_registry_enables_verified_residual_variant() -> None:
    registration = get_material_family("m3gnet")

    assert registration.supports("residual_gp") is True
    assert registration.pretrained.capabilities.direct_prediction is True
    assert registration.pretrained.capabilities.residual_gp is True
    assert registration.model_path("residual_gp").endswith(":M3GNetResidualGPModel")
