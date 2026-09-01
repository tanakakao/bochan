from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import bochan.composition.encoders.m3gnet as m3gnet_module
from bochan.composition import M3GNetEncoder, MaterialEncoder


def _si_structure() -> dict[str, object]:
    return {
        "lattice_mat": [
            [5.43, 0.0, 0.0],
            [0.0, 5.43, 0.0],
            [0.0, 0.0, 5.43],
        ],
        "coords": [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


class FakeM3GNetGraph:
    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
        self.pbc_offset = torch.zeros((2, 3), dtype=torch.float32)
        self.pos = torch.empty_like(self.frac_coords)
        self.pbc_offshift = torch.empty_like(self.pbc_offset)

    def to(self, device: torch.device | str) -> FakeM3GNetGraph:
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pos = self.pos.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        return self


class FakeM3GNetConverter:
    def __init__(self) -> None:
        self.calls = 0

    def get_graph(self, structure: Any) -> tuple[FakeM3GNetGraph, Tensor, list[float]]:
        self.calls += 1
        graph = FakeM3GNetGraph(structure)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        return graph, lattice, [0.0, 0.0]


class FakeM3GNet(nn.Module):
    """Differentiable stand-in for MatGL's bare M3GNet forward contract."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.is_intensive = True
        self.include_state = False
        self.element_types = ("Si",)
        self.cutoff = 5.0
        self.embedding = nn.Linear(3, output_dim, bias=False)
        self.readout = nn.Identity()
        self.final_layer = nn.Linear(output_dim, 1)
        self.feature_dict: dict[str, Tensor] = {}
        self.last_state_attr: Tensor | None | object = object()

    def forward(self, g: FakeM3GNetGraph, state_attr: Tensor | None = None) -> Tensor:
        self.last_state_attr = state_attr
        readout = self.embedding(g.pos.mean(dim=0)).unsqueeze(0)
        self.feature_dict = {"readout": readout, "final": self.final_layer(readout)}
        return self.feature_dict["final"].squeeze()


class MissingReadoutM3GNet(FakeM3GNet):
    def forward(self, g: FakeM3GNetGraph, state_attr: Tensor | None = None) -> Tensor:
        del g, state_attr
        self.feature_dict = {"final": torch.zeros(1)}
        return torch.zeros(())


class ExtensiveM3GNet(FakeM3GNet):
    def __init__(self) -> None:
        super().__init__()
        self.is_intensive = False


def test_public_composition_import_does_not_import_optional_matgl() -> None:
    assert M3GNetEncoder is m3gnet_module.M3GNetEncoder
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bochan.composition; assert 'matgl' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_constructing_pretrained_encoder_has_clear_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = m3gnet_module.import_module

    def missing_matgl(name: str):
        if name in {"matgl", "matgl.ext.pymatgen"}:
            raise ModuleNotFoundError("No module named 'matgl'", name="matgl")
        return original_import(name)

    monkeypatch.setattr(m3gnet_module, "import_module", missing_matgl)

    with pytest.raises(ImportError, match=r"bochan\[materials\]"):
        M3GNetEncoder()


def test_injected_encoder_uses_direct_differentiable_readout_path() -> None:
    pytest.importorskip("pymatgen")
    upstream = FakeM3GNet()
    converter = FakeM3GNetConverter()
    encoder = M3GNetEncoder(upstream, graph_converter=converter)

    features = encoder([_si_structure(), _si_structure()])

    assert isinstance(encoder, MaterialEncoder)
    assert encoder.output_dim == 4
    assert encoder.initialization == "injected"
    assert features.shape == torch.Size([2, 4])
    assert converter.calls == 2
    assert upstream.last_state_attr is None
    assert torch.isfinite(features).all()


def test_encoder_accepts_periodic_ase_structures() -> None:
    pytest.importorskip("pymatgen")
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms(
        symbols=["Si", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    converter = FakeM3GNetConverter()
    encoder = M3GNetEncoder(FakeM3GNet(), graph_converter=converter)

    features = encoder([atoms])

    assert features.shape == torch.Size([1, 4])
    assert converter.calls == 1


def test_outer_double_preserves_native_encoder_dtype_and_autograd() -> None:
    pytest.importorskip("pymatgen")
    upstream = FakeM3GNet()
    encoder = M3GNetEncoder(upstream, graph_converter=FakeM3GNetConverter()).double()

    features = encoder([_si_structure()])
    features.square().sum().backward()

    assert next(upstream.parameters()).dtype == torch.float32
    assert features.dtype == torch.float64
    assert upstream.embedding.weight.grad is not None
    assert torch.isfinite(upstream.embedding.weight.grad).all()


def test_backbone_modules_exclude_original_property_head() -> None:
    upstream = FakeM3GNet()
    encoder = M3GNetEncoder(upstream, graph_converter=FakeM3GNetConverter())

    backbone = encoder.backbone_modules()

    assert upstream.embedding in backbone
    assert upstream.readout in backbone
    assert upstream.final_layer not in backbone


def test_extensive_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="intensive"):
        M3GNetEncoder(ExtensiveM3GNet(), graph_converter=FakeM3GNetConverter())


def test_encoder_requires_feature_dict_readout_tensor() -> None:
    pytest.importorskip("pymatgen")
    encoder = M3GNetEncoder(MissingReadoutM3GNet(), graph_converter=FakeM3GNetConverter())

    with pytest.raises(TypeError, match="readout"):
        encoder([_si_structure()])


def test_real_pretrained_m3gnet_returns_graph_readout_on_cpu() -> None:
    pytest.importorskip("matgl")
    pytest.importorskip("pymatgen")
    encoder = M3GNetEncoder(model_name="M3GNet-PES-MatPES-PBE-2025.2")
    encoder.eval()

    with torch.no_grad():
        features = encoder([_si_structure()])

    assert encoder.initialization == "pretrained"
    assert encoder.output_dim > 0
    assert features.shape == torch.Size([1, encoder.output_dim])
    assert features.device.type == "cpu"
    assert features.dtype == next(encoder.encoder.parameters()).dtype
    assert torch.isfinite(features).all()
    assert "readout" in encoder.encoder.feature_dict
