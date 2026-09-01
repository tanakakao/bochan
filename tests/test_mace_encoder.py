from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import bochan.composition.encoders.mace as mace_module
from bochan.composition import MACEEncoder, MaterialEncoder
from bochan.structure import StructureAdapter


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


def _fake_batch(structure: dict[str, object]) -> dict[str, Tensor]:
    lattice = torch.tensor(structure["lattice_mat"], dtype=torch.float32)
    coords = torch.tensor(structure["coords"], dtype=torch.float32)
    positions = coords if bool(structure.get("cartesian", False)) else coords @ lattice
    return {"positions": positions}


class FakeDescriptorLinear(nn.Linear):
    def __init__(self, width: int) -> None:
        super().__init__(width, width, bias=False)
        self.irreps_out = f"{width}x0e + {width}x1o"


class FakeProduct(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = FakeDescriptorLinear(width)


class FakeMACE(nn.Module):
    """Small differentiable stand-in for the MACE 0.3.x descriptor contract."""

    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor([14], dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("num_interactions", torch.tensor(2, dtype=torch.int64))
        self.heads = ["Default"]
        self.node_embedding = nn.Linear(3, width, bias=False)
        self.radial_embedding = nn.Linear(1, width, bias=False)
        self.spherical_harmonics = nn.Identity()
        self.interactions = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2)])
        self.products = nn.ModuleList([FakeProduct(width) for _ in range(2)])
        self.readouts = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
        self.last_node_feats: Tensor | None = None

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        positions = data["positions"]
        first_invariants = self.node_embedding(positions)
        equivariant = torch.cat([positions, positions], dim=-1)
        final_invariants = self.interactions[-1](first_invariants)
        node_feats = torch.cat([first_invariants, equivariant, final_invariants], dim=-1)
        self.last_node_feats = node_feats
        return {
            "node_feats": node_feats,
            "energy": self.readouts[-1](final_invariants).sum(),
        }


class MissingNodeFeaturesMACE(FakeMACE):
    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        del data
        return {"energy": torch.zeros(())}


def test_public_composition_import_does_not_import_optional_mace() -> None:
    assert MACEEncoder is mace_module.MACEEncoder
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bochan.composition; assert 'mace' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_constructing_pretrained_encoder_has_clear_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = mace_module.import_module

    def missing_mace(name: str):
        if name.startswith("mace"):
            raise ModuleNotFoundError("No module named 'mace'", name="mace")
        return original_import(name)

    monkeypatch.setattr(mace_module, "import_module", missing_mace)

    with pytest.raises(ImportError, match=r"bochan\[materials\]"):
        MACEEncoder()


def test_structure_adapter_converts_mapping_directly_to_periodic_ase() -> None:
    ase = pytest.importorskip("ase")

    atoms = StructureAdapter().to_ase(_si_structure())

    assert isinstance(atoms, ase.Atoms)
    assert atoms.get_chemical_symbols() == ["Si", "Si"]
    assert atoms.get_pbc().all()
    assert torch.allclose(
        torch.tensor(atoms.get_positions(), dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]], dtype=torch.float64),
    )


def test_structure_adapter_converts_pymatgen_to_ase() -> None:
    pytest.importorskip("ase")
    pytest.importorskip("pymatgen")
    adapter = StructureAdapter()
    structure = adapter.to_pymatgen(_si_structure())

    atoms = adapter.to_ase(structure)

    assert len(atoms) == 2
    assert atoms.get_pbc().all()


def test_structure_adapter_rejects_nonperiodic_ase_for_mace() -> None:
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms(symbols=["Si"], positions=[[0.0, 0.0, 0.0]], pbc=False)

    with pytest.raises(ValueError, match="periodic"):
        StructureAdapter().to_ase(atoms)


def test_injected_mace_uses_differentiable_invariant_mean_pooling() -> None:
    pytest.importorskip("mace")
    upstream = FakeMACE()
    encoder = MACEEncoder(upstream, batch_builder=_fake_batch)

    features = encoder([_si_structure(), _si_structure()])

    assert isinstance(encoder, MaterialEncoder)
    assert encoder.initialization == "injected"
    assert encoder.model_name == "medium-mpa-0"
    assert encoder.num_layers == 2
    assert encoder.num_interactions == 2
    assert encoder.output_dim == 4
    assert encoder.pooling == "mean"
    assert features.shape == torch.Size([2, 4])
    assert torch.isfinite(features).all()

    features.square().sum().backward()
    assert upstream.node_embedding.weight.grad is not None
    assert torch.isfinite(upstream.node_embedding.weight.grad).all()


def test_num_layers_limits_invariant_descriptor_width() -> None:
    pytest.importorskip("mace")
    encoder = MACEEncoder(FakeMACE(), num_layers=1, batch_builder=_fake_batch)

    features = encoder([_si_structure()])

    assert encoder.num_layers == 1
    assert encoder.output_dim == 2
    assert features.shape == torch.Size([1, 2])


def test_sum_pooling_is_explicit_and_differs_from_mean() -> None:
    pytest.importorskip("mace")
    upstream_mean = FakeMACE()
    upstream_sum = FakeMACE()
    upstream_sum.load_state_dict(upstream_mean.state_dict())
    mean_encoder = MACEEncoder(upstream_mean, pooling="mean", batch_builder=_fake_batch)
    sum_encoder = MACEEncoder(upstream_sum, pooling="sum", batch_builder=_fake_batch)

    mean_features = mean_encoder([_si_structure()])
    sum_features = sum_encoder([_si_structure()])

    torch.testing.assert_close(sum_features, 2.0 * mean_features)


def test_outer_double_preserves_native_mace_dtype_and_autograd() -> None:
    pytest.importorskip("mace")
    upstream = FakeMACE()
    encoder = MACEEncoder(upstream, batch_builder=_fake_batch).double()

    features = encoder([_si_structure()])
    features.square().sum().backward()

    assert next(upstream.parameters()).dtype == torch.float32
    assert features.dtype == torch.float64
    assert upstream.node_embedding.weight.grad is not None
    assert torch.isfinite(upstream.node_embedding.weight.grad).all()


def test_backbone_modules_exclude_original_energy_readouts() -> None:
    pytest.importorskip("mace")
    upstream = FakeMACE()
    encoder = MACEEncoder(upstream, batch_builder=_fake_batch)

    backbone = encoder.backbone_modules()

    assert upstream.node_embedding in backbone
    assert upstream.interactions in backbone
    assert upstream.products in backbone
    assert upstream.readouts not in backbone


def test_raw_mace_forward_must_expose_node_features() -> None:
    pytest.importorskip("mace")
    encoder = MACEEncoder(MissingNodeFeaturesMACE(), batch_builder=_fake_batch)

    with pytest.raises(TypeError, match="node_feats"):
        encoder([_si_structure()])


def test_num_layers_cannot_exceed_mace_interactions() -> None:
    pytest.importorskip("mace")

    with pytest.raises(ValueError, match="num_layers exceeds"):
        MACEEncoder(FakeMACE(), num_layers=3, batch_builder=_fake_batch)


def test_real_pretrained_mace_returns_invariant_crystal_representation_on_cpu() -> None:
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    encoder = MACEEncoder(model_name="medium-mpa-0")
    encoder.eval()

    with torch.no_grad():
        features = encoder([_si_structure()])

    assert encoder.initialization == "pretrained"
    assert encoder.model_name == "medium-mpa-0"
    assert encoder.num_layers == encoder.num_interactions
    assert encoder.output_dim > 0
    assert features.shape == torch.Size([1, encoder.output_dim])
    assert features.device.type == "cpu"
    assert features.dtype == next(encoder.encoder.parameters()).dtype
    assert torch.isfinite(features).all()
    assert encoder.encoder.readouts not in encoder.backbone_modules()
