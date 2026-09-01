from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import bochan.composition.encoders.chgnet as chgnet_module
from bochan.composition import CHGNetEncoder, MaterialEncoder
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


class FakeCrystalGraph:
    def __init__(self, structure: Any) -> None:
        lattice = structure.lattice
        frac_coords = structure.frac_coords
        self.lattice = torch.tensor(lattice.matrix, dtype=torch.float32)
        self.atom_frac_coord = torch.tensor(frac_coords, dtype=torch.float32)
        self.neighbor_image = torch.zeros((1, 3), dtype=torch.float32)

    def to(self, device: str = "cpu") -> FakeCrystalGraph:
        self.lattice = self.lattice.to(device)
        self.atom_frac_coord = self.atom_frac_coord.to(device)
        self.neighbor_image = self.neighbor_image.to(device)
        return self


class FakeGraphConverter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, structure: object) -> FakeCrystalGraph:
        self.calls += 1
        return FakeCrystalGraph(structure)


class FakeCHGNet(nn.Module):
    """Differentiable stand-in matching CHGNet's public forward contract."""

    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.scale = nn.Parameter(torch.arange(1, output_dim + 1, dtype=torch.float32))
        self.graph_converter = FakeGraphConverter()
        self.last_task: str | None = None
        self.last_return_crystal_feas: bool | None = None

    def forward(
        self,
        graphs: Sequence[FakeCrystalGraph],
        *,
        task: str = "e",
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        self.last_task = task
        self.last_return_crystal_feas = return_crystal_feas
        base = torch.stack([graph.lattice[0, 0] for graph in graphs]).unsqueeze(-1)
        crystal_fea = base * self.scale
        return {
            "e": crystal_fea[:, 0],
            "crystal_fea": crystal_fea,
        }


class MissingCrystalFeatureCHGNet(FakeCHGNet):
    def forward(
        self,
        graphs: Sequence[FakeCrystalGraph],
        *,
        task: str = "e",
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        del graphs, task, return_crystal_feas
        return {"e": torch.zeros(1)}


def test_public_composition_import_does_not_import_optional_chgnet() -> None:
    assert CHGNetEncoder is chgnet_module.CHGNetEncoder
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bochan.composition; "
                "assert 'chgnet.model.model' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_constructing_upstream_encoder_has_clear_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_chgnet(name: str):
        if name == "chgnet.model.model":
            raise ModuleNotFoundError("No module named 'chgnet'", name="chgnet")
        return __import__(name, fromlist=["*"])

    monkeypatch.setattr(chgnet_module, "import_module", missing_chgnet)

    with pytest.raises(ImportError, match=r"bochan\[materials\]"):
        CHGNetEncoder()


def test_unknown_pretrained_model_is_rejected_before_optional_import() -> None:
    with pytest.raises(ValueError, match="model_name must be one of"):
        CHGNetEncoder(model_name="unknown")


def test_injected_encoder_uses_official_crystal_feature_forward_path() -> None:
    pytest.importorskip("pymatgen")
    upstream = FakeCHGNet()
    encoder = CHGNetEncoder(upstream)

    features = encoder([_si_structure(), _si_structure()])

    assert isinstance(encoder, MaterialEncoder)
    assert encoder.output_dim == 3
    assert encoder.initialization == "injected"
    assert encoder.checkpoint_path is None
    assert features.shape == torch.Size([2, 3])
    assert upstream.graph_converter.calls == 2
    assert upstream.last_task == "e"
    assert upstream.last_return_crystal_feas is True


def test_encoder_preserves_parameter_gradients_and_dtype() -> None:
    pytest.importorskip("pymatgen")
    upstream = FakeCHGNet().double()
    encoder = CHGNetEncoder(upstream).double()

    features = encoder([_si_structure()])
    features.square().sum().backward()

    assert features.dtype == torch.double
    assert upstream.scale.grad is not None
    assert torch.isfinite(upstream.scale.grad).all()


def test_injected_checkpoint_is_loaded_without_importing_upstream_chgnet(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chgnet-state.pt"
    torch.save({"state_dict": {"scale": torch.tensor([4.0, 5.0, 6.0])}}, checkpoint)

    encoder = CHGNetEncoder(FakeCHGNet(), checkpoint=checkpoint)

    assert encoder.initialization == "checkpoint"
    assert encoder.checkpoint_path == str(checkpoint)
    assert torch.equal(encoder.encoder.scale, torch.tensor([4.0, 5.0, 6.0]))


def test_encoder_requires_crystal_feature_tensor() -> None:
    pytest.importorskip("pymatgen")
    encoder = CHGNetEncoder(MissingCrystalFeatureCHGNet())

    with pytest.raises(TypeError, match="crystal_fea"):
        encoder([_si_structure()])


def test_structure_adapter_converts_mapping_directly_to_pymatgen() -> None:
    pymatgen = pytest.importorskip("pymatgen.core")
    structure = StructureAdapter().to_pymatgen(_si_structure())

    assert isinstance(structure, pymatgen.Structure)
    assert len(structure) == 2
    assert structure.composition.reduced_formula == "Si"


def test_structure_adapter_converts_periodic_ase_directly_to_pymatgen() -> None:
    pytest.importorskip("pymatgen")
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms(
        symbols=["Si", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )

    structure = StructureAdapter().to_pymatgen(atoms)

    assert len(structure) == 2
    assert structure.composition.reduced_formula == "Si"


def test_structure_adapter_pymatgen_path_rejects_filesystem_strings() -> None:
    with pytest.raises(TypeError, match="Filesystem paths"):
        StructureAdapter().to_pymatgen("POSCAR")


def test_real_pretrained_chgnet_returns_crystal_features_on_cpu() -> None:
    pytest.importorskip("chgnet")
    pytest.importorskip("pymatgen")
    encoder = CHGNetEncoder(model_name="0.3.0")

    features = encoder([_si_structure()])

    assert encoder.initialization == "pretrained"
    assert encoder.output_dim == 64
    assert features.shape == torch.Size([1, 64])
    assert features.device.type == "cpu"
    assert features.dtype == next(encoder.encoder.parameters()).dtype
    assert torch.isfinite(features).all()
