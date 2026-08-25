from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from bochan.models.regression.gaussian.deep import ALIGNNGPModel
from bochan.structure import ALIGNNGraphBuilder, ALIGNNGraphConfig, StructureAdapter


class _FakeJarvisAtoms:
    def __init__(
        self,
        *,
        lattice_mat: Any,
        coords: Any,
        elements: list[str],
        cartesian: bool = False,
        props: Any = None,
    ) -> None:
        self.lattice_mat = np.asarray(lattice_mat, dtype=float)
        self.coords = np.asarray(coords, dtype=float)
        self.elements = list(elements)
        self.cartesian = cartesian
        self.props = props

    @classmethod
    def from_cif(
        cls,
        *,
        filename: str,
        get_primitive_atoms: bool,
        use_cif2cell: bool,
    ) -> _FakeJarvisAtoms:
        assert Path(filename).suffix == ".cif"
        assert get_primitive_atoms is False
        assert use_cif2cell is False
        return cls(
            lattice_mat=np.eye(3),
            coords=[[0.0, 0.0, 0.0]],
            elements=["Si"],
            cartesian=False,
        )

    @classmethod
    def from_poscar(cls, *, filename: str) -> _FakeJarvisAtoms:
        assert Path(filename).name.upper() == "POSCAR"
        return cls(
            lattice_mat=2.0 * np.eye(3),
            coords=[[0.0, 0.0, 0.0]],
            elements=["Al"],
            cartesian=False,
        )


class _FakeASEAtoms:
    def get_pbc(self) -> np.ndarray:
        return np.array([True, True, True])

    def get_chemical_symbols(self) -> list[str]:
        return ["Si", "O"]

    def get_cell(self) -> np.ndarray:
        return 3.0 * np.eye(3)

    def get_positions(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])


@dataclass
class _FakeSpecies:
    symbol: str


@dataclass
class _FakeLattice:
    matrix: np.ndarray


class _FakePymatgenStructure:
    is_ordered = True

    def __init__(self) -> None:
        self.lattice = _FakeLattice(4.0 * np.eye(3))
        self.species = [_FakeSpecies("Na"), _FakeSpecies("Cl")]
        self.frac_coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


class _BundleEncoder(nn.Module):
    output_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.double))

    def encode(self, graph_bundle: tuple[Any, Any, torch.Tensor]) -> torch.Tensor:
        lattice = graph_bundle[2]
        return lattice.diagonal().to(device=self.anchor.device, dtype=self.anchor.dtype)


def test_structure_adapter_accepts_mapping_and_existing_jarvis_atoms() -> None:
    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)
    existing = _FakeJarvisAtoms(
        lattice_mat=np.eye(3),
        coords=[[0.0, 0.0, 0.0]],
        elements=["Si"],
    )
    assert adapter.to_jarvis(existing) is existing

    converted = adapter.to_jarvis(
        {
            "lattice_mat": 2.0 * np.eye(3),
            "coords": [[0.25, 0.25, 0.25]],
            "elements": ["C"],
            "cartesian": False,
        }
    )
    assert isinstance(converted, _FakeJarvisAtoms)
    assert converted.elements == ["C"]
    assert converted.cartesian is False
    np.testing.assert_allclose(converted.lattice_mat, 2.0 * np.eye(3))


def test_structure_adapter_converts_ase_and_pymatgen_inputs() -> None:
    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)

    ase_atoms = adapter.to_jarvis(_FakeASEAtoms())
    assert ase_atoms.elements == ["Si", "O"]
    assert ase_atoms.cartesian is True
    np.testing.assert_allclose(ase_atoms.coords[1], [1.5, 1.5, 1.5])

    pymatgen_atoms = adapter.to_jarvis(_FakePymatgenStructure())
    assert pymatgen_atoms.elements == ["Na", "Cl"]
    assert pymatgen_atoms.cartesian is False
    np.testing.assert_allclose(pymatgen_atoms.coords[1], [0.5, 0.5, 0.5])


def test_structure_adapter_reads_supported_local_paths(tmp_path: Path) -> None:
    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_test", encoding="utf-8")
    poscar_path = tmp_path / "POSCAR"
    poscar_path.write_text("test", encoding="utf-8")

    assert adapter.to_jarvis(cif_path).elements == ["Si"]
    assert adapter.to_jarvis(poscar_path).elements == ["Al"]


def test_structure_adapter_rejects_nonperiodic_ase_and_bad_mapping() -> None:
    class _NonPeriodicASE(_FakeASEAtoms):
        def get_pbc(self) -> np.ndarray:
            return np.array([True, True, False])

    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)
    with pytest.raises(ValueError, match="periodic in all three directions"):
        adapter.to_jarvis(_NonPeriodicASE())
    with pytest.raises(ValueError, match="missing required keys"):
        adapter.to_jarvis({"elements": ["Si"]})


def test_alignn_graph_config_matches_upstream_crystal_defaults() -> None:
    config = ALIGNNGraphConfig()
    assert config.neighbor_strategy == "k-nearest"
    assert config.cutoff == pytest.approx(5.0)
    assert config.max_neighbors == 12
    assert config.atom_features == "cgcnn"
    assert config.use_canonize is True
    assert config.three_body_cutoff == pytest.approx(3.5)


def test_alignn_graph_builder_normalizes_two_tuple_to_graph_linegraph_lattice() -> None:
    calls: list[dict[str, Any]] = []

    def graph_factory(**kwargs: Any) -> tuple[str, str]:
        calls.append(kwargs)
        return "graph", "line_graph"

    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)
    builder = ALIGNNGraphBuilder(structure_adapter=adapter, graph_factory=graph_factory)
    bundle = builder.build(
        {
            "lattice_mat": 3.0 * np.eye(3),
            "coords": [[0.0, 0.0, 0.0]],
            "elements": ["Si"],
        }
    )

    graph, line_graph, lattice = bundle
    assert graph == "graph"
    assert line_graph == "line_graph"
    assert lattice.dtype == torch.float32
    torch.testing.assert_close(lattice, 3.0 * torch.eye(3))
    assert calls[0]["compute_line_graph"] is True
    assert calls[0]["neighbor_strategy"] == "k-nearest"
    assert calls[0]["atom_features"] == "cgcnn"


def test_alignn_graph_builder_preserves_returned_lattice_and_builds_bank() -> None:
    returned_lattice = 4.0 * torch.eye(3, dtype=torch.float64)

    def graph_factory(**_: Any) -> tuple[str, str, torch.Tensor]:
        return "g", "lg", returned_lattice

    config = ALIGNNGraphConfig(dtype="float64")
    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)
    builder = ALIGNNGraphBuilder(config, structure_adapter=adapter, graph_factory=graph_factory)
    structures = [
        {"lattice_mat": np.eye(3), "coords": [[0.0, 0.0, 0.0]], "elements": ["Si"]},
        {"lattice_mat": 2.0 * np.eye(3), "coords": [[0.0, 0.0, 0.0]], "elements": ["Ge"]},
    ]

    bank = builder.build_many(structures)
    assert len(bank) == 2
    assert bank[0][2].dtype == torch.float64
    torch.testing.assert_close(bank[0][2], returned_lattice)


def test_alignn_graph_bank_connects_to_phase1_gp_model() -> None:
    def graph_factory(**_: Any) -> tuple[str, str]:
        return "g", "lg"

    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)
    builder = ALIGNNGraphBuilder(structure_adapter=adapter, graph_factory=graph_factory)
    structures = [
        {"lattice_mat": 3.0 * np.eye(3), "coords": [[0.0, 0.0, 0.0]], "elements": ["Si"]},
        {"lattice_mat": 4.0 * np.eye(3), "coords": [[0.0, 0.0, 0.0]], "elements": ["Ge"]},
    ]
    graph_bank = builder.build_many(structures)
    train_X = torch.tensor(
        [[0.0, 900.0], [1.0, 950.0], [0.0, 1000.0], [1.0, 1050.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.2], [0.7], [0.3], [0.8]], dtype=torch.double)

    model = ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=graph_bank,
        encoder=_BundleEncoder(),
        latent_dim=2,
        outcome_transform=None,
    )
    posterior = model.posterior(torch.tensor([[1.0, 975.0]], dtype=torch.double))

    assert model.num_structures == 2
    assert posterior.mean.shape == torch.Size([1, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_alignn_graph_builder_rejects_invalid_factory_contract() -> None:
    adapter = StructureAdapter(atoms_class=_FakeJarvisAtoms)

    def graph_factory(**_: Any) -> tuple[str]:
        return ("graph",)

    builder = ALIGNNGraphBuilder(structure_adapter=adapter, graph_factory=graph_factory)
    with pytest.raises(ValueError, match="exactly two or three values"):
        builder.build(
            {"lattice_mat": np.eye(3), "coords": [[0.0, 0.0, 0.0]], "elements": ["Si"]}
        )
