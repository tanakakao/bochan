from __future__ import annotations

from io import BytesIO
from json import dumps
from zipfile import ZipFile

import pytest
import torch

import bochan.structure.alignn as alignn_structure
from bochan.structure import (
    ALIGNNGraphBuilder,
    StructureAdapter,
    load_alignn_pretrained_bundle,
)


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


def test_structure_adapter_mapping_and_ase_inputs() -> None:
    from ase import Atoms as ASEAtoms

    adapter = StructureAdapter()
    jarvis_atoms = adapter.adapt(_si_structure())

    assert jarvis_atoms.elements == ["Si", "Si"]
    assert jarvis_atoms.num_atoms == 2

    ase_atoms = ASEAtoms(
        symbols=["Si", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    converted = adapter.adapt(ase_atoms)

    assert converted.elements == ["Si", "Si"]
    assert converted.num_atoms == 2


def test_structure_adapter_rejects_nonperiodic_ase() -> None:
    from ase import Atoms as ASEAtoms

    adapter = StructureAdapter()
    ase_atoms = ASEAtoms(
        symbols=["Si"],
        positions=[[0.0, 0.0, 0.0]],
        cell=[5.43, 5.43, 5.43],
        pbc=[True, True, False],
    )

    with pytest.raises(ValueError, match="periodic in all three directions"):
        adapter.adapt(ase_atoms)


def test_structure_adapter_rejects_disordered_pymatgen_before_conversion() -> None:
    disordered_class = type(
        "FakeStructure",
        (),
        {"__module__": "pymatgen.core.structure", "is_ordered": False},
    )

    with pytest.raises(ValueError, match="Disordered pymatgen"):
        StructureAdapter().adapt(disordered_class())


def test_structure_adapter_uses_explicit_file_boundary(tmp_path) -> None:
    adapter = StructureAdapter()
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Si\n"
        "1.0\n"
        "5.43 0.0 0.0\n"
        "0.0 5.43 0.0\n"
        "0.0 0.0 5.43\n"
        "Si\n"
        "2\n"
        "Direct\n"
        "0.0 0.0 0.0\n"
        "0.25 0.25 0.25\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="from_file"):
        adapter.adapt(str(poscar))

    atoms = adapter.from_file(poscar)
    assert atoms.elements == ["Si", "Si"]
    assert atoms.num_atoms == 2


def test_structure_adapter_reads_cif_without_primitive_reduction(tmp_path) -> None:
    adapter = StructureAdapter()
    cif = tmp_path / "silicon.cif"
    cif.write_text(
        "data_silicon\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 5.43\n"
        "_cell_length_b 5.43\n"
        "_cell_length_c 5.43\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_symmetry_equiv_pos_as_xyz\n"
        "'x,y,z'\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Si1 Si 0.0 0.0 0.0\n"
        "Si2 Si 0.25 0.25 0.25\n",
        encoding="utf-8",
    )

    atoms = adapter.from_file(cif)

    assert atoms.elements == ["Si", "Si"]
    assert atoms.num_atoms == 2


def test_structure_mapping_validation_is_strict() -> None:
    adapter = StructureAdapter()
    invalid = _si_structure()
    invalid["coords"] = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="same number of atoms"):
        adapter.adapt(invalid)

    singular = _si_structure()
    singular["lattice_mat"] = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="non-singular"):
        adapter.adapt(singular)


def test_alignn_graph_builder_defaults_match_current_pure_scalar_recipe() -> None:
    builder = ALIGNNGraphBuilder()

    assert builder.config == {
        "neighbor_strategy": "pure_torch",
        "cutoff": 8.0,
        "max_neighbors": 12,
        "atom_features": "cgcnn",
        "compute_line_graph": True,
        "dtype": "float32",
        "three_body_cutoff": 3.5,
    }


def test_alignn_graph_builder_rejects_non_pure_neighbor_strategy() -> None:
    with pytest.raises(ValueError, match="neighbor_strategy='pure_torch'"):
        ALIGNNGraphBuilder(neighbor_strategy="k-nearest")


def test_alignn_graph_builder_forwards_pure_graph_settings(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_builder(**kwargs: object) -> tuple[str, str]:
        calls.append(dict(kwargs))
        return "graph", "line_graph"

    monkeypatch.setattr(alignn_structure, "_pure_graph_builder", lambda: fake_builder)
    builder = ALIGNNGraphBuilder(
        cutoff=7.5,
        max_neighbors=10,
        atom_features="cgcnn",
        compute_line_graph=True,
        dtype="float64",
        three_body_cutoff=3.0,
    )

    result = builder.build(_si_structure())

    assert result == ("graph", "line_graph")
    assert len(calls) == 1
    call = calls[0]
    assert call["two_body_cutoff"] == 7.5
    assert call["three_body_cutoff"] == 3.0
    assert call["max_neighbors"] == 10
    assert call["atom_features"] == "cgcnn"
    assert call["compute_line_graph"] is True
    assert call["use_matscipy_topology"] is False
    assert call["positions"].dtype == torch.float64  # type: ignore[union-attr]
    assert call["lattice"].dtype == torch.float64  # type: ignore[union-attr]
    assert call["atoms"].elements == ["Si", "Si"]  # type: ignore[union-attr]


def test_alignn_graph_builder_uses_pure_training_config(monkeypatch) -> None:
    def fake_builder(**kwargs: object) -> tuple[str, str]:
        return "graph", "line_graph"

    monkeypatch.setattr(alignn_structure, "_pure_graph_builder", lambda: fake_builder)
    builder = ALIGNNGraphBuilder.from_training_config(
        {
            "neighbor_strategy": "pure_torch",
            "cutoff": 5.5,
            "max_neighbors": 8,
            "atom_features": "atomic_number",
            "compute_line_graph": True,
            "dtype": "float64",
            "three_body_cutoff": 3.0,
        }
    )

    graphs = builder.build_many([_si_structure(), _si_structure()])

    assert graphs == (("graph", "line_graph"), ("graph", "line_graph"))
    assert builder.config == {
        "neighbor_strategy": "pure_torch",
        "cutoff": 5.5,
        "max_neighbors": 8,
        "atom_features": "atomic_number",
        "compute_line_graph": True,
        "dtype": "float64",
        "three_body_cutoff": 3.0,
    }


def test_real_pure_torch_graph_builder_runs_without_dgl_graphs() -> None:
    pytest.importorskip("alignn")

    graph, line_graph = ALIGNNGraphBuilder(cutoff=5.0).build(_si_structure())

    assert graph.__class__.__module__ == "alignn.torch_graph_builder"
    assert graph.__class__.__name__ == "TorchGraph"
    assert line_graph.__class__.__module__ == "alignn.torch_graph_builder"
    assert graph.num_nodes == 2
    assert graph.src.dtype == torch.long
    assert graph.dst.dtype == torch.long
    assert graph.ndata["atom_features"].shape == (2, 92)
    assert graph.edata["r"].shape[-1] == 3
    assert line_graph.num_nodes == graph.num_edges
    assert "h" in line_graph.edata


def _pure_bundle_config() -> dict[str, object]:
    return {
        "neighbor_strategy": "pure_torch",
        "cutoff": 8.0,
        "max_neighbors": 12,
        "atom_features": "cgcnn",
        "compute_line_graph": True,
        "dtype": "float32",
        "three_body_cutoff": 3.5,
        "model": {
            "name": "alignn_atomwise_pure",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "hidden_features": 256,
            "output_features": 1,
            "calculate_gradient": False,
            "gradwise_weight": 0.0,
            "energy_mult_natoms": False,
        },
    }


def test_local_pretrained_zip_keeps_pure_model_and_graph_configs_together(
    tmp_path, monkeypatch
) -> None:
    config = _pure_bundle_config()
    checkpoint_buffer = BytesIO()
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint_buffer)

    bundle_path = tmp_path / "alignn_model.zip"
    with ZipFile(bundle_path, "w") as archive:
        archive.writestr("run/config.json", dumps(config))
        archive.writestr("run/best_model.pt", checkpoint_buffer.getvalue())

    bundle = load_alignn_pretrained_bundle(bundle_path)
    graph_builder = bundle.build_graph_builder()

    assert bundle.checkpoint_name == "run/best_model.pt"
    assert bundle.model_config["name"] == "alignn_atomwise_pure"
    assert graph_builder.cutoff == 8.0
    assert graph_builder.max_neighbors == 12
    assert graph_builder.neighbor_strategy == "pure_torch"
    assert graph_builder.three_body_cutoff == 3.5

    captured: dict[str, object] = {}

    class FakeEncoder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(alignn_structure, "ALIGNNEncoder", FakeEncoder)
    bundle.build_encoder(strict=False)

    assert captured["config"] == config["model"]
    assert captured["strict_checkpoint"] is False
    assert isinstance(captured["checkpoint"], dict)


def test_pretrained_bundle_selects_numbered_checkpoint_numerically(tmp_path) -> None:
    config = _pure_bundle_config()
    checkpoint_9 = BytesIO()
    checkpoint_10 = BytesIO()
    torch.save({"model": {"step": torch.tensor([9])}}, checkpoint_9)
    torch.save({"model": {"step": torch.tensor([10])}}, checkpoint_10)

    bundle_path = tmp_path / "checkpoints.zip"
    with ZipFile(bundle_path, "w") as archive:
        archive.writestr("config.json", dumps(config))
        archive.writestr("checkpoint_9.pt", checkpoint_9.getvalue())
        archive.writestr("checkpoint_10.pt", checkpoint_10.getvalue())

    bundle = load_alignn_pretrained_bundle(bundle_path)

    assert bundle.checkpoint_name == "checkpoint_10.pt"


def test_pretrained_bundle_rejects_legacy_dgl_alignn(tmp_path) -> None:
    config = {
        "neighbor_strategy": "k-nearest",
        "model": {"name": "alignn"},
    }
    checkpoint_buffer = BytesIO()
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint_buffer)
    bundle_path = tmp_path / "legacy.zip"
    with ZipFile(bundle_path, "w") as archive:
        archive.writestr("config.json", dumps(config))
        archive.writestr("best_model.pt", checkpoint_buffer.getvalue())

    with pytest.raises(NotImplementedError, match="pure-PyTorch"):
        load_alignn_pretrained_bundle(bundle_path)
