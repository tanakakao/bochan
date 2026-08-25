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


def test_structure_mapping_validation_is_strict() -> None:
    adapter = StructureAdapter()
    invalid = _si_structure()
    invalid["coords"] = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="same number of atoms"):
        adapter.adapt(invalid)


def test_alignn_graph_builder_preserves_upstream_graph_settings(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeGraph:
        @staticmethod
        def atom_dgl_multigraph(**kwargs: object) -> tuple[str, str]:
            calls.append(dict(kwargs))
            return "graph", "line_graph"

    monkeypatch.setattr(alignn_structure, "_alignn_graph_class", lambda: FakeGraph)
    builder = ALIGNNGraphBuilder(
        neighbor_strategy="k-nearest",
        cutoff=7.5,
        max_neighbors=10,
        atom_features="cgcnn",
        compute_line_graph=True,
        use_canonize=False,
    )

    result = builder.build(_si_structure())

    assert result == ("graph", "line_graph")
    assert len(calls) == 1
    call = calls[0]
    assert call["neighbor_strategy"] == "k-nearest"
    assert call["cutoff"] == 7.5
    assert call["max_neighbors"] == 10
    assert call["atom_features"] == "cgcnn"
    assert call["compute_line_graph"] is True
    assert call["use_canonize"] is False
    assert call["atoms"].elements == ["Si", "Si"]  # type: ignore[union-attr]


def test_alignn_graph_builder_uses_training_config(monkeypatch) -> None:
    class FakeGraph:
        @staticmethod
        def atom_dgl_multigraph(**kwargs: object) -> tuple[str, str]:
            return "graph", "line_graph"

    monkeypatch.setattr(alignn_structure, "_alignn_graph_class", lambda: FakeGraph)
    builder = ALIGNNGraphBuilder.from_training_config(
        {
            "neighbor_strategy": "radius_graph_jarvis",
            "cutoff": 5.5,
            "max_neighbors": 8,
            "atom_features": "atomic_number",
            "compute_line_graph": True,
            "use_canonize": True,
        }
    )

    graphs = builder.build_many([_si_structure(), _si_structure()])

    assert graphs == (("graph", "line_graph"), ("graph", "line_graph"))
    assert builder.config == {
        "neighbor_strategy": "radius_graph_jarvis",
        "cutoff": 5.5,
        "max_neighbors": 8,
        "atom_features": "atomic_number",
        "compute_line_graph": True,
        "use_canonize": True,
    }


def test_local_pretrained_zip_keeps_model_and_graph_configs_together(tmp_path, monkeypatch) -> None:
    config = {
        "neighbor_strategy": "k-nearest",
        "cutoff": 8.0,
        "max_neighbors": 12,
        "atom_features": "cgcnn",
        "compute_line_graph": True,
        "use_canonize": True,
        "model": {
            "name": "alignn",
            "hidden_features": 4,
            "output_features": 1,
        },
    }
    checkpoint_buffer = BytesIO()
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint_buffer)

    bundle_path = tmp_path / "alignn_model.zip"
    with ZipFile(bundle_path, "w") as archive:
        archive.writestr("run/config.json", dumps(config))
        archive.writestr("run/best_model.pt", checkpoint_buffer.getvalue())

    bundle = load_alignn_pretrained_bundle(bundle_path)
    graph_builder = bundle.build_graph_builder()

    assert bundle.checkpoint_name == "run/best_model.pt"
    assert bundle.model_config["name"] == "alignn"
    assert graph_builder.cutoff == 8.0
    assert graph_builder.max_neighbors == 12
    assert graph_builder.neighbor_strategy == "k-nearest"

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
    config = {"model": {"name": "alignn"}}
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


def test_pretrained_bundle_rejects_non_scalar_alignn(tmp_path) -> None:
    config = {"model": {"name": "alignn_atomwise"}}
    checkpoint_buffer = BytesIO()
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint_buffer)
    bundle_path = tmp_path / "atomwise.zip"
    with ZipFile(bundle_path, "w") as archive:
        archive.writestr("config.json", dumps(config))
        archive.writestr("best_model.pt", checkpoint_buffer.getvalue())

    with pytest.raises(NotImplementedError, match="model.name='alignn'"):
        load_alignn_pretrained_bundle(bundle_path)
