# ALIGNN crystal-structure input

Phase 2 adds the crystal-structure boundary that converts user-facing structures into the graph bank consumed by `ALIGNNGPModel` and `ALIGNNDKLModel`.

## Supported Python inputs

`StructureAdapter.adapt()` accepts in-memory structures:

- `jarvis.core.atoms.Atoms`
- mappings with `lattice_mat`, `coords`, `elements`, and optional `cartesian`
- periodic `ase.Atoms`
- ordered `pymatgen.core.Structure`

Local filesystem access is intentionally separate:

```python
from bochan.structure import StructureAdapter

adapter = StructureAdapter()
atoms = adapter.from_file("sample.cif")
atoms = adapter.from_file("POSCAR")
```

Keeping paths out of `adapt()` prevents later API/Web layers from accidentally treating arbitrary user strings as server-side paths.

Disordered pymatgen structures and ASE structures that are not periodic in all three directions are rejected because the current ALIGNN crystal-graph contract assumes a resolved periodic crystal.

## Building the ALIGNN graph bank

```python
from bochan.structure import ALIGNNGraphBuilder

builder = ALIGNNGraphBuilder()
structure_graphs = builder.build_many(structures)
```

The upstream `Graph.atom_dgl_multigraph` graph/line-graph result is kept as the low-level Bochan structure-bank entry. Phase-1 `ALIGNNEncoder` executes the upstream representation backbone directly and stops at pooled `readout`, so it does not call the scalar property head.

The graph bank is passed directly to the model:

```python
model = ALIGNNGPModel(
    train_X=train_X,
    train_Y=train_Y,
    structure_graphs=structure_graphs,
    checkpoint=checkpoint,
)
```

`train_X[:, 0]` remains the discrete structure index. Remaining columns are continuous process variables. Structure selection must therefore be enumerated/fixed during acquisition optimization rather than relaxed into a continuous coordinate.

## Graph defaults

The default `ALIGNNGraphBuilder` follows the current upstream crystal-training configuration:

- neighbor strategy: `k-nearest`
- cutoff: `5.0 Å`
- cutoff extension: `3.0 Å`
- maximum neighbors: `12`
- atom features: `cgcnn`
- canonicalized edges: enabled
- graph dtype: `float32`
- three-body cutoff: `3.5 Å`

For transferred pretrained models, graph construction should follow the original training metadata rather than generic defaults.

## Local pretrained bundles

`load_alignn_pretrained_bundle()` loads either an extracted training directory or a local upstream pretrained ZIP containing `config.json` plus `best_model.pt` or `checkpoint_*.pt`.

```python
from bochan.structure import load_alignn_pretrained_bundle

bundle = load_alignn_pretrained_bundle("alignn_model.zip")
encoder = bundle.build_encoder()
builder = bundle.build_graph_builder()
```

The bundle keeps model configuration, checkpoint weights, and graph-construction settings together. `best_model.pt` is preferred; otherwise the numerically largest `checkpoint_<N>.pt` is selected. Checkpoints are loaded with `torch.load(..., weights_only=True)`.

## Optional dependency policy

Bochan keeps ALIGNN/JARVIS/DGL imports lazy. The core package therefore remains importable without the atomistic stack.

The Phase-2 CI validates against `alignn==2026.8.11`. ALIGNN does not install DGL as a hard dependency, and DGL compatibility depends on the local PyTorch/CUDA platform, so this phase intentionally does not pin a universal DGL build into Bochan's runtime dependencies.
