# ALIGNN crystal-structure input

Bochan's canonical ALIGNN path is pure PyTorch. Crystal structures are converted to upstream `TorchGraph` atom/line graphs and consumed by `ALIGNNGPModel` / `ALIGNNDKLModel` without DGL.

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

Keeping paths out of `adapt()` prevents API/Web layers from accidentally treating arbitrary user strings as server-side paths.

Disordered pymatgen structures and ASE structures that are not periodic in all three directions are rejected because the ALIGNN crystal-graph contract assumes a resolved periodic crystal.

## Building the pure-PyTorch ALIGNN graph bank

```python
from bochan.structure import ALIGNNGraphBuilder

builder = ALIGNNGraphBuilder()
structure_graphs = builder.build_many(structures)
```

Internally Bochan calls upstream `alignn.torch_graph_builder.build_pure_torch_graph()`.
Each structure-bank entry is:

```text
(TorchGraph atom_graph, TorchGraph line_graph)
```

The atom graph stores pair connectivity, periodic image offsets, atomic features, and displacement vectors. The line graph uses parent bonds as nodes and stores bond-angle cosines for three-body message passing.

`ALIGNNEncoder` uses upstream `ALIGNNAtomWisePure`, runs its ALIGNN + GCN representation backbone, mean-pools the final atom hidden states, and stops before the property head. No DGL graph or DGL convolution is created on this path.

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

The defaults follow the current upstream pure-PyTorch scalar-property recipe:

- neighbor strategy: `pure_torch`
- pair cutoff: `8.0 Å`
- maximum neighbors per source atom: `12`
- atom features: `cgcnn`
- graph dtype: `float32`
- three-body cutoff: `3.5 Å`
- line graph: enabled
- matscipy topology shortcut: disabled by Bochan

The pure graph builder performs periodic neighbor search and line-graph construction using torch tensor/index/scatter operations. For transferred pretrained models, graph settings must follow the original pure training metadata rather than generic defaults.

## Local pretrained bundles

`load_alignn_pretrained_bundle()` loads either an extracted training directory or a local upstream pretrained ZIP containing `config.json` plus `best_model.pt` or `checkpoint_*.pt`.

```python
from bochan.structure import load_alignn_pretrained_bundle

bundle = load_alignn_pretrained_bundle("alignn_model.zip")
encoder = bundle.build_encoder()
builder = bundle.build_graph_builder()
```

The bundle requires:

```text
model.name = "alignn_atomwise_pure"
neighbor_strategy = "pure_torch"
```

Legacy DGL `model.name="alignn"` bundles are rejected instead of being loaded into a different model/graph contract. The bundle keeps model configuration, checkpoint weights, and graph settings together. `best_model.pt` is preferred; otherwise the numerically largest `checkpoint_<N>.pt` is selected. Checkpoints are loaded with `torch.load(..., weights_only=True)`.

## Dependency policy

Bochan keeps ALIGNN/JARVIS imports lazy, so the core package remains importable without the atomistic stack.

The CI validates against `alignn==2026.8.11`. The Bochan ALIGNN-GP/DKL path does **not** require DGL. A compatible ALIGNN installation plus Bochan's normal PyTorch stack is sufficient for graph construction and encoder execution.
