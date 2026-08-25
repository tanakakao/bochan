# ALIGNN crystal-structure input

Phase 2 adds the crystal-structure boundary that converts user-facing structures into the graph bank consumed by `ALIGNNGPModel` and `ALIGNNDKLModel`.

## Supported Python inputs

`StructureAdapter` accepts:

- `jarvis.core.atoms.Atoms`
- mappings with `lattice_mat`, `coords`, `elements`, and optional `cartesian` / `props`
- periodic `ase.Atoms`
- ordered `pymatgen.core.Structure`
- trusted local `.cif`, `.vasp`, `.poscar`, `POSCAR`, or `CONTCAR` files

Disordered pymatgen structures and non-periodic ASE cells are rejected explicitly because the current ALIGNN crystal-graph contract assumes a resolved periodic crystal structure.

## Building the ALIGNN graph bank

```python
from bochan.structure import ALIGNNGraphBuilder

builder = ALIGNNGraphBuilder()
structure_graphs = builder.build_many(structures)
```

Each structure is normalized to:

```text
(graph, line_graph, lattice_tensor)
```

This deliberately follows the current upstream scalar ALIGNN forward contract while remaining compatible with Bochan's Phase-1 `ALIGNNEncoder`, which consumes the graph and line graph to expose the pooled structure representation before the property head.

The resulting bank is passed directly to the GP/DKL model:

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

`ALIGNNGraphConfig` follows the current upstream crystal-training defaults:

- neighbor strategy: `k-nearest`
- cutoff: `5.0 Å`
- maximum neighbors: `12`
- atom features: `cgcnn`
- canonicalized edges: enabled
- graph dtype: `float32`
- three-body cutoff: `3.5 Å`

These values should match the graph construction used for the checkpoint being transferred. A pretrained ALIGNN checkpoint is not graph-construction agnostic.

## Optional dependencies

Bochan keeps ALIGNN/JARVIS/DGL imports lazy. The core Bochan package therefore remains importable without the atomistic stack.

The current upstream `alignn` package does not install DGL as a hard dependency even though its standard scalar `ALIGNN` model uses the DGL backend. For that reason Bochan does not guess a universal DGL pin in this phase. Install the ALIGNN/JARVIS stack and a DGL build compatible with the local PyTorch platform before using the default graph builder and upstream encoder.

A later packaging phase can add a tested platform-specific dependency matrix once the supported PyTorch/DGL combinations are fixed for Bochan's deployment environments.
