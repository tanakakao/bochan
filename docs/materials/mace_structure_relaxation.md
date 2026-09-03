# MACE structure relaxation

`MACEStructureRelaxer` performs fast periodic crystal relaxation with a pretrained MACE potential and ASE.

```python
from bochan.models.regression.gaussian.materials.structure import MACEStructureRelaxer

relaxer = MACEStructureRelaxer(
    model_name="medium-mpa-0",
    device="cpu",
)
result = relaxer.relax(
    structure,
    optimizer="FIRE",
    fmax=0.05,
    max_steps=200,
    relax_cell=False,
)

print(result.converged)
print(result.energy)
print(result.max_force)
print(result.structure)
```

The input may be any in-memory periodic structure accepted by `StructureAdapter.to_ase`, including bochan's `lattice_mat` / `coords` / `elements` mapping, ASE `Atoms`, pymatgen `Structure`, or JARVIS `Atoms`.

## Relaxation modes

- `relax_cell=False`: relax atomic positions while keeping the lattice fixed.
- `relax_cell=True`: relax atomic positions and lattice degrees of freedom with ASE `FrechetCellFilter`.

Supported ASE optimizers are `FIRE`, `BFGS`, and `LBFGS`. `FIRE` is the default.

## Result contract

`StructureRelaxationResult` stores:

- relaxed structure
- initial and final potential energy
- final per-atom forces
- final full 3x3 stress tensor
- maximum per-atom force norm
- optimizer step count
- convergence flag
- optimizer and threshold metadata
- backend and pretrained model name

`result.as_dict()` returns a JSON-compatible representation.

Bochan does not perform implicit energy, force, or stress unit conversion and does not change the MACE stress sign convention.

## Current scope

This phase uses the pretrained MACE potential directly for relaxation. The residual GP is not differentiated with respect to atomic positions, so `MACE + GP residual` differentiable structure relaxation is intentionally out of scope.
