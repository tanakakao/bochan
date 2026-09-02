# Material models architecture Phase 2

## Goal

Phase 2 introduces the canonical material-aware Gaussian package skeleton without relocating any concrete CrabNet, Roost, ALIGNN, CHGNet, M3GNet, or MACE Gaussian wrappers.

The new package is:

```text
src/bochan/models/regression/gaussian/materials/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── base.py
│   └── fusion.py
├── composition/
│   └── __init__.py
└── structure/
    └── __init__.py
```

## Canonical common contracts

The neutral `MaterialEncoder` contract now lives at:

```python
bochan.models.regression.gaussian.materials.common.MaterialEncoder
```

Material/process fusion contracts now live at:

```python
bochan.models.regression.gaussian.materials.common.MaterialProcessFusion
bochan.models.regression.gaussian.materials.common.ConcatFusion
bochan.models.regression.gaussian.materials.common.build_material_process_fusion
```

These modules depend only on PyTorch and do not import concrete material-model families.

## Compatibility

The historical paths remain valid:

```python
bochan.composition.encoders.base.MaterialEncoder
bochan.composition.encoders.fusion.MaterialProcessFusion
bochan.composition.encoders.fusion.ConcatFusion
bochan.composition.encoders.fusion.build_material_process_fusion
```

Those modules are now compatibility shims that re-export the canonical objects. Therefore existing encoders continue to subclass the exact same `MaterialEncoder` class object, and existing `isinstance` checks and fusion type checks remain valid.

No public model class, model type, FastAPI/Web schema, optimizer call, acquisition behavior, partial-target behavior, known `train_Yvar` handling, or failed/pending observation behavior changes in this phase.

## Optional-dependency boundary

A key Phase 2 requirement is that importing:

```python
import bochan.models.regression.gaussian.materials
```

must not import `bochan.composition` or concrete CrabNet/Roost/ALIGNN/CHGNet/M3GNet/MACE encoder modules. This keeps the new common package independent of optional material backends and establishes the dependency direction required for later relocation.

Concrete model namespaces are intentionally empty in Phase 2:

```python
bochan.models.regression.gaussian.materials.composition
bochan.models.regression.gaussian.materials.structure
```

Phase 3 and Phase 4 populate them while preserving old `gaussian.deep` import paths.

## Tests

`tests/test_material_models_architecture_phase2.py` verifies:

- canonical common exports;
- legacy-path object identity;
- importability of composition and structure namespaces;
- canonical material imports do not load concrete material encoders.

## Out of scope

Phase 2 does not:

- move concrete Gaussian wrappers from `gaussian.deep`;
- move CrabNet/Roost concrete encoder implementations;
- move ALIGNN/CHGNet/M3GNet/MACE concrete encoder implementations;
- move `deep/material.py` feature extraction yet;
- move `deep/structure.py` structure caching yet;
- normalize GP/DKL, mixed, multitask, pretrained, or residual implementations.

Those changes remain sequenced after package/import stability.

## Next phase

Phase 3 relocates the composition Gaussian wrappers for CrabNet and Roost under `gaussian.materials.composition`, while retaining `gaussian.deep` compatibility modules and preserving existing `bochan.composition` encoder imports.
