# Material model architecture Phase 3

## Scope

Phase 3 establishes canonical composition-family import paths for CrabNet and Roost under:

```text
bochan.models.regression.gaussian.materials.composition
```

The historical implementations under `gaussian.deep` remain the runtime source during this compatibility step. The new canonical namespace re-exports the exact same class objects, so existing imports, serialized class identity, factories, and model behavior remain unchanged.

## Canonical families

```text
gaussian/materials/composition/
├── __init__.py
├── crabnet/
│   └── __init__.py
└── roost/
    └── __init__.py
```

Published CrabNet classes:

- `CrabNetGPModel`
- `CrabNetDKLModel`
- `CrabNetMixedGPModel`
- `CrabNetMixedDKLModel`
- `CrabNetMultiTaskGPModel`
- `CrabNetMultiTaskDKLModel`
- `CrabNetMixedMultiTaskGPModel`
- `CrabNetMixedMultiTaskDKLModel`

Published Roost classes:

- `RoostGPModel`
- `RoostDKLModel`

## Compatibility strategy

Phase 3 intentionally establishes the canonical namespace before physically moving implementation bodies. This ordering protects:

- direct legacy imports from `gaussian.deep`;
- pickle/module-path compatibility;
- existing internal relative imports among CrabNet wrappers;
- model factories and `model_type` strings;
- GP/DKL, mixed and multitask behavior;
- known `train_Yvar` and partial-observation behavior.

The contract tests assert object identity between old and new import paths. A later relocation step may invert the dependency so the legacy modules become shims only after serialization and relative-import boundaries are safe to change.

## Behavior changes

None. This phase changes import topology only.

## Next phase

Phase 4 introduces canonical structure-family namespaces for ALIGNN, CHGNet, M3GNet and MACE, including the shared structure feature infrastructure, while preserving legacy imports and structure-cache behavior.
