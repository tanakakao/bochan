# Material model architecture Phase 4

## Scope

Phase 4 establishes the canonical structure-model namespace for material-aware Gaussian models without changing runtime model behavior.

Canonical package:

```text
bochan.models.regression.gaussian.materials.structure
├── alignn.py
├── chgnet.py
├── m3gnet.py
├── mace.py
└── common.py
```

The following model families are exposed from the new namespace:

- ALIGNN
- CHGNet
- M3GNet
- MACE

For each family, the existing GP, DKL, mixed, and correlated multitask classes already implemented under `gaussian.deep` remain the implementation objects.

## Compatibility-first migration

Phase 4 intentionally does not move class definitions physically out of `gaussian.deep` yet.

The canonical structure namespace re-exports the existing classes so that:

- old and new imports resolve to the exact same class objects;
- `__module__` remains the historical `gaussian.deep.*` path;
- pickle/model-save compatibility is preserved;
- relative imports inside the current implementations remain unchanged;
- existing structure-feature cache logic remains unchanged;
- no optional backend dependency loading policy is changed.

This mirrors the Phase 3 composition migration strategy.

## Canonical imports

Examples:

```python
from bochan.models.regression.gaussian.materials.structure import (
    ALIGNNGPModel,
    CHGNetGPModel,
    M3GNetGPModel,
    MACEGPModel,
)
```

Family-specific imports are also supported:

```python
from bochan.models.regression.gaussian.materials.structure.mace import MACEGPModel
```

Historical imports remain valid:

```python
from bochan.models.regression.gaussian.deep.mace import MACEGPModel
```

The canonical and historical imports return the same class object.

## Structure-common infrastructure

The existing shared structure implementation in `gaussian.deep.structure` owns:

- structure-bank validation;
- discrete structure-index validation;
- process-only normalization;
- frozen representation caching;
- cache invalidation on encoder policy changes;
- state-dict/pickle cache cleanup;
- unique-structure batching;
- material/process fusion through the shared material feature extractor.

Phase 4 exposes this implementation through:

```text
bochan.models.regression.gaussian.materials.structure.common
```

as exact aliases. The implementation is not duplicated, so cache and serialization behavior cannot diverge between old and new import paths.

## Model coverage

### ALIGNN

- `ALIGNNGPModel`
- `ALIGNNDKLModel`
- `ALIGNNMixedGPModel`
- `ALIGNNMixedDKLModel`
- `ALIGNNMultiTaskGPModel`
- `ALIGNNMultiTaskDKLModel`
- `ALIGNNMixedMultiTaskGPModel`
- `ALIGNNMixedMultiTaskDKLModel`

### CHGNet

- `CHGNetGPModel`
- `CHGNetDKLModel`
- `CHGNetMixedGPModel`
- `CHGNetMixedDKLModel`
- `CHGNetMultiTaskGPModel`
- `CHGNetMultiTaskDKLModel`
- `CHGNetMixedMultiTaskGPModel`
- `CHGNetMixedMultiTaskDKLModel`

### M3GNet

- `M3GNetGPModel`
- `M3GNetDKLModel`
- `M3GNetMixedGPModel`
- `M3GNetMixedDKLModel`
- `M3GNetMultiTaskGPModel`
- `M3GNetMultiTaskDKLModel`
- `M3GNetMixedMultiTaskGPModel`
- `M3GNetMixedMultiTaskDKLModel`

### MACE

- `MACEGPModel`
- `MACEDKLModel`
- `MACEMixedGPModel`
- `MACEMixedDKLModel`
- `MACEMultiTaskGPModel`
- `MACEMultiTaskDKLModel`
- `MACEMixedMultiTaskGPModel`
- `MACEMixedMultiTaskDKLModel`

## Non-goals

Phase 4 does not change:

- GP or DKL mathematics;
- encoder loading or fine-tuning behavior;
- structure representation caching;
- mixed-process semantics;
- correlated multitask semantics;
- `train_Yvar` handling;
- partial-target handling;
- failed/pending observation semantics;
- acquisition functions;
- FastAPI or Web schemas;
- public `model_type` strings.

## Tests

Phase 4 adds contract tests that verify:

1. every canonical structure model import is identical to its historical `gaussian.deep` class;
2. canonical classes retain historical `__module__` values for pickle compatibility;
3. the canonical structure-common aliases are the exact existing cache/validation implementation objects.

## Next phase

Phase 5 should normalize the encoder training-policy layer across composition and structure families.

The preferred direction is to centralize only the generic lifecycle:

```text
frozen / partial / full
```

while leaving backend-specific layer discovery in each family. This reduces duplication without hiding model-specific fine-tuning semantics.
