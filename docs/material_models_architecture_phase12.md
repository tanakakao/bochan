# Material model architecture — Phase 12

## Goal

Protect historical import and serialization boundaries while the canonical
material-model API lives under `bochan.models.regression.gaussian.materials`.

## Current serialization boundary

Concrete material model classes still originate from modules under
`bochan.models.regression.gaussian.deep`.  The canonical composition and
structure namespaces re-export those exact class objects.  Consequently a
class such as `MACEGPModel` is importable from the canonical namespace while
its `__module__` intentionally remains the historical `gaussian.deep.mace`
path.

That behavior is currently required for compatibility with Python pickle and
other model-save formats that persist the class module path.

## Phase 12 compatibility metadata

`materials.common.compatibility` records historical-to-canonical mappings with
`MaterialCompatibilityPath` and `LEGACY_MATERIAL_MODEL_PATHS`.

The mapping is metadata-only and does not import concrete model modules.
Therefore importing the common materials package still does not eagerly load
optional MACE, CHGNet, M3GNet, ALIGNN, CrabNet, or Roost backends.

## Protected behavior

For every protected path in this phase:

1. the historical `gaussian.deep.*` import remains valid;
2. the canonical `gaussian.materials.*` import remains valid;
3. both paths resolve to the exact same Python class object;
4. the class `__module__` remains the historical implementation module;
5. pickling and unpickling the class resolves through the historical path.

The contract tests intentionally test class objects rather than constructing
backend-specific models so optional material dependencies are not required.

## Why the old modules are not converted to shims yet

Turning the historical implementation files into re-export shims would first
require moving the actual class definitions to the canonical package.  That
would change `__module__` for newly serialized models unless custom pickle
migration machinery were introduced.  It would also increase the risk of
relative-import cycles during the staged migration.

Phase 12 therefore treats the old modules as serialization-protected
implementation locations rather than deprecated shims.

## Deprecation policy

No runtime deprecation warning is emitted for the protected paths in this
phase.  A warning would suggest that users should stop relying on a path that
Bochan still needs to load historical saved models.

Deprecation can begin only after all of the following are true:

- concrete implementations have a stable canonical ownership location;
- save/load compatibility is tested for existing persisted models;
- a migration or custom unpickling strategy exists for historical module
  references;
- internal relative imports no longer depend on the historical layout.

## Scope

Phase 12 does not change:

- public model classes or `model_type` strings;
- GP/DKL mathematics;
- pretrained loading behavior;
- encoder training policy;
- mixed or multitask behavior;
- partial observations or `train_Yvar` semantics;
- structure feature caches;
- FastAPI/Web schemas or acquisition behavior.

## Next phase

Phase 13 consolidates architecture contract tests across package imports,
encoder lifecycle, process fusion, multitask behavior, surrogate construction,
pretrained capabilities, residual GP, registry metadata, and serialization
compatibility.
