# Material Models Architecture — Phase 11

## Goal

Phase 11 introduces one lazy registry for the material-aware Gaussian model
families. The registry is metadata-first: importing it must not import optional
material backends.

## Registered families

| Family | Domain | Current Bochan variants |
| --- | --- | --- |
| CrabNet | composition | GP, DKL, mixed GP/DKL, multitask GP/DKL, mixed multitask GP/DKL |
| Roost | composition | GP, DKL |
| ALIGNN | structure | GP, DKL, mixed GP/DKL, multitask GP/DKL, mixed multitask GP/DKL |
| CHGNet | structure | GP, DKL, mixed GP/DKL, multitask GP/DKL, mixed multitask GP/DKL |
| M3GNet | structure | GP, DKL, mixed GP/DKL, multitask GP/DKL, mixed multitask GP/DKL |
| MACE | structure | GP, DKL, mixed GP/DKL, multitask GP/DKL, mixed multitask GP/DKL |

The registry stores canonical model paths such as
`bochan.models.regression.gaussian.materials.structure:MACEGPModel` and resolves
them only when requested.

## Pretrained capability policy

Phase 9 introduced the capability vocabulary. Phase 11 attaches conservative
family metadata to registry entries.

All six current adapters expose representations and have GP/DKL wrappers, so
`representation=True` and `fine_tuning=True` are registered. Direct pretrained
prediction is intentionally left `False` for every family at this stage because
the current Bochan material adapters used by Gaussian models expose
representations, not one neutral `DirectMaterialPredictor` contract.

Consequently `residual_gp=False` is also retained for every family. Phase 10's
Residual GP layer is ready, but no family is advertised as supported until a
concrete direct-prediction adapter is verified and connected.

## Loading metadata

The registry records the loading routes already represented by the current
adapters:

- CrabNet / Roost / ALIGNN: checkpoint or injected backend.
- CHGNet: checkpoint, model name, or injected backend; default model `0.3.0`.
- M3GNet: model name or injected backend; default model
  `M3GNet-PES-MatPES-PBE-2025.2`.
- MACE: model name or injected backend; default model `medium-mpa-0`.

This metadata does not load the model.

## Public API

```python
from bochan.models.regression.gaussian.materials import (
    MATERIAL_FAMILY_REGISTRY,
    get_material_family,
    list_material_families,
)

mace = get_material_family("mace")
assert mace.domain == "structure"
assert mace.supports("mixed_multitask_dkl")
MACEGPModel = mace.resolve_model_class("gp")
```

`resolve_model_class()` is the explicit lazy-import boundary.

## Compatibility guarantees

Phase 11 does not rename public `model_type` strings or concrete model classes.
Historical `gaussian.deep` module paths remain untouched for pickle/model-save
compatibility. Existing FastAPI/Web schemas, acquisition logic, mixed-input
semantics, multitask behavior, partial observations, known `train_Yvar`,
structure caches, and encoder loading behavior are unchanged.

## Next phase

Phase 12 can turn historical paths into explicit compatibility/deprecation shims
only after downstream imports and serialization boundaries are audited. The
registry is now the preferred internal metadata source for future factory
simplification.
