# Material models architecture Phase 9

Phase 9 introduces a backend-neutral capability layer for pretrained material
models.  The goal is to let Bochan reason about what a family can do before the
optional third-party backend is imported or loaded.

## Canonical capability contract

`bochan.models.regression.gaussian.materials.common.pretrained` defines:

- `PretrainedMaterialCapabilities`
- `PretrainedMaterialSpec`
- `MaterialDomain`
- `PretrainedLoadingMode`
- `resolve_pretrained_loading_mode`

The contract distinguishes:

1. fixed-width representation extraction for GP/DKL,
2. direct pretrained property prediction,
3. supported loading routes (`checkpoint`, `model_name`, `injected`),
4. device and dtype support,
5. fine-tuning suitability,
6. residual-GP suitability.

Residual-GP capability requires direct pretrained prediction.  DKL capability
requires both representation extraction and declared fine-tuning support.

## Why capability metadata is separate from adapters

MACE, CHGNet, M3GNet, ALIGNN, CrabNet, and Roost have different third-party
loading APIs and optional dependencies.  The neutral capability layer therefore
does not import any of those libraries and does not load checkpoints itself.
Family adapters remain responsible for concrete loading and prediction.

This also keeps `gaussian.materials` import-safe when optional material packages
are not installed.

## Loading resolution

`resolve_pretrained_loading_mode` validates that exactly one supported route is
selected.  A family with a declared default model name may resolve to
`model_name` without the caller repeating that name.

The helper performs routing validation only; it does not instantiate models.

## Compatibility guarantees

Phase 9 does not change:

- existing material encoder classes,
- pretrained checkpoint/model-name behavior,
- public model classes or `model_type` strings,
- GP/DKL mathematics,
- frozen/partial/full training semantics,
- structure caches,
- mixed categorical behavior,
- multitask/partial-target/known-noise behavior,
- FastAPI/Web schemas,
- serialized model module paths.

## Follow-up

Phase 10 can consume this contract to implement a generic residual-GP layer:

`pretrained direct prediction + GP residual correction`.

Phase 11 can then attach concrete capability metadata to material-family registry
entries without scattering capability checks through factory `if/elif` logic.
