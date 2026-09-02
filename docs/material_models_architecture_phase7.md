# Material model architecture — Phase 7

Phase 7 defines a neutral multitask contract for material-aware Gaussian surrogates without changing existing model behavior.

## Scope

Material models currently expose two distinct multi-output semantics:

1. **Correlated multitask**: one shared material encoder / latent representation and a `MultitaskKernel` that learns cross-property covariance.
2. **Independent output**: one surrogate per output, typically composed as a `ModelListGP`-style path.

These semantics should remain explicit rather than being hidden behind one ambiguous `multitask=True` switch.

## New common contract

`gaussian.materials.common.multitask` adds:

- `MaterialTaskMode = Literal["correlated", "independent"]`
- `MaterialMultiTaskSpec`
- `validate_wide_material_targets`
- `validate_correlated_task_kernel`
- `task_covar_module`

The contract describes output semantics and removes the need for every material family to redefine the same wide-target and correlated-kernel checks.

## Observation-aware behavior

The common validator intentionally checks shapes only. It does **not** reinterpret target values, remove NaNs, fill missing values, or alter `train_Yvar` values.

This is deliberate: partial observations and known observation variance are already handled by the established Gaussian construction / observation-aware paths. Phase 7 must not create a second masking implementation with subtly different semantics.

Required invariants:

- `train_Y` stays wide as `[n, m]` at the material multitask boundary.
- optional `train_Yvar` must match the wide `train_Y` shape.
- missing/partial observations remain represented exactly as expected by downstream observation-aware code.
- correlated models keep one shared encoder and `MultitaskKernel`.
- independent-output models keep separate output surrogates.
- public model names and `model_type` strings remain unchanged.

## Family migration

The following family-specific implementations can migrate incrementally to this common contract:

- CrabNet correlated GP / DKL / mixed multitask
- ALIGNN correlated GP / DKL / mixed multitask
- CHGNet correlated GP / DKL / mixed multitask
- M3GNet correlated GP / DKL / mixed multitask
- MACE correlated GP / DKL / mixed multitask

Backend-specific material feature extraction, structure banks, encoder fine-tuning, mixed-process layouts, and cache behavior remain outside this common multitask module.

## Compatibility

Phase 7 introduces no changes to:

- GP or DKL mathematics;
- task covariance parameterization;
- posterior output ordering;
- partial observation semantics;
- known-noise `train_Yvar` semantics;
- material encoder sharing;
- mixed categorical-process handling;
- structure feature caches;
- API / Web schemas;
- acquisition behavior;
- saved-model module paths.

## Next phase

Phase 8 can normalize GP/DKL surrogate construction after package, encoder lifecycle, process-layout, and multitask boundaries are stable.
