# Multi-output Residual GP — Phase 5

This phase extends the CHGNet, M3GNet, and MACE residual-GP path from one target to correlated wide multi-output targets.

## Model semantics

For `train_Y` with shape `[n, m]`, the pretrained structure model currently contributes one scalar physical property. `baseline_output_index` selects the target column representing that property.

For example, with three targets and `baseline_output_index=1`:

```text
pretrained baseline = [0, pretrained_energy, 0]
residual target      = train_Y - pretrained baseline
```

The residual model is the existing correlated family multitask GP using a shared material representation and `MultitaskKernel`. Therefore all target columns are learned jointly; only the physically compatible target column is shifted by the pretrained baseline.

At prediction time:

```text
corrected mean = routed pretrained baseline + correlated residual-GP mean
corrected variance = correlated residual-GP variance
```

## Added models

- `CHGNetMultiTaskResidualGPModel`
- `M3GNetMultiTaskResidualGPModel`
- `MACEMultiTaskResidualGPModel`

Registry variant: `multitask_residual_gp`.

## Baseline routing

`RoutedDirectMaterialPredictor` is backend-neutral and routes an existing scalar `DirectMaterialPredictor` to one output column. Other columns receive deterministic zero baseline, which means those targets are modeled directly by the correlated GP.

Negative `baseline_output_index` values are accepted using standard Python indexing semantics.

## Target contract

The selected baseline output column must use the same physical quantity, definition, and units as the pretrained direct predictor:

- CHGNet: selected energy output contract from the configured pretrained CHGNet model.
- M3GNet: selected scalar output contract from the configured pretrained M3GNet model.
- MACE: selected-head energy output contract from the configured MACE model.

Bochan does not automatically convert units, total/per-atom conventions, reference energies, or unrelated property definitions.

## Observation behavior

`train_Yvar` is passed unchanged to the established correlated multitask Gaussian backend. NaN partial observations are preserved during residual subtraction so the existing observation-aware path remains responsible for missing-target semantics.

## Compatibility

No historical `gaussian.deep.*` class is moved or renamed. Existing GP, DKL, mixed, multitask, single-output residual, and mixed residual models keep their current public behavior and saved-model paths.

## Not included yet

This phase does not yet combine categorical mixed inputs with correlated multi-output residuals. The next step is `mixed_multitask_residual_gp`, which should compose this routed baseline with the already implemented mixed residual input semantics.
