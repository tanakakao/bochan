# CHGNet residual GP — Phase 1

## Scope

This phase connects the generic material residual-GP infrastructure to the first concrete pretrained structure family: CHGNet energy prediction.

The corrected model is

```text
observed energy = CHGNet pretrained energy + GP residual
```

The pretrained baseline depends on crystal structure only. Optional process variables remain available to the GP residual, so the correction can learn systematic deviations associated with processing conditions while retaining CHGNet's structure-informed prior prediction.

## Public classes

- `CHGNetDirectEnergyPredictor`
- `CHGNetResidualGPModel`

Both are exposed from `bochan.models.regression.gaussian.materials.structure`.

## Input contract

`X[..., 0]` is the integer-valued index into the configured structure bank, matching the existing CHGNet GP/DKL convention. Remaining columns are process variables.

The direct CHGNet baseline intentionally ignores process columns. The residual GP receives the complete `X`, so process effects are represented only in the learned correction.

## Target contract

`train_Y` must describe the same CHGNet energy quantity and units returned by the selected pretrained model. Residual targets are computed as

```text
train_Y_residual = train_Y - CHGNet_energy
```

`train_Yvar` is not modified. Partial observations retain their existing NaN mask and continue through the established observation-aware Gaussian path.

## Shared backend

The residual model reuses one frozen `CHGNetEncoder` for both:

1. direct pretrained energy prediction;
2. CHGNet representation extraction inside the residual `CHGNetGPModel`.

No second pretrained checkpoint is loaded.

The final BoTorch posterior is supplied by `ResidualMaterialGPModel`:

```text
mean_final = CHGNet_energy + mean_residual_GP
variance_final = variance_residual_GP
```

This preserves compatibility with standard BoTorch acquisition functions.

## Registry

The `chgnet` family now advertises:

- `direct_prediction=True`
- `residual_gp=True`
- `residual_gp` as an implemented model variant

Other material families remain conservative until their direct-prediction adapters are implemented and verified.

## Compatibility

No historical `gaussian.deep.*` class is moved or renamed. Existing GP/DKL/mixed/multitask model types and saved-model module paths are unchanged. `CHGNetResidualGPModel` is a new canonical model and therefore does not require a historical compatibility alias.

## Follow-up

After validating this implementation, the same structure-indexed direct-predictor pattern can be extended to M3GNet and MACE. Their direct outputs and property/unit conventions must be made explicit before their registry capability flags are enabled.
