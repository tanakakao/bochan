# Mixed residual GP support

This phase extends the verified CHGNet, M3GNet, and MACE residual-GP paths to categorical process inputs.

## Model semantics

The pretrained baseline remains structure-only:

```text
baseline = pretrained(structure)
residual = observed_target - baseline
```

The residual Gaussian process receives the complete mixed input. Feature 0 remains the structure-bank selector, `cat_dims` identifies categorical process columns, and remaining process columns are treated as numeric by the established material mixed-GP implementation.

This yields:

```text
corrected posterior = pretrained(structure) + GP_residual(
    structure representation,
    numeric process variables,
    categorical process variables,
)
```

## New public models

- `CHGNetMixedResidualGPModel`
- `M3GNetMixedResidualGPModel`
- `MACEMixedResidualGPModel`

The family registry exposes these as `mixed_residual_gp` for CHGNet, M3GNet, and MACE.

## Target contracts

The direct baseline contract does not change from the corresponding single-input residual models. CHGNet and MACE targets must match the selected pretrained energy definition and units. M3GNet targets must match its selected pretrained scalar property. No automatic unit, reference-energy, intensive/extensive, or per-atom/total conversion is performed.

## Categorical contract

`cat_dims` must contain categorical process columns only. Feature 0 is reserved for the structure selector and remains outside the categorical kernel. This validation is delegated to the established `*MixedGPModel` implementations so mixed semantics remain identical to non-residual material models.

## Compatibility

No historical `gaussian.deep.*` implementation is moved or renamed. Existing GP, DKL, mixed, multitask, and residual model classes remain unchanged. The new mixed residual classes live in the canonical `gaussian.materials.structure` namespace.

## Next step

Multi-output residual support should be implemented separately because the direct pretrained output contract must be defined per target. A later mixed-multitask residual variant can compose that target contract with the same categorical process semantics introduced here.
