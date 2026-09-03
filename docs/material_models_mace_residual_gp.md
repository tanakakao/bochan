# MACE Residual GP

Bochan can use a pretrained MACE energy prediction as a deterministic baseline and fit an exact Gaussian process to the residual.

```text
observed energy = MACE pretrained energy + GP residual
```

## Input contract

- `X[..., 0]` is the integer index into the configured structure bank.
- Remaining columns are optional process variables.
- MACE baseline prediction depends only on structure.
- Process variables are retained by the GP and can therefore explain process-dependent deviations from the pretrained energy.

## Target contract

`train_Y` must represent the same energy definition and units as the selected MACE pretrained model and selected head. Bochan does not automatically convert reference energies, per-atom versus total energies, or physical units.

The residual target is:

```text
train_Y_residual = train_Y - MACE_energy
```

`train_Yvar` is unchanged because subtracting a deterministic baseline does not alter observation noise.

## Head selection

MACE models may expose multiple heads. Bochan reuses the existing `MACEEncoder` head-resolution rules. A unique default head may be selected automatically; otherwise `head=` must be supplied explicitly. If an already constructed `MACEEncoder` is supplied together with a conflicting `head`, construction fails.

## Model reuse

One frozen `MACEEncoder` instance is reused for both:

1. direct pretrained energy prediction;
2. invariant representation extraction for the residual GP.

The pretrained model is therefore not loaded twice.

## Registry

After this change MACE advertises:

- `direct_prediction=True`
- `residual_gp=True`
- `residual_gp` model variant

CHGNet, M3GNet, and MACE now share the same high-level residual-GP pattern.
