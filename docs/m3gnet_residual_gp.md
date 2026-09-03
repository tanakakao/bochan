# M3GNet Residual GP

Bochan can use the scalar direct prediction of a pretrained MatGL M3GNet model as a deterministic baseline and fit an exact Gaussian process to the residual.

## Model

For structure/process input `x`,

- pretrained baseline: `f_m3gnet(structure)`
- residual target: `y - f_m3gnet(structure)`
- corrected posterior mean: `f_m3gnet(structure) + mean_gp(x)`
- corrected posterior variance: `var_gp(x)`

The baseline intentionally ignores process variables. The residual GP receives the full Bochan input, so process-dependent corrections can still be learned.

## Public classes

- `M3GNetDirectPredictor`
- `M3GNetResidualGPModel`

Both are exposed from `bochan.models.regression.gaussian.materials.structure`.

## Target contract

`train_Y` must represent the same scalar physical quantity and units returned by the selected pretrained M3GNet model. Bochan does not silently convert property definitions, intensive/extensive conventions, reference energies, or units before subtracting the baseline.

The current residual adapter requires exactly one scalar direct prediction per structure. A model returning multiple outputs is rejected until an explicit output-selection contract is added.

## Input convention

Column 0 of `train_X` is the integer index into the configured structure bank. Remaining columns are process variables. Repeated structure indices are deduplicated before direct prediction.

## Compatibility

The historical M3GNet GP/DKL classes under `gaussian.deep` are unchanged. `M3GNetResidualGPModel` is a new canonical class and therefore has no historical serialization path to preserve.

## Next step

MACE can use the same residual architecture, but its direct energy output and head selection must be made explicit before the registry advertises residual support.
