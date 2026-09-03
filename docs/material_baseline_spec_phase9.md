# Material Baseline Specification

Residual GP models subtract a deterministic pretrained prediction from an observed target. That subtraction is only physically meaningful when both values represent the same quantity, unit, and aggregation convention.

## Core contracts

`MaterialPropertyContract` records:

- `quantity`: stable property identifier such as `energy` or `band_gap`
- `unit`: exact unit string such as `eV`, `eV/atom`, or `GPa`
- `aggregation`: `total`, `per_atom`, `intensive`, or `unspecified`

`MaterialBaselineSpec` records:

- pretrained `family`
- physical-property contract
- optional `output_name` or `output_index`
- optional pretrained `model_name`
- whether the baseline is enabled

Only one output selector may be supplied. Neither selector is required for a scalar single-output residual model.

## Compatibility rules

Bochan does not silently convert physical quantities or units before residual subtraction.

- quantity comparison is case-insensitive
- unit comparison is exact and case-sensitive
- different explicit aggregation conventions are rejected
- `unspecified` aggregation does not imply a conversion; it only means the aggregation convention is not asserted by that contract

For example, `total energy [eV]` and `energy [eV/atom]` are incompatible and must not be subtracted directly.

## Residual model integration

`ResidualMaterialGPModel` accepts an optional `baseline_spec`. Existing callers may omit it and retain the previous behavior.

When attached, the model:

- rejects disabled baseline specifications
- validates the baseline family against `PretrainedMaterialSpec` when both are available
- exposes JSON-compatible `baseline_metadata`
- exposes `validate_target_contract(...)`
- preserves the baseline specification when conditioning on new observations

`compute_material_residual_targets(...)` also accepts optional `baseline_spec` and `target_contract` arguments so physical compatibility can be checked immediately before subtraction.

## Scope

This phase defines the stable contract required for subsequent multiple-baseline routing. It does not yet assign different pretrained models to different outputs automatically and does not perform unit conversion.
