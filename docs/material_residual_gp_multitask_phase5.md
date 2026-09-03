# Material Residual GP Phase 5: correlated multi-output support

Phase 5 extends CHGNet, M3GNet, and MACE residual Gaussian processes to wide correlated targets and to mixed categorical process inputs.

## Supported variants

For each of CHGNet, M3GNet, and MACE, the canonical material namespace now exposes:

- `*ResidualGPModel`
- `*MixedResidualGPModel`
- `*MultiTaskResidualGPModel`
- `*MixedMultiTaskResidualGPModel`

The multitask variants reuse the existing correlated `MultitaskKernel` material models. They do not introduce an independent-output `ModelList` interpretation.

## Baseline mapping

Current pretrained adapters expose one scalar direct property: CHGNet energy, the selected M3GNet scalar property, or MACE energy for the selected head. A wide target can contain additional measured properties that have no pretrained baseline.

`pretrained_output_index` selects which target column receives the pretrained baseline. The remaining columns receive a deterministic zero baseline and are therefore learned directly by the correlated residual GP.

For example, with

```text
train_Y = [energy, strength, conductivity]
pretrained_output_index = 0
```

the residual targets are

```text
[energy - pretrained_energy, strength, conductivity]
```

and the existing multitask covariance learns correlations among all three outputs.

Negative output indices are accepted using normal Python indexing semantics.

## Mixed multitask semantics

The mixed multitask variants preserve the existing input contract:

- feature 0 is the structure selector;
- `cat_dims` contains categorical process columns only;
- remaining process columns are numeric;
- the pretrained baseline depends on structure only;
- numeric and categorical process effects are learned by the residual GP.

## Observation semantics

The residual mapping performs only deterministic subtraction. It does not mask, fill, or remove missing targets. Existing partial-observation and known-`train_Yvar` behavior therefore remains owned by the established Gaussian observation path.

## Physical-property contract

The target selected by `pretrained_output_index` must use the same physical quantity, reference convention, and units as the selected pretrained backend output. Bochan does not automatically convert total/per-atom energy, reference energies, property definitions, or units.

Other target columns do not require pretrained equivalents.

## Scope

This phase implements correlated multi-output residual models. It does not yet add:

- multiple different pretrained baselines mapped to several target columns;
- vector/tensor direct properties such as atom-wise forces or stress tensors;
- independent-output residual `ModelList` models;
- FastAPI/Web `model_type` routing for the new residual variants.

Those concerns can be added independently without changing the current correlated multitask contract.
