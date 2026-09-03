# Independent multi-output material Residual GP

This phase adds an independent-output alternative to the correlated `*_multitask_residual_gp` models for CHGNet, M3GNet, and MACE.

## Public model types

Continuous process inputs:

- `chgnet_multioutput_residual_gp`
- `m3gnet_multioutput_residual_gp`
- `mace_multioutput_residual_gp`

Categorical + continuous process inputs:

- `chgnet_mixed_multioutput_residual_gp`
- `m3gnet_mixed_multioutput_residual_gp`
- `mace_mixed_multioutput_residual_gp`

These model types are available through the tabular/Python workflow and the existing material-residual FastAPI surface. No Web UI exposure is added.

## Semantics

For targets

```text
[energy, strength, conductivity]
```

with

```text
pretrained_output_index = 0
```

the model is constructed as independent BoTorch submodels:

```text
ModelListGP
├── energy       : pretrained structure prediction + residual GP
├── strength     : ordinary structure/process GP
└── conductivity : ordinary structure/process GP
```

Only the selected output is required to match the pretrained predictor's physical property definition and units. Other outputs have no deterministic baseline.

This differs from `*_multitask_residual_gp`:

- `multioutput_residual_gp`: independent output GPs; no learned cross-output covariance.
- `multitask_residual_gp`: one correlated multitask GP; learns cross-output covariance.

## Structure/process contract

All submodels use the same raw structure bank and fitted tabular layout:

```text
[structure_index, process...]
```

The structure selector remains feature 0. For mixed variants, `cat_dims` contains categorical process columns only. Each submodel owns its own trainable GP state and material encoder instance; the immutable structure bank is shared by reference.

The pretrained direct predictor ignores process variables. Numeric and categorical process dependence is learned by the residual GP for the selected output and by ordinary family GPs for the other outputs.

## BoTorch compatibility

`ResidualMaterialGPModel` now implements the `GPyTorchModel` protocol and delegates `likelihood`, `train_inputs`, `train_targets`, and batch/output metadata to its exact residual backend. This allows a residual wrapper and ordinary exact GP models to coexist in the standard BoTorch `ModelListGP`.

Each submodel is fitted independently through Bochan's existing multi-output fitting path. Known observation variance is sliced per output and passed to the corresponding submodel unchanged.

## FastAPI

The existing endpoint is reused:

```text
POST /api/v1/tabular/material-residual/models
```

Example model configuration:

```json
{
  "task_type": "multi_objective",
  "model_type": "mace_multioutput_residual_gp",
  "model_kwargs": {
    "model_name": "medium-mpa-0",
    "pretrained_output_index": 0
  }
}
```

`pretrained_output_index` must select exactly one target column. User-supplied `multi_output_config` is rejected because Bochan derives the safe output decomposition from the model type.

## Not included

This phase intentionally does not add:

- multiple different pretrained predictors mapped to multiple output columns;
- force/stress tensor outputs;
- correlated covariance between independent outputs;
- Web UI model selectors.

Multiple pretrained baselines are the next natural extension of this independent-output architecture.
