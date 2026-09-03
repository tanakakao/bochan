# Material Residual GP FastAPI

Bochan exposes structure-aware residual Gaussian processes through the common FastAPI application at:

```text
POST /api/v1/tabular/material-residual/models
```

The endpoint supports CHGNet, M3GNet, and MACE pretrained baselines.  The pretrained prediction is deterministic and structure-only; the exact GP learns the correction from structure plus process variables.

## Public model types

Scalar residual GP:

- `chgnet_residual_gp`
- `chgnet_mixed_residual_gp`
- `m3gnet_residual_gp`
- `m3gnet_mixed_residual_gp`
- `mace_residual_gp`
- `mace_mixed_residual_gp`

Correlated multitask residual GP:

- `chgnet_multitask_residual_gp`
- `chgnet_mixed_multitask_residual_gp`
- `m3gnet_multitask_residual_gp`
- `m3gnet_mixed_multitask_residual_gp`
- `mace_multitask_residual_gp`
- `mace_mixed_multitask_residual_gp`

`*_mixed_*` is required when at least one process column is categorical.  The structure selector itself is not a mixed-kernel categorical dimension; it is enumerated separately as feature 0.

## Target contract

Scalar residual models require exactly one continuous target. The target must represent the same pretrained property and physical units as the selected pretrained model.

Correlated multitask residual models require at least two continuous targets. `model_config.model_kwargs.pretrained_output_index` selects the one target corrected around the pretrained scalar baseline. Its default is `0`. Other targets receive a zero deterministic baseline and are learned entirely by the correlated GP.

The API does not perform physical-unit conversion, per-atom/total-energy conversion, or reference-energy correction automatically.

## Structure and process inputs

Requests use the same inline structure catalog contract as the existing CHGNet/M3GNet/MACE endpoints:

- mapping (`lattice_mat`, `coords`, `elements`)
- inline CIF text
- inline POSCAR text

`structure_col` must be included in `input_cols`. Continuous process variables require column-addressed bounds. Categorical process variables are listed in `categorical_cols` and require a `*_mixed_*` model type.

The fitted tensor layout is always:

```text
[structure_index, process...]
```

The pretrained baseline ignores process variables. Process dependence is learned by the residual GP.

## Pretrained configuration

CHGNet accepts the existing server-safe pretrained model names and server-owned checkpoint identifiers. Client filesystem paths and injected Python encoders are rejected.

M3GNet uses the supported MatGL pretrained model name. MACE accepts `model_name`, `num_layers`, `pooling`, and optional `head` using the same validation rules as the existing MACE serving surface.

Residual models always freeze the pretrained encoder. `encoder_training` and `trainable_encoder_layers` are rejected.

## Candidate generation

The following endpoints are available:

```text
POST /api/v1/tabular/material-residual/models/{model_id}/predict
POST /api/v1/tabular/material-residual/models/{model_id}/candidates
POST /api/v1/tabular/material-residual/models/{model_id}/ask
POST /api/v1/tabular/material-residual/models/{model_id}/tell
```

Candidate generation reuses the canonical structure-aware optimizer. Small structure catalogs are enumerated. Larger catalogs may use `optimize_structure_alternating` when the requested BoTorch optimizer/settings are compatible. No residual-family-specific candidate optimizer is introduced.

## Save and load

Residual models use the existing trusted `.bochan.pt` artifact envelope:

```text
POST /api/v1/tabular/material-residual/models/{model_id}/save
POST /api/v1/tabular/material-residual/models/load
```

Loading requires the existing explicit pickle-trust flag. The artifact preserves the fitted structure catalog, process-category mappings, pretrained encoder, residual GP, and observation/pending state.

Fit/save/load metadata includes a `material_residual` section describing the family, model type, deterministic baseline contract, output dependency, structure IDs, process layout, and residual model class.
