# Multiple pretrained baselines over FastAPI

The material-residual FastAPI surface supports independent outputs whose pretrained baselines come from different structure-model families.

## Public model types

Use one of:

```text
material_multi_baseline_residual_gp
material_mixed_multi_baseline_residual_gp
```

The mixed variant is required when categorical process columns are present. The structure selector remains feature 0 and is not included in `cat_dims`.

## Example request fragment

```json
{
  "model_config": {
    "task_type": "multi_objective",
    "model_type": "material_multi_baseline_residual_gp",
    "model_kwargs": {}
  },
  "target_cols": ["energy", "band_gap", "strength"],
  "baseline_specs": [
    {
      "family": "mace",
      "output_name": "energy",
      "quantity": "energy",
      "unit": "eV",
      "aggregation": "total",
      "model_name": "medium-mpa-0"
    },
    {
      "family": "m3gnet",
      "output_name": "band_gap",
      "quantity": "band_gap",
      "unit": "eV",
      "aggregation": "intensive",
      "model_name": "M3GNet-PES-MatPES-PBE-2025.2"
    }
  ],
  "ordinary_family": "chgnet",
  "ordinary_model_kwargs": {
    "model_name": "0.3.0"
  }
}
```

The resulting independent wrapper is conceptually:

```text
energy    -> MACE pretrained baseline + residual GP
band_gap  -> M3GNet pretrained baseline + residual GP
strength  -> ordinary CHGNet GP
```

`ordinary_family` is required only when at least one target is not assigned a pretrained baseline. It is intentionally explicit so Bochan never silently chooses a structure encoder for unassigned outputs.

## Physical contract

Each baseline route declares `quantity`, `unit`, and `aggregation`. These values are stored in `MaterialBaselineSpec` metadata. No automatic unit or aggregation conversion is performed.

## Family-specific settings

Each baseline may supply `model_kwargs`. The same server-side safety policy as the existing material-residual API applies: clients cannot inject encoders, structures, graph builders, or trainable encoder controls.

For CHGNet, `checkpoint` remains a filename identifier. The server resolves it only below `BOCHAN_CHGNET_CHECKPOINT_ROOT`; client-controlled filesystem paths are rejected.

## Candidate optimization

The two multiple-baseline model types are structure-aware. Candidate generation therefore supports `structure_ids`, mixed process-category enumeration, and the existing alternating structure optimizer for larger structure catalogs.

## Compatibility

Existing `chgnet_*_residual_gp`, `m3gnet_*_residual_gp`, and `mace_*_residual_gp` public model types are unchanged. The cross-family model types use the same `/api/v1/tabular/material-residual/models` endpoint family and do not add Web UI exposure.
