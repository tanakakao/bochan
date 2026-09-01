# M3GNet structure-aware FastAPI

Bochan exposes MatGL M3GNet-backed structure/process Bayesian optimization through
`/api/v1/tabular/m3gnet/models`.

## Supported model types

| model_type | output dependency | encoder training |
| --- | --- | --- |
| `m3gnet_gp` | single / independent multi-output | frozen |
| `m3gnet_dkl` | single / independent multi-output | partial or full |
| `m3gnet_multitask` | correlated multi-output | frozen |
| `m3gnet_multitask_dkl` | correlated multi-output | partial or full |

Independent multi-output models use one M3GNet/GP state per target. Correlated
multitask models share one M3GNet representation and learn output covariance
through the multitask GP kernel.

## Structure payload

`structure_catalog` maps stable structure IDs to inline structures. The structure
ID column must be included in `input_cols`. Mapping, CIF text, and POSCAR text are
accepted. Client-controlled server filesystem paths are not accepted.

```json
{
  "structure_col": "phase",
  "structure_catalog": {
    "alpha": {
      "format": "mapping",
      "lattice_mat": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
      "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
      "elements": ["Si", "Si"],
      "cartesian": false
    }
  }
}
```

The first internal model feature is the discrete structure index. Remaining
features are process variables. Continuous process variables require
column-addressed bounds. Categorical process variables are enumerated together
with structures during candidate generation.

## Pretrained M3GNet

The FastAPI surface currently allowlists the MatGL model validated by bochan:

- `M3GNet-PES-MatPES-PBE-2025.2`

Example configuration:

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "m3gnet_gp",
    "model_kwargs": {
      "model_name": "M3GNet-PES-MatPES-PBE-2025.2",
      "latent_dim": 32
    }
  }
}
```

Python object injection such as `encoder`, `adapter`, `graph_converter`, or raw
`structures` is intentionally rejected over HTTP. Use the Python API when a
custom in-memory encoder is required.

## Representation contract

Bochan calls the bare MatGL M3GNet model directly so the structure representation
remains differentiable for DKL. Intensive models use the graph-level readout.
Extensive MatPES-style models mean-pool the final message-passing node features,
keeping the representation upstream of the original property head.

Fit/save/load responses expose the selected `representation_mode` in
`metadata.m3gnet`.

## DKL fine-tuning

For `m3gnet_dkl` and `m3gnet_multitask_dkl`, use the API-safe alias:

```json
{"model_kwargs": {"encoder_training": "partial"}}
```

`partial` trains the final M3GNet graph layer. `full` trains the representation
backbone exposed by `M3GNetEncoder.backbone_modules()`. The original property
head remains outside the representation fine-tuning contract.

## Endpoints

- `POST /api/v1/tabular/m3gnet/models` — fit/store a model
- `POST /api/v1/tabular/m3gnet/models/{model_id}/tell` — append observations
- `POST /api/v1/tabular/m3gnet/models/{model_id}/predict` — posterior prediction
- `POST /api/v1/tabular/m3gnet/models/{model_id}/candidates` — generate candidates
- `POST /api/v1/tabular/m3gnet/models/{model_id}/ask` — generate and register pending candidates
- `POST /api/v1/tabular/m3gnet/models/{model_id}/save` — save a trusted tabular artifact
- `POST /api/v1/tabular/m3gnet/models/load` — load a trusted artifact

Candidate requests may pass `structure_ids` to restrict the discrete structure
search space.

## Artifact contract

M3GNet uses the common versioned tabular `.bochan.pt` artifact envelope. The
artifact retains:

- structure catalog and stable structure-ID mapping
- process category maps
- fitted GP/DKL parameters
- M3GNet encoder state
- pretrained model and representation metadata
- correlated vs independent output contract
- pending/observed training state

Loading pickle-backed model artifacts requires explicit `trust_pickle=true`.
Only load artifacts from trusted sources.

## Response metadata

Fit/save/load responses include `metadata.m3gnet` with the pretrained model name,
encoder output width, representation mode, training mode, structure IDs, process
dimensions/categories, output names, output dependency, and multitask kernel
information when applicable.

## Runnable client

```bash
python -m pip install -e ".[api,tabular,materials]"
python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000
python examples/m3gnet_fastapi_client.py
```
