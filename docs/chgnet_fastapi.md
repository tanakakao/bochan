# CHGNet structure-aware FastAPI

Bochan exposes CHGNet-backed structure/process Bayesian optimization through
`/api/v1/tabular/chgnet/models`.

## Supported model types

| model_type | output dependency | encoder training |
| --- | --- | --- |
| `chgnet_gp` | single / independent multi-output | frozen |
| `chgnet_dkl` | single / independent multi-output | partial or full |
| `chgnet_multitask` | correlated multi-output | frozen |
| `chgnet_multitask_dkl` | correlated multi-output | partial or full |

Independent multi-output models use one CHGNet/GP state per target. Correlated
multitask models share one CHGNet representation and learn output covariance
through the multitask GP kernel.

## Structure payload

`structure_catalog` is a mapping from stable structure IDs to inline structures.
The structure ID column must be included in `input_cols`. Mapping, CIF text, and
POSCAR text are accepted. Client-controlled server filesystem paths are not.

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
column-addressed bounds; categorical process variables are enumerated together
with structures during candidate generation.

## Pretrained CHGNet

FastAPI accepts the upstream CHGNet pretrained names:

- `0.3.0` (default)
- `0.2.0`
- `r2scan`

Example model configuration:

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "chgnet_gp",
    "model_kwargs": {
      "model_name": "0.3.0",
      "latent_dim": 32
    }
  }
}
```

Python object injection (`encoder`, `adapter`, raw `structures`) is intentionally
not accepted over HTTP.

## DKL fine-tuning

For `chgnet_dkl` and `chgnet_multitask_dkl`, use the API-safe alias:

```json
{"model_kwargs": {"encoder_training": "partial"}}
```

`partial` trains the final CHGNet atom-convolution block. `full` trains the
representation backbone contributing to `crystal_fea`; the unrelated CHGNet
property head remains frozen.

## Server-side checkpoints

Custom checkpoint paths are never accepted directly from clients. Configure a
server-owned directory:

```bash
export BOCHAN_CHGNET_CHECKPOINT_ROOT=/srv/bochan/chgnet-checkpoints
```

Then send only a filename identifier:

```json
{"model_kwargs": {"checkpoint": "my-chgnet.pt"}}
```

The resolved file must remain directly under the configured root.

## Endpoints

- `POST /api/v1/tabular/chgnet/models` — fit/store a model
- `POST /api/v1/tabular/chgnet/models/{model_id}/tell` — append observations
- `POST /api/v1/tabular/chgnet/models/{model_id}/predict` — posterior prediction
- `POST /api/v1/tabular/chgnet/models/{model_id}/candidates` — generate candidates
- `POST /api/v1/tabular/chgnet/models/{model_id}/ask` — generate and register pending candidates
- `POST /api/v1/tabular/chgnet/models/{model_id}/save` — save a trusted tabular artifact
- `POST /api/v1/tabular/chgnet/models/load` — load a trusted artifact

Candidate requests may pass `structure_ids` to restrict the discrete structure
search space.

## Artifact contract

CHGNet uses the common versioned tabular `.bochan.pt` artifact envelope rather
than a CHGNet-specific file format. The artifact retains:

- structure catalog and stable structure-ID mapping
- process category maps
- fitted GP/DKL parameters
- CHGNet encoder state
- pretrained/checkpoint initialization metadata
- correlated vs independent output contract
- pending/observed training state

Loading pickle-backed model artifacts requires explicit `trust_pickle=true`.
Only load artifacts from trusted sources.

## Response metadata

Fit/save/load responses include a `metadata.chgnet` block containing the
pretrained model name, encoder output width, training mode, structure IDs,
process dimensions/categories, output names, output dependency, and multitask
kernel information when applicable.

## Runnable client

Start the API with materials support and run:

```bash
python -m pip install -e ".[api,tabular,materials]"
python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000
python examples/chgnet_fastapi_client.py
```
