# MACE structure-aware FastAPI

Bochan exposes MACE-backed structure/process Bayesian optimization through
`/api/v1/tabular/mace/models`.

## Supported model types

| model_type | output dependency | encoder training |
| --- | --- | --- |
| `mace_gp` | single / independent multi-output | frozen |
| `mace_dkl` | single / independent multi-output | partial or full |
| `mace_multitask` | correlated multi-output | frozen |
| `mace_multitask_dkl` | correlated multi-output | partial or full |

Independent multi-output models use one MACE/GP state per target. Correlated
multitask models share one MACE representation and learn output covariance
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

## Pretrained MACE

The public FastAPI surface currently allowlists the MACE foundation checkpoint
validated by bochan:

- `medium-mpa-0`

Example configuration:

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "mace_gp",
    "model_kwargs": {
      "model_name": "medium-mpa-0",
      "latent_dim": 32,
      "num_layers": -1,
      "pooling": "mean"
    }
  }
}
```

Python object injection such as `encoder`, `adapter`, raw `structures`, or a
custom `batch_builder` is intentionally rejected over HTTP. Use the Python API
when a custom in-memory encoder is required.

## Representation contract

Bochan calls the raw torch MACE model directly. It reads `output["node_feats"]`,
uses MACE's invariant extraction utility to retain the scalar `l=0` channels,
and then mean- or sum-pools those atomwise descriptors into one crystal vector.
The original energy readout is not used as the GP feature representation.

Fit/save/load responses expose this as `representation_mode="invariant_l0"` in
`metadata.mace`, together with:

- `model_name`
- `encoder_output_dim`
- `num_layers` and `num_interactions`
- `pooling`
- selected `head` and `available_heads`
- MACE cutoff
- frozen/partial/full encoder-training mode

## DKL fine-tuning

For `mace_dkl` and `mace_multitask_dkl`, use the API-safe alias:

```json
{"model_kwargs": {"encoder_training": "partial"}}
```

`partial` trains the final MACE interaction/product pair. `full` trains the
representation backbone exposed by `MACEEncoder.backbone_modules()`. Energy
readout modules remain outside the representation fine-tuning contract.

The lower-level `trainable_encoder_layers` argument is intentionally not accepted
over FastAPI; it remains available through the Python API.

## Endpoints

- `POST /api/v1/tabular/mace/models` — fit/store a model
- `POST /api/v1/tabular/mace/models/{model_id}/tell` — append observations
- `POST /api/v1/tabular/mace/models/{model_id}/predict` — posterior prediction
- `POST /api/v1/tabular/mace/models/{model_id}/candidates` — generate candidates
- `POST /api/v1/tabular/mace/models/{model_id}/ask` — generate and register pending candidates
- `POST /api/v1/tabular/mace/models/{model_id}/save` — save a trusted tabular artifact
- `POST /api/v1/tabular/mace/models/load` — load a trusted artifact

Candidate requests may pass `structure_ids` to restrict the discrete structure
search space.

## Artifact contract

MACE uses the common versioned tabular `.bochan.pt` artifact envelope. The
artifact retains the structure catalog and stable IDs, process category maps,
fitted GP/DKL state, MACE encoder state, representation metadata, correlated vs
independent output contract, and observation state.

Loading pickle-backed model artifacts requires explicit `trust_pickle=true`.
Only load artifacts from trusted sources.

## Runnable client

```bash
python -m pip install -e ".[api,tabular,materials]"
python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000
python examples/mace_fastapi_client.py
```
