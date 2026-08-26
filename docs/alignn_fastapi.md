# ALIGNN-GP / ALIGNN-DKL FastAPI

This page shows the Phase-3 HTTP workflow for Bayesian optimization over a
**known catalog of crystal structures** and **continuous process conditions**.

The structure itself is a discrete design choice. Bochan maps the user-facing
structure ID to an integer `structure_index`, enumerates that coordinate during
candidate optimization, and optimizes only the continuous process variables.
The optimizer therefore never interpolates between structure IDs.

## Current scope

Supported:

- single-output continuous regression
- `alignn_gp`
- `alignn_dkl`
- one structure-ID column plus continuous process columns
- inline crystal mappings
- inline CIF text
- inline POSCAR text
- prediction for known structures
- candidate generation across all known structures or a selected subset
- `/ask` pending-candidate registration through the normal tabular optimizer

Not yet supported:

- unknown/new structure generation
- structure topology mutation
- composition + crystal structure joint optimization
- categorical process variables
- multi-output ALIGNN
- client-controlled server filesystem paths for CIF/POSCAR

## Installation

From the repository root:

```bash
python -m pip install -e ".[api,tabular,alignn]"
```

The current Bochan ALIGNN-GP/DKL implementation uses the historical scalar
`alignn.models.alignn.ALIGNN` backbone and the DGL graph builder. Install a DGL
build compatible with the local PyTorch/CUDA environment before a real ALIGNN
fit request.

Do not install an arbitrary DGL version merely to satisfy the import: DGL,
PyTorch and CUDA versions must be compatible. If a suitable native Windows DGL
build is unavailable for the environment, WSL2/Linux is the safer test path.

Upstream ALIGNN now also provides pure-PyTorch model variants that do not need
DGL. Bochan does not yet use that pure-PyTorch backbone in this Phase-3 path;
that migration should be implemented separately rather than silently mixing
legacy checkpoints with a different graph/model contract.

## Start the API

```bash
python -m uvicorn bochan.serving.fastapi.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

OpenAPI / Swagger is available at:

```text
http://127.0.0.1:8000/docs
```

All ALIGNN tabular endpoints are under:

```text
/api/v1/tabular/alignn/models
```

## Fastest smoke test

A runnable client is included:

```bash
python examples/alignn_fastapi_client.py
```

It executes:

```text
POST /api/v1/tabular/alignn/models
  -> model_id
POST /api/v1/tabular/alignn/models/{model_id}/predict
POST /api/v1/tabular/alignn/models/{model_id}/candidates
```

The example uses `fit_config.skip_fit=true`. This checks the complete HTTP,
structure parsing, graph construction, model construction, prediction and
candidate-routing path quickly, but the numerical predictions are not intended
for scientific interpretation.

## 1. Fit an ALIGNN-GP model

A structure catalog is a JSON object whose keys are stable structure IDs.
Training rows refer to these IDs through `structure_col`.

```python
import httpx

base = "http://127.0.0.1:8000/api/v1"

payload = {
    "data": [
        {"phase": "alpha", "temperature": 900.0, "pressure": 0.8, "property": 0.40},
        {"phase": "beta",  "temperature": 930.0, "pressure": 1.0, "property": 0.72},
        {"phase": "alpha", "temperature": 980.0, "pressure": 1.2, "property": 0.68},
        {"phase": "beta",  "temperature": 1020.0, "pressure": 1.4, "property": 1.05},
    ],
    "input_cols": ["phase", "temperature", "pressure"],
    "target_cols": "property",
    "structure_col": "phase",
    "structure_catalog": {
        "alpha": {
            "format": "mapping",
            "lattice_mat": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
            "coords": [[0, 0, 0]],
            "elements": ["Si"],
            "cartesian": False,
        },
        "beta": {
            "format": "mapping",
            "lattice_mat": [[5.55, 0, 0], [0, 5.55, 0], [0, 0, 5.55]],
            "coords": [[0, 0, 0]],
            "elements": ["Si"],
            "cartesian": False,
        },
    },
    "bounds": {
        "temperature": [850.0, 1150.0],
        "pressure": [0.5, 2.0],
    },
    "model_config": {
        "task_type": "regression",
        "model_type": "alignn_gp",
        "model_kwargs": {
            "latent_dim": 8,
        },
    },
    "fit_config": {"skip_fit": True},
}

response = httpx.post(
    f"{base}/tabular/alignn/models",
    json=payload,
    timeout=120.0,
)
response.raise_for_status()
result = response.json()
model_id = result["model_id"]
print(result)
```

The fit response contains `metadata.alignn`, including:

```text
encoder_training
encoder_initialization
checkpoint_configured
structure_col
structure_ids
num_structures
process_dim
graph_config
```

The feature order returned by the fitted model is canonicalized so the
structure selector is feature index 0 even if the incoming `input_cols` order
was different.

## 2. Predict

Prediction rows use the original structure IDs, not integer indices.

```python
prediction = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/predict",
    json={
        "data": [
            {"phase": "alpha", "temperature": 1000.0, "pressure": 1.1},
            {"phase": "beta", "temperature": 1000.0, "pressure": 1.1},
        ],
        "include_input": True,
    },
    timeout=120.0,
)
prediction.raise_for_status()
print(prediction.json())
```

Only structure IDs present in the fitted structure catalog are valid.

## 3. Generate a candidate

```python
candidate = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/candidates",
    json={
        "acquisition_config": {"name": "logei"},
        "optimize_config": {
            "q": 1,
            "num_restarts": 8,
            "raw_samples": 128,
        },
    },
    timeout=120.0,
)
candidate.raise_for_status()
print(candidate.json())
```

Internally this is equivalent to enumerating:

```python
fixed_features_list = [
    {0: 0.0},  # alpha
    {0: 1.0},  # beta
]
```

while `temperature` and `pressure` remain continuous optimization variables.
The returned DataFrame/JSON is decoded back to `phase="alpha"` or
`phase="beta"`.

### Restrict the structure subset

```python
candidate = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/candidates",
    json={
        "acquisition_config": {"name": "logei"},
        "optimize_config": {"q": 1},
        "structure_ids": ["beta"],
    },
)
```

Do not pass `fixed_features_list` directly for ALIGNN tabular models. Use
`structure_ids`; Bochan owns the structure-index mapping.

## 4. Use CIF or POSCAR content

The HTTP API deliberately does not accept a client-provided server path such as
`C:/data/sample.cif` or `/tmp/sample.cif`.

Read the file on the client and send its contents:

```python
from pathlib import Path

structure_catalog = {
    "sample-cif": {
        "format": "cif",
        "content": Path("sample.cif").read_text(encoding="utf-8"),
    },
    "sample-poscar": {
        "format": "poscar",
        "content": Path("POSCAR").read_text(encoding="utf-8"),
    },
}
```

The server parses the supplied text through the canonical `StructureAdapter`.
Temporary files are internal implementation details and are removed after
parsing.

## 5. ALIGNN-DKL

Change the model section to:

```python
"model_config": {
    "task_type": "regression",
    "model_type": "alignn_dkl",
    "model_kwargs": {
        "latent_dim": 16,
        "encoder_training": "partial",
    },
}
```

`encoder_training` is:

- `partial`: fine-tune the final ALIGNN/GCN block
- `full`: fine-tune the complete representation backbone

For `alignn_gp`, the ALIGNN encoder is frozen and `encoder_training` must not be
specified.

## Checkpoints and scientifically meaningful use

If no checkpoint is configured, Bochan constructs an upstream ALIGNN encoder
with random initialization. This is useful for plumbing tests and for a DKL
workflow that intentionally learns from scratch, but a frozen `alignn_gp` with
random ALIGNN features is not a meaningful pretrained materials model.

For a frozen ALIGNN-GP workflow, provide a compatible checkpoint through
`model_config.model_kwargs.checkpoint` and ensure that the encoder model config
and graph-construction settings match the checkpoint training configuration.
For example:

```python
"model_config": {
    "task_type": "regression",
    "model_type": "alignn_gp",
    "model_kwargs": {
        "checkpoint": "C:/trusted/server/path/checkpoint.pt",
        "encoder_config": {
            "name": "alignn"
        },
        "latent_dim": 16,
    },
}
```

`checkpoint` is a server-side trusted path. Unlike CIF/POSCAR inputs, model
checkpoint loading is not an upload API in this phase.

When using a pretrained checkpoint, also set `structure_graph_config` to the
settings used during its training. A checkpoint is not graph-construction
agnostic.

## Real fitting versus smoke fitting

For the included smoke example:

```python
"fit_config": {"skip_fit": True}
```

For real fitting, remove `skip_fit` or set it to `False` and configure the fit
budget appropriate for the chosen model. DKL fine-tuning is substantially more
expensive than a frozen ALIGNN-GP because crystal representations must be
recomputed while encoder parameters are being optimized.

## Endpoint summary

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/tabular/alignn/models` | Fit/store ALIGNN-GP or ALIGNN-DKL |
| POST | `/api/v1/tabular/alignn/models/{model_id}/predict` | Predict known structures/process conditions |
| POST | `/api/v1/tabular/alignn/models/{model_id}/candidates` | Generate structure + process candidates |
| POST | `/api/v1/tabular/alignn/models/{model_id}/ask` | Generate candidates and register them pending |

The model is stored in the same tabular model store as the existing generic
FastAPI endpoints, so existing model listing/deletion infrastructure remains
shared rather than introducing a second model registry.
