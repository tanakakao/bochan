# ALIGNN-GP / ALIGNN-DKL FastAPI

This page shows the HTTP workflow for Bayesian optimization over a **known
catalog of crystal structures** and **continuous process conditions** using
Bochan's pure-PyTorch ALIGNN integration.

The crystal structure is a discrete design choice. Bochan maps the user-facing
structure ID to an integer `structure_index`, enumerates that coordinate during
candidate optimization, and optimizes only the continuous process variables.
The optimizer never interpolates between structures.

## Backend contract

The canonical path is now:

```text
crystal structure
    -> StructureAdapter
    -> JARVIS Atoms
    -> alignn.torch_graph_builder.build_pure_torch_graph
    -> (TorchGraph atom_graph, TorchGraph line_graph)
    -> ALIGNNAtomWisePure representation backbone
    -> structure embedding
    -> process-feature fusion
    -> exact GP / DKL
    -> BoTorch acquisition optimization
```

No DGL graph, DGL convolution, or DGL runtime dependency is used on this path.

The upstream pure-PyTorch implementation uses tensor/index/scatter operations
for periodic neighbor search, line-graph construction, message passing, and
readout. Bochan keeps the first model input coordinate as the discrete
`structure_index`; process columns remain differentiable continuous variables.

## Current scope

Supported:

- single-output continuous regression
- `alignn_gp`
- `alignn_dkl`
- pure-PyTorch `ALIGNNAtomWisePure`
- pure-PyTorch `TorchGraph` atom/line graphs
- one structure-ID column plus continuous process columns
- inline crystal mappings
- inline CIF text
- inline POSCAR text
- prediction for known structures
- candidate generation across all known structures or a selected subset
- `/ask` candidate registration as pending observations

Not yet supported:

- unknown/new structure generation
- structure topology mutation during acquisition optimization
- composition + crystal structure joint optimization
- categorical process variables
- multi-output ALIGNN
- legacy DGL ALIGNN checkpoints on the canonical path
- client-controlled server filesystem paths for CIF/POSCAR

## Installation

From the repository root:

```bash
python -m pip install -e ".[api,tabular]"
python -m pip install "alignn==2026.8.11"
```

DGL is **not required**.

The ALIGNN dependency is installed explicitly for now so the atomistic stack
remains optional to the Bochan core package.

## Start the API

```bash
python -m uvicorn bochan.serving.fastapi.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Open Swagger / OpenAPI at:

```text
http://127.0.0.1:8000/docs
```

All structure-aware ALIGNN endpoints are under:

```text
/api/v1/tabular/alignn/models
```

## Fastest end-to-end smoke test

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

The example uses `fit_config.skip_fit=true`. It verifies the HTTP layer,
structure parsing, pure-Torch graph construction, pure ALIGNN encoder
construction, prediction routing, and candidate routing quickly. Its numerical
predictions are not intended for scientific interpretation.

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
            "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
            "elements": ["Si", "Si"],
            "cartesian": False,
        },
        "beta": {
            "format": "mapping",
            "lattice_mat": [[5.55, 0, 0], [0, 5.55, 0], [0, 0, 5.55]],
            "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
            "elements": ["Si", "Si"],
            "cartesian": False,
        },
    },
    "structure_graph_config": {
        "neighbor_strategy": "pure_torch",
        "cutoff": 8.0,
        "max_neighbors": 12,
        "three_body_cutoff": 3.5,
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

The default pure graph settings follow the current upstream scalar-property kNN
recipe:

```text
neighbor_strategy = pure_torch
cutoff             = 8.0 Å
max_neighbors      = 12
atom_features      = cgcnn
three_body_cutoff  = 3.5 Å
compute_line_graph = true
```

The fit response contains `metadata.alignn`, including the encoder training
mode, initialization, structure IDs, process dimension, and graph config.

The feature order is canonicalized so the structure selector is feature index
0 even when the incoming `input_cols` order differs.

## 2. Predict

Prediction rows use original structure IDs, not integer indices.

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

## 3. Generate a structure + process candidate

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

Internally the structure coordinate is enumerated conceptually as:

```python
fixed_features_list = [
    {0: 0.0},  # alpha
    {0: 1.0},  # beta
]
```

while `temperature` and `pressure` remain continuous optimization variables.
The response is decoded back to `phase="alpha"` or `phase="beta"`.

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

Use `structure_ids`, not a client-supplied `fixed_features_list`; Bochan owns the
structure ID/index mapping.

## 4. Asynchronous `/ask`

`/ask` uses the same candidate generation but additionally registers the
returned experiment as pending. Subsequent acquisitions can therefore consume
it as `X_pending` and avoid proposing the same experiment while its result is
outstanding.

```python
asked = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/ask",
    json={
        "acquisition_config": {"name": "logei"},
        "optimize_config": {"q": 1},
    },
)
```

## 5. Use CIF or POSCAR content

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

The server parses the supplied text through `StructureAdapter`, then creates
pure upstream `TorchGraph` objects. Temporary structure files used internally
for parser compatibility are removed after parsing.

## 6. ALIGNN-DKL

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
- `full`: fine-tune the complete pure ALIGNN representation backbone

For `alignn_gp`, the encoder is frozen and `encoder_training` must not be
specified.

## Checkpoints and scientifically meaningful use

If no checkpoint is configured, Bochan constructs `ALIGNNAtomWisePure` with
random initialization. This is useful for plumbing tests and for a DKL workflow
that intentionally learns from scratch. A frozen `alignn_gp` with random ALIGNN
features should not be treated as a pretrained materials model.

Bochan's canonical checkpoint contract is pure PyTorch:

```text
model.name = "alignn_atomwise_pure"
neighbor_strategy = "pure_torch"
```

Legacy DGL `model.name="alignn"` checkpoints are rejected rather than silently
loaded into a different representation implementation.

For FastAPI, the client cannot send an arbitrary server filesystem path. The
server operator first configures an allowlisted checkpoint directory, for
example:

Windows cmd:

```bat
set BOCHAN_ALIGNN_CHECKPOINT_ROOT=C:\models\alignn
```

PowerShell:

```powershell
$env:BOCHAN_ALIGNN_CHECKPOINT_ROOT = "C:\models\alignn"
```

Linux/macOS:

```bash
export BOCHAN_ALIGNN_CHECKPOINT_ROOT=/opt/bochan/alignn-checkpoints
```

Then the client sends only a filename identifier:

```python
"model_config": {
    "task_type": "regression",
    "model_type": "alignn_gp",
    "model_kwargs": {
        "checkpoint": "formation_energy_pure.pt",
        "encoder_config": {
            "name": "alignn_atomwise_pure",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,
            "calculate_gradient": False,
            "gradwise_weight": 0.0,
            "energy_mult_natoms": False,
        },
        "latent_dim": 16,
    },
}
```

The server resolves that filename only inside
`BOCHAN_ALIGNN_CHECKPOINT_ROOT`. Path traversal and absolute client paths are
rejected.

When transferring a pretrained model, `structure_graph_config` must match the
training graph settings. Model weights are not graph-construction agnostic.

## Real fitting versus smoke fitting

The included example uses:

```python
"fit_config": {"skip_fit": True}
```

For real fitting, remove `skip_fit` or set it to `False` and configure an
appropriate optimization budget. DKL fine-tuning is more expensive than a
frozen ALIGNN-GP because structure representations are recomputed while encoder
parameters are updated.

## Endpoint summary

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/tabular/alignn/models` | Fit/store pure ALIGNN-GP or ALIGNN-DKL |
| POST | `/api/v1/tabular/alignn/models/{model_id}/predict` | Predict known structures/process conditions |
| POST | `/api/v1/tabular/alignn/models/{model_id}/candidates` | Generate structure + process candidates |
| POST | `/api/v1/tabular/alignn/models/{model_id}/ask` | Generate and register pending candidates |

The model uses the same tabular model store as the existing generic FastAPI
endpoints; no second model registry is introduced.
