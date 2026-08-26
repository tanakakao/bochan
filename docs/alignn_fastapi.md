# ALIGNN-GP / ALIGNN-DKL FastAPI

This page shows the HTTP workflow for Bayesian optimization over a **known
catalog of crystal structures** plus **continuous and categorical process
conditions** using Bochan's pure-PyTorch ALIGNN integration.

The crystal structure is a discrete design choice. Bochan maps the user-facing
structure ID to an integer `structure_index`, encodes the corresponding graph
with ALIGNN, and never interpolates between crystal structures. Continuous
process variables remain differentiable optimization coordinates. Categorical
process variables are handled by the mixed GP kernel and discrete acquisition
optimization.

## Backend contract

The canonical path is:

```text
crystal structure
    -> StructureAdapter
    -> JARVIS Atoms
    -> alignn.torch_graph_builder.build_pure_torch_graph
    -> (TorchGraph atom_graph, TorchGraph line_graph)
    -> ALIGNNAtomWisePure representation backbone
    -> structure embedding

continuous process variables -------------------+
                                                 +-> ALIGNN mixed GP / DKL
categorical process variables -> categorical GP +        -> acquisition
```

No DGL graph, DGL convolution, or DGL runtime dependency is used on this path.

For a canonical fitted feature layout such as:

```text
[structure_index, temperature, furnace, pressure, atmosphere]
```

Bochan treats the coordinates as:

```text
structure_index       discrete graph selector; never a categorical kernel dim
temperature           continuous
furnace               categorical GP dim
pressure              continuous
atmosphere            categorical GP dim
```

Thus `dataset.cat_dims` includes the structure selector and process categories,
whereas `model.cat_dims` / `bundle.cat_dims` include **process categories only**.

## Current scope

Supported:

- single-output continuous regression
- `alignn_gp`
- `alignn_dkl`
- pure-PyTorch `ALIGNNAtomWisePure`
- pure-PyTorch `TorchGraph` atom/line graphs
- known structure-ID catalog
- continuous process variables
- categorical process variables
- automatic normal/mixed model selection from `categorical_cols`
- explicit or inferred category maps
- inline crystal mappings
- inline CIF text
- inline POSCAR text
- prediction for known structures and fitted categories
- candidate generation across all known structures or a selected subset
- candidate enumeration over observed joint process-category assignments
- `/ask` candidate registration as pending observations

Not yet supported:

- unknown/new structure generation
- structure topology mutation during acquisition optimization
- composition + crystal structure joint optimization
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

## Start the API

```bash
python -m uvicorn bochan.serving.fastapi.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Swagger / OpenAPI:

```text
http://127.0.0.1:8000/docs
```

All structure-aware ALIGNN endpoints are under:

```text
/api/v1/tabular/alignn/models
```

## Fastest end-to-end smoke test

A runnable mixed-process client is included:

```bash
python examples/alignn_fastapi_client.py
```

It executes fit, predict, and candidate generation using crystal structure,
continuous process variables, and categorical process variables. The example
uses `fit_config.skip_fit=true`; its numerical output is only a plumbing smoke
test and is not scientifically meaningful without an appropriate trained or
pretrained ALIGNN setup.

## 1. Fit an ALIGNN mixed GP model

A structure catalog is a JSON object whose keys are stable structure IDs.
Training rows refer to these IDs through `structure_col`.

```python
import httpx

base = "http://127.0.0.1:8000/api/v1"

payload = {
    "data": [
        {
            "phase": "alpha",
            "temperature": 900.0,
            "pressure": 0.8,
            "furnace": "A",
            "atmosphere": "air",
            "property": 0.40,
        },
        {
            "phase": "beta",
            "temperature": 930.0,
            "pressure": 1.0,
            "furnace": "B",
            "atmosphere": "N2",
            "property": 0.72,
        },
        {
            "phase": "alpha",
            "temperature": 980.0,
            "pressure": 1.2,
            "furnace": "A",
            "atmosphere": "Ar",
            "property": 0.68,
        },
        {
            "phase": "beta",
            "temperature": 1020.0,
            "pressure": 1.4,
            "furnace": "B",
            "atmosphere": "N2",
            "property": 1.05,
        },
    ],
    "input_cols": [
        "phase",
        "temperature",
        "pressure",
        "furnace",
        "atmosphere",
    ],
    "categorical_cols": ["furnace", "atmosphere"],
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
        "model_kwargs": {"latent_dim": 8},
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
```

### Bounds and categories

Only **continuous process variables** require bounds. Do not invent numerical
bounds for categorical variables merely to satisfy the HTTP schema:

```python
"categorical_cols": ["furnace", "atmosphere"],
"bounds": {
    "temperature": [850.0, 1150.0],
    "pressure": [0.5, 2.0],
},
```

When `category_maps` is omitted, Bochan fits stable category maps from the
training data and reuses them for prediction and subsequent updates. They can
also be supplied explicitly:

```python
"category_maps": {
    "furnace": {"A": 0, "B": 1},
    "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
},
```

The structure ID mapping is separate and remains owned by `structure_catalog`.
Do not treat `structure_col` as an ordinary process categorical kernel input.

### Fit response metadata

`metadata.alignn` exposes the resolved contract, including:

```text
input_type
structure_col
structure_ids
continuous_process_cols
categorical_process_cols
categorical_process_dims
category_maps
process_dim
graph_config
encoder_training
encoder_initialization
```

This makes it possible for an API client to verify whether the server resolved
the model as `normal` or `mixed` and which process coordinates are categorical.

## 2. Predict

Prediction rows use the original structure IDs and original process-category
labels:

```python
prediction = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/predict",
    json={
        "data": [
            {
                "phase": "alpha",
                "temperature": 1000.0,
                "pressure": 1.1,
                "furnace": "A",
                "atmosphere": "air",
            },
            {
                "phase": "beta",
                "temperature": 1000.0,
                "pressure": 1.1,
                "furnace": "B",
                "atmosphere": "N2",
            },
        ],
        "include_input": True,
    },
    timeout=120.0,
)
prediction.raise_for_status()
print(prediction.json())
```

The fitted category maps are reused. Unknown category labels are therefore not
silently recoded with a new ordering.

## 3. Generate structure + mixed-process candidates

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

For example, with two structures and observed process-category assignments:

```text
(A, air)
(B, N2)
(A, Ar)
```

Bochan constructs the discrete search as:

```text
{alpha, beta}
    x
{(A, air), (B, N2), (A, Ar)}
```

and optimizes only continuous variables such as `temperature` and `pressure`
inside each discrete configuration. The categorical combinations come from the
**current training inputs**, so combinations added later through `tell()` /
`update_data()` participate in subsequent candidate generation.

The candidate response is decoded back to original values such as:

```text
phase="beta"
furnace="B"
atmosphere="N2"
```

rather than returning category codes.

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
structure ID/index mapping and the mixed discrete enumeration.

## 4. Asynchronous `/ask`

`/ask` uses the same structure/category/continuous candidate generation but also
registers the returned experiment as pending:

```python
asked = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/ask",
    json={
        "acquisition_config": {"name": "logei"},
        "optimize_config": {"q": 1},
    },
)
```

Subsequent acquisitions can consume it as `X_pending` and avoid proposing the
same outstanding experiment.

## 5. Normal continuous-only ALIGNN remains supported

Omit `categorical_cols` when all process variables are continuous:

```python
"input_cols": ["phase", "temperature", "pressure"],
"bounds": {
    "temperature": [850.0, 1150.0],
    "pressure": [0.5, 2.0],
},
```

The same public model names are used in both cases:

```text
alignn_gp
alignn_dkl
```

Bochan resolves `input_type="normal"` or `input_type="mixed"` from the fitted
process-column contract instead of introducing separate HTTP model names.

## 6. Use CIF or POSCAR content

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

## 7. ALIGNN-DKL

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

The same mixed process-column contract applies to both GP and DKL variants.

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

For FastAPI, configure an allowlisted checkpoint directory on the server, e.g.

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

Then the client sends only a filename identifier. Path traversal and absolute
client paths are rejected.

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
| POST | `/api/v1/tabular/alignn/models/{model_id}/predict` | Predict known structure/process conditions |
| POST | `/api/v1/tabular/alignn/models/{model_id}/candidates` | Generate structure + continuous/categorical process candidates |
| POST | `/api/v1/tabular/alignn/models/{model_id}/ask` | Generate and register pending candidates |

The model uses the same tabular model store as the existing generic FastAPI
endpoints; no second model registry is introduced.
