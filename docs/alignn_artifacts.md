# ALIGNN FastAPI persistence and observation updates

Phase 5 extends the existing structure-aware ALIGNN FastAPI surface with
observation updates and common `.bochan.pt` persistence. The artifact format is
not ALIGNN-specific: it is the same trusted, versioned tabular artifact envelope
used by the rest of Bochan.

## Endpoints

```text
POST /api/v1/tabular/alignn/models/{model_id}/tell
POST /api/v1/tabular/alignn/models/{model_id}/save
POST /api/v1/tabular/alignn/models/load
```

The existing fit, predict, candidates, and ask endpoints are unchanged.

## Tell

`tell` accepts original user-facing structure and category labels. The server
reuses the fitted structure-ID mapping and process category maps before
appending the encoded observation to the underlying optimizer.

```python
import httpx

base = "http://127.0.0.1:8000/api/v1"

response = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/tell",
    json={
        "data": [
            {
                "phase": "beta",
                "temperature": 1080.0,
                "pressure": 1.6,
                "furnace": "B",
                "atmosphere": "Ar",
                "property": 1.25,
            }
        ],
        "refit": True,
    },
    timeout=120.0,
)
response.raise_for_status()
```

For mixed ALIGNN, a newly observed **joint** categorical combination can be
added as long as each category label already exists in the fitted category
maps. After the update it becomes eligible for later mixed candidate
enumeration because Phase 3 candidate generation reads the optimizer's current
training inputs.

## Save

The API writes a synchronized tabular optimizer to the common artifact store.
The storage root is controlled by `BOCHAN_API_MODEL_DIR`.

```python
saved = httpx.post(
    f"{base}/tabular/alignn/models/{model_id}/save",
    json={
        "filename": "alignn-process-model",
        "overwrite": False,
    },
).json()

print(saved["filename"])
# alignn-process-model.bochan.pt
```

The artifact contains the fitted `TabularBayesianOptimizer`, including the
structure adapter/catalog state, pure-PyTorch graph bank, ALIGNN encoder and GP
state, tabular feature layout, category maps, current training inputs, and
observation state held by the optimizer.

## Load

Model artifacts use `torch.load`/pickle and must therefore be treated as trusted
executable Python artifacts. Loading requires explicit opt-in:

```python
loaded = httpx.post(
    f"{base}/tabular/alignn/models/load",
    json={
        "filename": "alignn-process-model.bochan.pt",
        "map_location": "cpu",
        "trust_pickle": True,
    },
).json()

loaded_model_id = loaded["model_id"]
```

Set `trust_pickle=True` only for artifacts produced by a trusted Bochan process.
The ALIGNN-specific load endpoint rejects tabular artifacts whose fitted model
is not `alignn_gp` or `alignn_dkl`, or whose structure contract is missing.

## Restored contract

The load response uses the same ALIGNN-aware metadata as the fit response. In
particular, it exposes:

```text
metadata.alignn.input_type
metadata.alignn.structure_col
metadata.alignn.structure_ids
metadata.alignn.continuous_process_cols
metadata.alignn.categorical_process_cols
metadata.alignn.categorical_process_dims
metadata.alignn.category_maps
metadata.alignn.graph_config
metadata.alignn.encoder_training
metadata.alignn.encoder_initialization
```

This allows an API client to verify that a restored mixed model retained the
same structure IDs, category encodings, and process layout.

## Continue after restore

The returned `model_id` can immediately be used with the normal ALIGNN routes:

```text
POST /api/v1/tabular/alignn/models/{model_id}/predict
POST /api/v1/tabular/alignn/models/{model_id}/candidates
POST /api/v1/tabular/alignn/models/{model_id}/ask
POST /api/v1/tabular/alignn/models/{model_id}/tell
```

The Phase 5 integration test exercises a real pure-PyTorch `TorchGraph` path and
checks `fit -> tell -> save -> load -> predict`, including restoration of mixed
process category maps and structure IDs.

## Scope

Phase 5 does not add another artifact format, Web UI, unknown-structure
generation, or multi-output ALIGNN. It hardens the existing known-structure
FastAPI workflow for long-lived model operation.
