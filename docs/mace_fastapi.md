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

## Large structure catalogs

MACE participates in the same generic structure-aware candidate backend as
ALIGNN, CHGNet, and M3GNet. Structure selection is always feature 0 and remains
separate from process categorical variables.

For more than 10 candidate structures, `q=1`, `return_best_only=true`, and the
standard BoTorch optimizer path, bochan switches from full structure/process
enumeration to `optimize_structure_alternating`. The structure ID is optimized
categorically while observed joint process-category assignments remain feasible.

For `q>1`, exact enumeration is retained so separate batch slots may choose
different structure/category assignments without changing batch semantics.
Callers continue to use `structure_ids` to restrict the search space; the scaling
backend is selected internally.

## Acquisition coverage

Phase 7 closes the structure/process optimization path for the Gaussian MACE
family across the main acquisition modes used by bochan:

- single-objective LogEI / qLogEI-style structure-process optimization;
- single-objective UCB structure-process optimization;
- mixed categorical process variables plus discrete structure selection;
- independent multi-output NEHVI through `ModelListGP`;
- correlated MACE multitask NEHVI with one shared representation and task covariance.

The structure selector and process categorical columns are discrete, so they are
handled by fixed-feature enumeration or the alternating backend. Continuous
process dimensions retain acquisition gradients through the GP or DKL path.

## Ask / tell state contract

`/ask` registers returned candidates as pending observations. A matching `/tell`
resolves the pending row in canonical encoded model-input space rather than
creating a duplicate observation. Completed replicates at an already observed X
remain independent observations when there is no matching pending row.

The MACE FastAPI surface uses the same observation-state and artifact helpers as
other tabular structure models, so observed, failed, and pending masks survive
save/load round trips.

## Frozen representation cache

Frozen `mace_gp` and `mace_multitask` models cache the complete MACE structure
representation bank. Repeated posterior and acquisition evaluations reuse the
cached invariant descriptors. Loading model state invalidates the non-persistent
cache and rebuilds it on the next feature request, preventing stale descriptors
from surviving a state change.

DKL variants disable the frozen structure cache because trainable MACE backbone
layers must be reevaluated as their weights change.

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

## Phase 7 CI contract

`.github/workflows/mace-phase7-smoke.yml` validates the deterministic MACE
optimization matrix: large-catalog scaling, batch exact enumeration, LogEI, UCB,
mixed process categories, independent and correlated NEHVI, cache invalidation,
FastAPI regressions, and Ruff.

The official pretrained `medium-mpa-0` end-to-end FastAPI/posterior path remains
covered by the Phase 6 workflow.

## Runnable client

```bash
python -m pip install -e ".[api,tabular,materials]"
python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000
python examples/mace_fastapi_client.py
```
