# FastAPI reference

bochan exposes HTTP / JSON APIs for generic Bayesian optimization, tabular workflows, material residual models, and unified MLIP workflows.

This page provides the current high-level serving map. Backend-specific pages remain available for specialized details, but new integrations should prefer the canonical endpoints described here.

## Running the API

Install serving dependencies:

```bash
pip install -e ".[api]"
```

Run the application using the project FastAPI entry point used by your deployment. Once running, OpenAPI documentation is available through the standard FastAPI `/docs` and `/openapi.json` surfaces.

## API groups

### Generic optimizer and model APIs

The generic serving layer covers model fit/load/save, prediction, acquisition construction, and candidate generation. These endpoints mirror the same concepts used by `BayesianOptimizer` and the lower-level `build_model -> fit_model -> build_acquisition -> optimize_candidates` flow.

### Tabular APIs

Tabular endpoints work with JSON records corresponding to DataFrame-like data and preserve named feature/output columns. Use these when the caller naturally works with tables rather than tensors.

### Material residual APIs

The fitted residual-GP lifecycle is exposed under:

```text
POST /api/v1/tabular/material-residual/models
POST /api/v1/tabular/material-residual/models/{model_id}/tell
POST /api/v1/tabular/material-residual/models/{model_id}/predict
POST /api/v1/tabular/material-residual/models/{model_id}/candidates
POST /api/v1/tabular/material-residual/models/{model_id}/ask
POST /api/v1/tabular/material-residual/models/{model_id}/save
POST /api/v1/tabular/material-residual/models/load
```

This surface owns fit/store/predict/candidate lifecycle for pretrained material baseline + GP residual models. The unified MLIP workflow API does not duplicate those persistence responsibilities.

## Unified MLIP capability discovery

### List all capabilities

```text
GET /api/v1/materials/mlip/capabilities
```

The response contains supported backends, physical quantities, model modes, workflow modes, and backend-specific constraints.

### Inspect one backend

```text
GET /api/v1/materials/mlip/capabilities/{backend}
```

Backend aliases are normalized. For example, `alignn_ff` resolves to `alignn-ff`.

## Validate an MLIP workflow

```text
POST /api/v1/materials/mlip/workflows/validate
```

Example request:

```json
{
  "backend": "alignn_ff",
  "quantity": "energy",
  "model_mode": "residual-gp",
  "workflow_mode": "bo"
}
```

Example canonical response:

```json
{
  "valid": true,
  "spec": {
    "backend": "alignn-ff",
    "quantity": "energy",
    "model_mode": "residual_gp",
    "workflow_mode": "relax_acquisition"
  },
  "requirements": [
    "structures",
    "train_X",
    "train_Y",
    "structure_graphs"
  ]
}
```

This endpoint is dependency-light: it normalizes and validates the specification without importing or downloading an MLIP model.

## Configure a relaxation workflow

```text
POST /api/v1/materials/mlip/workflows/configure
```

Example request:

```json
{
  "backend": "mace",
  "quantity": "energy",
  "model_mode": "direct",
  "workflow_mode": "relax_rank",
  "relaxation": {
    "optimizer": "FIRE",
    "fmax": 0.05,
    "max_steps": 200,
    "relax_cell": false
  }
}
```

Common relaxation settings are:

| Field | Type | Default |
|---|---|---:|
| `optimizer` | `FIRE`, `BFGS`, or `LBFGS` | `FIRE` |
| `fmax` | positive float | `0.05` |
| `max_steps` | positive integer | `200` |
| `relax_cell` | boolean | `false` |

For `relax_rank` and `relax_acquisition`, omitted relaxation settings receive these defaults. `model_only` does not accept relaxation settings.

## Execute structure relaxation

```text
POST /api/v1/materials/mlip/workflows/execute/relaxation
```

This is the runtime endpoint. Unlike `validate` and `configure`, it lazily creates the selected real backend and executes ASE relaxation.

A request contains:

- workflow identity;
- one or more periodic structures;
- optional common relaxation settings;
- optional backend constructor options, such as a model name or device.

The endpoint supports MACE, CHGNet, M3GNet, and ALIGNN-FF through a common contract and returns serialized relaxation results containing structure, energy, force, stress, convergence, and backend metadata.

Current runtime behavior:

- `relax_rank` and `relax_acquisition` identities are accepted because they contain a relaxation stage;
- `model_only` is rejected by this endpoint;
- requests are limited to 100 structures;
- missing optional MLIP dependencies return HTTP 503;
- invalid input/runtime contracts return HTTP 422;
- backend runtime failures return HTTP 400.

## Choosing the right material API

| Need | Use |
|---|---|
| Discover supported MLIP combinations | `/materials/mlip/capabilities` |
| Validate UI/API selections without loading models | `/materials/mlip/workflows/validate` |
| Canonicalize relaxation settings | `/materials/mlip/workflows/configure` |
| Actually relax structures with a real MLIP | `/materials/mlip/workflows/execute/relaxation` |
| Fit and store a residual GP | `/tabular/material-residual/models` |
| Predict with a stored residual GP | `/tabular/material-residual/models/{model_id}/predict` |
| Generate BO candidates from a stored residual GP | `/tabular/material-residual/models/{model_id}/candidates` or `/ask` |

## Backend-specific references

For older backend-specific request examples and implementation details, see:

- [ALIGNN FastAPI](../alignn_fastapi.md)
- [CHGNet FastAPI](../chgnet_fastapi.md)
- [M3GNet FastAPI](../m3gnet_fastapi.md)
- [MACE FastAPI](../mace_fastapi.md)
- [CrabNet FastAPI / Web](../crabnet_fastapi_web.md)

These pages are useful supplementary references, while this page defines the preferred top-level serving map.
