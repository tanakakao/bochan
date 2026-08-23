# CrabNet-GP / CrabNet-DKL in FastAPI and Web

Phase 9 exposes the Phase 8 tabular selectors through the HTTP and React
workbench layers without introducing additional model aliases:

- `model_type="crabnet_gp"` — frozen CrabNet encoder plus an exact Gaussian GP.
- `model_type="crabnet_dkl"` — CrabNet encoder fine-tuned jointly with the GP.

The supported data shape is one formula column, zero or more continuous process
columns, and one continuous regression target. Candidate responses preserve the
formula and every process column.

## Installation

For the tabular FastAPI endpoints, install all three optional groups:

```bash
pip install -e ".[api,tabular,materials]"
```

The Web extra includes the pinned CrabNet dependency:

```bash
pip install -e ".[web]"
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

## Tabular FastAPI request

`composition_sites` is the only formula configuration entry point. The
checkpoint is optional and must be a path readable by the FastAPI server.

```json
{
  "data": [
    {
      "formula": "Ba0.45Sr0.15Ti0.10O0.30",
      "temperature": 1000.0,
      "holding_time": 1.5,
      "property": 0.4
    }
  ],
  "input_cols": ["formula", "temperature", "holding_time"],
  "target_cols": ["property"],
  "bounds": {
    "temperature": [950.0, 1400.0],
    "holding_time": [1.0, 10.0]
  },
  "composition_sites": {
    "formula": {
      "column": "formula",
      "elements": ["Ba", "Sr", "Ti", "O"],
      "representation": "ilr",
      "coordinate_bounds": [-3.0, 3.0],
      "bounds": {
        "Ba": [0.05, 0.70],
        "Sr": [0.05, 0.70],
        "Ti": [0.05, 0.70],
        "O": [0.05, 0.80]
      }
    }
  },
  "model_config": {
    "task_type": "regression",
    "model_type": "crabnet_dkl",
    "model_kwargs": {
      "checkpoint": "/srv/checkpoints/crabnet.pth",
      "encoder_training": "partial"
    }
  },
  "fit_config": {"num_epochs": 32, "lr": 0.001}
}
```

Send this payload to `POST /api/v1/tabular/models`. For CrabNet-DKL,
`encoder_training` accepts only `partial` or `full`; omission resolves to the
recommended `partial` mode. CrabNet-GP always freezes its encoder and therefore
rejects `encoder_training`.

The fit response retains the normal `TabularModelFitResponse` shape and adds
inspectable `metadata.crabnet` fields for the effective encoder policy,
initialization source, composition column, and process dimension. It does not
return the checkpoint path.

Generate candidates with the fitted model id:

```json
{
  "acquisition_config": {"name": "logei"},
  "optimize_config": {
    "q": 2,
    "num_restarts": 4,
    "raw_samples": 64,
    "sequential": true
  }
}
```

`POST /api/v1/tabular/models/{model_id}/candidates` returns records containing
`formula`, `temperature`, and `holding_time` rather than internal ILR columns.

## React workbench

1. On **Select**, choose the formula as an explanatory column and set its input
   notation to **組成式**. Keep process columns numeric.
2. On **Model**, choose the **深層・表現学習** family and then **CrabNet-GP** or
   **CrabNet-DKL**. The existing composition control continues to own the element
   vocabulary and representation.
3. Optionally enter a server-readable checkpoint path. For CrabNet-DKL, choose
   **Partial (recommended)** or **Full** encoder training.
4. Generate candidates normally. Results and CSV export retain the formula and
   process columns.

The model-reuse fingerprint and saved-project request include the checkpoint,
encoder policy, and model-affecting composition settings. Importing a saved model
restores those controls.

## Explicitly unsupported combinations

FastAPI and Web reject these combinations before or during request preparation:

- classification, ordinal, hybrid, or multiple target columns;
- multiple composition sites;
- categorical process columns;
- independent composition descriptor columns;
- input perturbation;
- `encoder_training` on `crabnet_gp`;
- values other than `partial` or `full` on `crabnet_dkl`;
- the low-level `trainable_encoder_layers` control at the HTTP boundary.

Use the Python API when direct module injection or a numeric
`trainable_encoder_layers` value is required.
