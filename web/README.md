# bochan Web

`bochan Web` is the React/FastAPI interface for the `bochan` Bayesian optimization toolkit.
The browser client lives in `web/`; the Python backend lives under `bochan.serving` and reuses the same tabular, modeling, acquisition, optimization, inspection, and visualization layers as the Python API.

## Architecture

```text
Browser
  React + TypeScript + Vite
          |
          | HTTP / JSON
          v
FastAPI routes and Web orchestration
  bochan.serving.webapp
          |
          +-- dataset storage / loading / profiling
          |     bochan.serving.workbench.datasets
          |
          +-- shared Web encoding / repair helpers
          |     bochan.serving.workbench.workflow_utils
          |
          +-- tabular model fitting
          |     bochan.tabular
          |
          +-- Bayesian optimization / active learning
          |     bochan.api
          |
          +-- models and acquisitions
          |     bochan.models / bochan.acquisition
          |
          +-- diagnostics and figures
                bochan.inspection / bochan.visualization
```

`bochan.serving.webapp` owns HTTP schemas, routes, Web workflow orchestration, project/model artifacts, visualization sessions, and logging.
`bochan.serving.workbench` contains transport-independent backend support used by those Web workflows. It deliberately has no FastAPI application import side effects.

## Main directories

```text
src/bochan/serving/
├── fastapi/                 # shared tensor-oriented API transport
├── webapp/                  # React-facing routes and Web workflow orchestration
└── workbench/               # dataset services and shared Web workflow helpers

web/
├── public/
├── src/
├── test/
├── package.json
└── vite.config.ts
```

## Supported Web workflow

The Web application currently provides:

- CSV and Excel upload with in-memory dataset profiling;
- regression, classification, ordinal, hybrid, and multi-objective workflows;
- Bayesian optimization, active learning, and level-set estimation paths;
- numeric, categorical, mixed, and composition-aware inputs;
- GP, deep, multitask, tree-ensemble, and supported foundation/external surrogate models;
- gradient and derivative-free candidate-search backends where supported;
- feature constraints, target constraints, fixed variables, grid steps, and candidate repair;
- cross-validation, feature importance, model diagnostics, and Plotly visualizations;
- input-perturbation / risk-aware workflows where supported;
- model-artifact import/export and experiment-project archives;
- structured JSONL execution logging.

The authoritative runtime capability list is available from `GET /api/v1/capabilities`.

## Installation

From the repository root:

```bash
pip install -e ".[web]"
cd web
npm install
```

The `web` optional dependency group installs the Python dependencies required by the Web backend, including FastAPI, pandas, scikit-learn, Plotly, Excel support, and the Web-supported optional surrogate packages.

## Development startup

Start the backend from the repository root:

```bash
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
cd web
npm run dev
```

Then open `http://localhost:5173`.

On Windows, the repository-level `start_web.bat` can be used for the configured local startup flow.

## Data handling

Browser uploads currently accept CSV and Excel files. Uploaded data is held in an in-memory `DatasetStore`; the Web backend does not use the removed desktop data-source layer.

Model artifacts can restore fitted Web sessions, and project archives can persist dataset lineage, workbench settings, experiment history, and selected model artifacts.

## API and logs

Useful endpoints include:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/capabilities` | Runtime Web capability metadata |
| `GET` | `/api/v1/datasets` | List in-memory datasets |
| `POST` | `/api/v1/datasets` | Upload CSV or Excel data |
| `GET` | `/api/v1/datasets/{dataset_id}` | Dataset profile and preview |
| `POST` | `/api/v1/regression/run` | Run the tabular Web workflow |
| `POST` | `/api/v1/runs/{run_id}/visualizations` | Build a visualization from a retained session |
| `GET` | `/api/v1/runs/{run_id}/model-artifact` | Download a fitted model artifact |
| `POST` | `/api/v1/model-artifacts/import` | Import a model artifact or project archive |
| `GET` | `/api/v1/logs` | Read recent structured Web logs |

The default structured log file is:

```text
.bochan/logs/bochan-web.jsonl
```

Each HTTP request receives an `X-Request-ID`, which is included in the response and associated structured log records.

## Design boundary

Core optimization and model code must not depend on the Web transport. Web-specific concerns belong in `bochan.serving.webapp`; reusable Web-backend support belongs in `bochan.serving.workbench`. This boundary keeps the browser application replaceable without coupling `bochan.api`, `bochan.models`, or `bochan.tabular` to FastAPI or React.
