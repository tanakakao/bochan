# bochan Web

`bochan Web` is a browser-based interface for the `bochan` Bayesian optimization library.
The frontend is implemented with React, TypeScript, and Vite. The backend is a FastAPI application that reuses the existing `bochan` model, candidate-generation, tabular-data, and visualization implementations.

The current Web MVP focuses on single-objective regression. It is intended for local experimentation and internal use while the persistent project and asynchronous-job architecture is still under development.

---

## 1. What the Web application can do

The current implementation supports the following workflow:

1. Upload a CSV or Excel file.
2. Inspect the detected columns and a preview of the data.
3. Select one numeric target column.
4. Select numeric and categorical feature columns.
5. Configure search bounds, grid steps, and fixed variables.
6. Select a Gaussian-process model and acquisition function.
7. Fit the model and generate the next experimental candidates.
8. Inspect predicted means, standard deviations, acquisition values, and Plotly figures.
9. Review structured execution logs for the latest optimization request.

### Current regression MVP scope

| Category | Supported options |
|---|---|
| Task | Single-objective regression |
| Input files | CSV, XLSX, XLS |
| Feature types | Numeric, categorical |
| Target type | Numeric |
| Direction | Maximize, minimize |
| Models | Base GP, MAP-SAAS, Deep Kernel GP |
| Acquisition functions | EI, NEI, UCB |
| Candidate optimizer | `optimize_acqf` |
| Search settings | Lower bound, upper bound, step, fixed value |
| Output | Candidate table, prediction uncertainty, acquisition value, Plotly figures, execution logs |

The browser UI does not yet expose multi-objective optimization, classification, active learning, ordinal regression, robust optimization, a visual constraint editor, persistent projects, or asynchronous training jobs.

---

## 2. Architecture

```text
Browser
  React + TypeScript + Vite
          |
          | HTTP / JSON
          v
FastAPI
  bochan.serving.webapp
          |
          +-- dataset loading and profiling
          |     bochan.desktop.services
          |
          +-- model fitting and candidate generation
          |     bochan.api.BayesianOptimizer
          |
          +-- result figures
          |     bochan.visualization
          |
          +-- structured logs
                .bochan/logs/bochan-web.jsonl
```

The existing tensor-oriented FastAPI endpoints remain available in the same process. The Web-specific endpoints use the `/api/v1` prefix.

### Relevant directories

```text
src/bochan/serving/webapp/
├── __init__.py
├── app.py          # FastAPI app, routes, request middleware
├── logging.py      # JSONL logging, rotation, request IDs
└── workflows.py    # Regression workflow and visualization orchestration

web/
├── index.html
├── package.json
├── vite.config.ts
└── src/
    ├── App.tsx
    ├── api.ts
    ├── ExecutionLogs.tsx
    ├── ResultVisualizations.tsx
    ├── styles.css
    └── types.ts
```

---

## 3. Requirements

### Backend

- Python 3.11 or later
- A working `bochan` development installation
- FastAPI and Uvicorn
- pandas
- scikit-learn
- Plotly
- openpyxl for Excel files

### Frontend

- Node.js 18 or later is recommended
- npm

---

## 4. Installation

Run these commands from the repository root.

### Backend dependencies

```bash
pip install -e ".[web]"
```

The `web` optional dependency group includes the packages needed by the FastAPI Web backend, tabular loading, and Plotly visualization.

### Frontend dependencies

```bash
cd web
npm install
```

Return to the repository root before starting the backend, or open a second terminal.

---

## 5. Starting the application

Two processes are required during development.

### Terminal 1: FastAPI backend

From the repository root:

```bash
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

Useful backend URLs:

```text
Web API health:   http://127.0.0.1:8000/api/v1/health
OpenAPI UI:       http://127.0.0.1:8000/docs
OpenAPI JSON:     http://127.0.0.1:8000/openapi.json
```

### Terminal 2: React frontend

```bash
cd web
npm run dev
```

Open:

```text
http://localhost:5173
```

Vite proxies `/api`, `/health`, `/models`, and `/acquisitions` to `http://127.0.0.1:8000` during local development.

### Windows command prompt example

```bat
rem Terminal 1
pip install -e ".[web]"
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

```bat
rem Terminal 2
cd web
npm install
npm run dev
```

### PowerShell example

```powershell
# Terminal 1
pip install -e ".[web]"
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

```powershell
# Terminal 2
Set-Location web
npm install
npm run dev
```

---

## 6. Browser workflow

### Step 1: Upload data

Select a CSV or Excel file. The browser converts the selected file to a data URL and sends it to FastAPI. The backend decodes the file, creates a pandas DataFrame, profiles each column, and stores the DataFrame in an in-memory dataset store.

The upload response includes:

- generated dataset ID;
- file name and source type;
- row and column counts;
- column types;
- missing-value counts and rates;
- unique-value counts;
- numeric summary values where available;
- a preview of the first rows.

#### CSV assumptions

- Default encoding: `utf-8-sig`
- Delimiter: automatically inferred unless explicitly supplied through the API
- Column names should be unique

#### Excel assumptions

- The browser UI currently reads the first sheet (`sheet_name=0`)
- `openpyxl` must be installed

### Step 2: Select variables

Select:

- one numeric target column;
- one or more feature columns.

The initial UI selection uses the last detected numeric column as the target and selects the remaining numeric and categorical columns as features. Review this automatically generated selection before fitting the model.

Rows with missing values in the selected feature or target columns are removed by the current browser workflow.

### Step 3: Configure the optimization

#### Direction

- `maximize`: search for larger target values;
- `minimize`: internally multiply the target by `-1` for optimization, then convert displayed predictions back to the original scale.

#### Model

| UI label | `model_type` | Intended use |
|---|---|---|
| Base GP | `base` | Standard Gaussian-process regression |
| MAP-SAAS | `saas` | High-dimensional problems where only some variables are expected to matter |
| Deep Kernel GP | `deepkernel` | Nonlinear learned feature representation followed by a GP |

`fit maxiter` controls the maximum fitting iterations passed to the existing `FitConfig`.

#### Acquisition function

| Name | Meaning | Typical behavior |
|---|---|---|
| EI | Expected Improvement | Balances predicted improvement and uncertainty |
| NEI | Noisy Expected Improvement | Suitable when observations contain noise |
| UCB | Upper Confidence Bound | Explicit exploration–exploitation control using `beta` |

For UCB, larger `beta` values increase the contribution from uncertainty and generally encourage exploration.

#### Candidate-generation settings

| Setting | Meaning |
|---|---|
| `q` | Number of candidates generated in one request |
| `num_restarts` | Number of local optimization restarts |
| `raw_samples` | Number of raw initial samples used by acquisition optimization |
| `sequential` | Currently fixed to `true` in the browser client |

Increasing `num_restarts` and `raw_samples` can improve acquisition optimization but increases runtime.

### Search-variable settings

#### Numeric variable

- `lower`: lower search bound;
- `upper`: upper search bound;
- `step`: optional grid step for candidate repair;
- `fixed`: exclude the variable from optimization and use a fixed value;
- `fixed_value`: value used when fixed.

By default, numeric bounds are initialized from the observed minimum and maximum in the uploaded data.

#### Categorical variable

Categories are inferred from the uploaded data and encoded internally. Candidate results are decoded back to the original category labels before being returned to the browser.

A categorical variable can also be fixed to one category.

### Step 4: Review the results

The result screen shows:

- task and model summary;
- best observed target value;
- candidate table;
- predicted target mean;
- predicted target standard deviation;
- acquisition value;
- result visualizations;
- execution logs.

---

## 7. Result visualizations

Figures are generated on the FastAPI side through `bochan.visualization`, serialized with Plotly, and rendered in React with `react-plotly.js`.

### Observed versus predicted plot

Generated with `show_yyplot`.

It compares the training target values with model predictions and includes predictive uncertainty. Candidate predictions are also included when supported by the visualization helper.

### One-dimensional prediction plot

Generated when at least one non-fixed numeric feature is available.

The Web workflow selects the first available non-fixed numeric feature and uses:

- `grid_1d_plot` to calculate the prediction grid;
- `show_1dplot_with_pred` to render the prediction mean, uncertainty, observed data, and candidates.

Other features are fixed by the existing visualization utility's representative-value behavior.

### Two-dimensional prediction contour

Generated when at least two non-fixed numeric features are available.

The Web workflow selects the first two available non-fixed numeric features and uses:

- `grid_2d` to calculate the prediction surface;
- `show_scatter_with_acqf` to render the contour, observed data, and candidates.

The grid currently uses `n=40`. Increasing the grid density would increase visualization cost substantially because model predictions are evaluated over the full grid.

### Visualization failure behavior

Visualization errors do not discard successful candidate generation. The API returns the candidate table together with entries in `visualization_warnings`, and the browser displays those warnings above the available figures.

### Visualization response format

`POST /api/v1/regression/run` returns Plotly-compatible JSON:

```json
{
  "visualizations": [
    {
      "id": "yyplot",
      "title": "実測値と予測値",
      "description": "...",
      "figure": {
        "data": [],
        "layout": {}
      }
    }
  ],
  "visualization_warnings": []
}
```

---

## 8. Structured logging

The Web backend records operational logs in two forms:

1. human-readable console output;
2. structured JSON Lines output.

### Default log file

```text
.bochan/logs/bochan-web.jsonl
```

The `.bochan/` directory is excluded by `.gitignore`.

### Rotation defaults

| Setting | Default |
|---|---|
| Maximum active file size | 10 MiB |
| Number of rotated backups | 5 |
| Default level | `INFO` |

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `BOCHAN_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` | `INFO` |
| `BOCHAN_LOG_DIR` | Output directory | `.bochan/logs` |
| `BOCHAN_LOG_MAX_BYTES` | Maximum bytes per log file | `10485760` |
| `BOCHAN_LOG_BACKUP_COUNT` | Number of rotated files retained | `5` |

Windows command prompt:

```bat
set BOCHAN_LOG_LEVEL=DEBUG
set BOCHAN_LOG_DIR=C:\logs\bochan
set BOCHAN_LOG_MAX_BYTES=20971520
set BOCHAN_LOG_BACKUP_COUNT=10
```

PowerShell:

```powershell
$env:BOCHAN_LOG_LEVEL = "DEBUG"
$env:BOCHAN_LOG_DIR = "C:\logs\bochan"
$env:BOCHAN_LOG_MAX_BYTES = "20971520"
$env:BOCHAN_LOG_BACKUP_COUNT = "10"
```

Bash:

```bash
export BOCHAN_LOG_LEVEL=DEBUG
export BOCHAN_LOG_DIR=/var/log/bochan
export BOCHAN_LOG_MAX_BYTES=20971520
export BOCHAN_LOG_BACKUP_COUNT=10
```

### Request IDs

Every HTTP request receives a request ID.

- A client may send an `X-Request-ID` header.
- When absent, FastAPI generates a UUID-like hexadecimal ID.
- The response includes `X-Request-ID`.
- All logs emitted during that request include the same ID.
- Frontend API errors include the request ID so that an error can be matched to server logs.

Example error displayed in the browser:

```text
Model fitting failed [request_id=4f8d...]
```

### Logged workflow stages

Representative events include:

```text
http_request_started
http_request_completed
http_request_failed
dataset_load_started
dataset_load_completed
dataset_load_failed
regression_run_requested
workflow_started
workflow_data_prepared
model_fit_started
model_fit_completed
model_fit_failed
candidate_generation_started
candidate_generation_completed
candidate_generation_failed
candidate_prediction_completed
visualization_created
visualization_failed
visualization_batch_completed
workflow_completed
regression_run_completed
regression_run_failed
```

Each stage records relevant structured fields such as model type, acquisition name, candidate count, row count, feature count, and elapsed time.

### Example JSONL record

```json
{
  "timestamp": "2026-07-04T03:00:00.000000+00:00",
  "level": "INFO",
  "logger": "bochan.web.workflow",
  "message": "Model fitting completed",
  "request_id": "4f8d...",
  "event": "model_fit_completed",
  "model_type": "base",
  "duration_ms": 853.4
}
```

### Log API

```text
GET /api/v1/logs
```

Supported query parameters:

| Parameter | Meaning |
|---|---|
| `limit` | Number of recent records, from 1 to 1000 |
| `level` | Exact log-level filter |
| `event` | Exact event-name filter |
| `request_id` | Exact request-ID filter |

Examples:

```text
GET /api/v1/logs?limit=200
GET /api/v1/logs?request_id=<request-id>
GET /api/v1/logs?level=ERROR
GET /api/v1/logs?event=model_fit_completed
```

Example response:

```json
{
  "entries": [],
  "count": 0,
  "log_file": ".bochan/logs/bochan-web.jsonl"
}
```

The result page reads this endpoint and displays records associated with the most recent completed or failed regression request.

### Security note

The log endpoint is currently unauthenticated and may expose file paths, model settings, column names, and exception text. It is suitable for local or controlled internal deployments only. Add authentication or disable the endpoint before exposing the service publicly.

---

## 9. Timing information in regression results

The regression response contains stage timings under `metadata.timings_ms`.

```json
{
  "metadata": {
    "request_id": "4f8d...",
    "timings_ms": {
      "prepare": 12.3,
      "fit": 853.4,
      "candidate": 241.8,
      "prediction": 4.2,
      "visualization": 87.6,
      "total": 1201.5
    }
  }
}
```

The values are wall-clock measurements collected inside the FastAPI process. They are intended for operational diagnostics rather than precise benchmarking.

---

## 10. Web API endpoints

### Web-specific endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Web application health check |
| `GET` | `/api/v1/capabilities` | List the currently exposed Web features |
| `GET` | `/api/v1/datasets` | List in-memory datasets |
| `POST` | `/api/v1/datasets` | Upload and profile CSV or Excel data |
| `GET` | `/api/v1/datasets/{dataset_id}` | Get dataset profile and preview |
| `POST` | `/api/v1/regression/run` | Fit, generate candidates, predict, visualize, and return timings |
| `GET` | `/api/v1/logs` | Read recent structured logs |

The original tensor-oriented serving endpoints are also mounted, including `/models`, prediction endpoints, candidate endpoints, and acquisition-name endpoints.

### Upload request example

```json
{
  "source_type": "csv",
  "name": "experiment.csv",
  "content_base64": "data:text/csv;base64,...",
  "encoding": "utf-8-sig",
  "sep": null,
  "sheet_name": 0
}
```

### Regression request example

```json
{
  "dataset_id": "dataset-id",
  "feature_columns": ["temperature", "pressure", "machine"],
  "target_column": "yield",
  "direction": "maximize",
  "model_type": "base",
  "model_kwargs": {},
  "fit_maxiter": 128,
  "normalize": true,
  "outcome_transform": true,
  "input_perturbation": false,
  "n_w": 16,
  "perturbation_std": 0.1,
  "search_space": [
    {
      "name": "temperature",
      "type": "numeric",
      "lower": 100.0,
      "upper": 200.0,
      "step": 5.0,
      "fixed": false
    },
    {
      "name": "machine",
      "type": "categorical",
      "categories": ["A", "B", "C"],
      "fixed": false
    }
  ],
  "constraints": [],
  "k_sparse": null,
  "acquisition": {
    "name": "EI",
    "beta": 2.0,
    "acqf_kwargs": {}
  },
  "optimizer": {
    "name": "optimize_acqf",
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 256,
    "sequential": true
  },
  "drop_missing": true
}
```

---

## 11. State and persistence

The current application stores uploaded datasets and fitted optimizer objects in the FastAPI process.

Consequences:

- restarting FastAPI clears the current session;
- multiple Uvicorn workers do not share datasets or fitted models;
- horizontal scaling is not supported;
- the current implementation should not be treated as a durable experiment database.

A production-oriented version should separate:

- project and experiment metadata;
- uploaded dataset storage;
- fitted model artifacts;
- background-job state;
- generated reports and candidate exports.

---

## 12. Development commands

### Frontend development server

```bash
cd web
npm run dev
```

### Frontend production build

```bash
cd web
npm run build
```

The build output is written to `web/dist/`.

### Preview the production build

```bash
cd web
npm run preview
```

### Backend tests

From the repository root:

```bash
pytest tests/test_webapp_api.py
```

The Web tests cover health responses, request ID propagation, capabilities, recent-log access, JSON log formatting, and Plotly serialization.

---

## 13. Troubleshooting

### The browser cannot reach FastAPI

Check that:

- Uvicorn is running on port 8000;
- Vite is running on port 5173;
- the browser is opened at `http://localhost:5173`;
- another process is not occupying either port.

Directly test:

```text
http://127.0.0.1:8000/api/v1/health
```

### CORS error

The backend currently permits these frontend origins:

```text
http://localhost:5173
http://127.0.0.1:5173
```

Using another host or port requires updating the CORS settings in `bochan.serving.webapp.app`.

### Excel upload fails

Install the Web dependencies again:

```bash
pip install -e ".[web]"
```

Confirm that `openpyxl` is installed and that the workbook is not password protected.

### CSV columns are parsed incorrectly

Check:

- file encoding;
- delimiter;
- duplicate column names;
- inconsistent row lengths;
- values that mix numeric and string formats in the same column.

The browser currently sends `utf-8-sig`. Use the API directly when another encoding or separator is required.

### Model fitting fails

Inspect:

- the error message and request ID shown in the browser;
- the execution-log panel;
- `.bochan/logs/bochan-web.jsonl`;
- whether enough valid rows remain after missing-value deletion;
- whether the target is numeric;
- whether numeric bounds have positive width;
- whether categorical values in the search space match uploaded categories.

### Candidate generation is slow

Reduce one or more of:

- `q`;
- `num_restarts`;
- `raw_samples`;
- fitting iterations;
- model complexity.

MAP-SAAS and Deep Kernel models can require substantially more computation than a base GP.

### Figures are missing but candidates exist

Check `visualization_warnings` and the execution logs. Candidate generation and visualization are intentionally isolated so that a plotting failure does not discard optimization results.

### Log file is not created

Check that the process can create and write to `BOCHAN_LOG_DIR`. When no directory is configured, the current working directory must allow creation of `.bochan/logs`.

---

## 14. Current limitations

- Regression only in the browser UI
- One target column
- Synchronous model fitting and candidate generation
- In-memory state
- No authentication or authorization
- Unauthenticated log endpoint
- No upload-size policy in the current MVP
- No persistent dataset or model artifact storage
- No visual linear-constraint or k-sparse editor
- No user-selectable variables for the 1D and 2D plots; the first available numeric variables are used
- No server-sent progress updates during long fitting operations
- No frontend tests yet

---

## 15. Planned extensions

Likely next implementation steps are:

1. persistent projects and experiment records;
2. background jobs with progress and cancellation;
3. saved model and candidate artifacts;
4. CSV export and experiment-result import;
5. linear, equality, and k-sparse constraint editors;
6. multi-objective optimization;
7. active learning and level-set estimation;
8. binary, multiclass, and ordinal workflows;
9. robust and heteroscedastic optimization;
10. authentication and deployment configuration.
