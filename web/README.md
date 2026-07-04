# bochan Web

React + TypeScript frontend for the regression-focused bochan web MVP.

## Development

Start the FastAPI backend from the repository root:

```bash
pip install -e ".[web]"
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`.

## MVP scope

- CSV and Excel upload
- dataset profile and preview
- numeric target selection
- numeric and categorical feature selection
- bounds, grid step, and fixed feature settings
- base GP, MAP-SAAS, and Deep Kernel model selection
- EI, NEI, and UCB candidate generation
- candidate mean, standard deviation, and acquisition value display
- Plotly figures generated through `bochan.visualization`
  - observed-versus-predicted YY plot
  - one-dimensional prediction mean and uncertainty curve
  - two-dimensional prediction contour for the first two numeric search variables
- structured execution logs with request IDs and stage timings

The one-dimensional and two-dimensional figures are generated only when enough non-fixed numeric search variables are available. Visualization failures are reported separately and do not discard the candidate table.

Datasets and fitted models are stored in memory. Restarting FastAPI clears the current session.

## Visualization response

`POST /api/v1/regression/run` returns Plotly-compatible JSON under `visualizations`:

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

The React frontend renders these payloads with `react-plotly.js`.

## Logging

The backend writes human-readable logs to the console and structured JSONL records to:

```text
.bochan/logs/bochan-web.jsonl
```

Log files rotate at 10 MiB and retain five backups by default. Configure logging with environment variables:

```bash
set BOCHAN_LOG_LEVEL=DEBUG
set BOCHAN_LOG_DIR=C:\logs\bochan
set BOCHAN_LOG_MAX_BYTES=20971520
set BOCHAN_LOG_BACKUP_COUNT=10
```

For bash or PowerShell, use the corresponding environment-variable syntax.

Each HTTP response includes an `X-Request-ID` header. The same ID is attached to dataset, model fitting, candidate generation, prediction, visualization, completion, and error records.

Recent records are available from:

```text
GET /api/v1/logs?limit=200
GET /api/v1/logs?request_id=<request-id>
GET /api/v1/logs?level=ERROR
GET /api/v1/logs?event=model_fit_completed
```

The result page automatically locates the most recent regression request ID and displays its execution log. The regression response also includes stage timings in `metadata.timings_ms`.
