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

The one-dimensional and two-dimensional figures are generated only when enough non-fixed numeric search variables are available. Visualization failures are reported separately and do not discard the candidate table.

Datasets and fitted models are stored in memory. Restarting FastAPI clears the current session.
