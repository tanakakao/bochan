# bochan desktop regression MVP

`bochan.desktop` is a local desktop shell for regression-only Bayesian optimization workflows.
It is intended as the first desktop-app layer on top of `bochan.api`.

The current MVP supports:

- CSV input
- Excel input
- SQLite / DuckDB query input
- data preview and column profiling
- explanatory-variable and target-variable selection
- single-output regression only
- `model_type` selection for registered regression models
- numeric search ranges
- per-variable step rounding
- fixed features
- linear constraints
- k-sparse post-processing
- regression acquisition selection
- candidate table display
- a simple candidate mean chart
- candidate JSON export

The implementation keeps state in memory. It is meant for local interactive use, not a long-running shared service.

---

## Install

```bash
pip install -e ".[desktop]"
```

For the normal development setup:

```bash
pip install -e ".[dev,desktop,evo]"
```

`DuckDB` is included in the desktop optional dependency. SQLite uses Python's standard library.

---

## Launch as a desktop window

```bash
python -m bochan.desktop
```

This starts a local FastAPI server and opens a `pywebview` window.

---

## Launch as a browser app

```bash
uvicorn bochan.desktop.app:app --reload --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

---

## Workflow

1. Load a CSV, Excel file, or SQL query result.
2. Select feature columns and one numeric target column.
3. Configure search ranges, steps, and fixed values.
4. Optionally configure linear constraints and k-sparse settings as JSON.
5. Select a regression model and acquisition function.
6. Run candidate generation.
7. Review candidates, predicted mean / standard deviation, acquisition value, and constraint status.

---

## Constraint JSON example

```json
[
  {
    "name": "total_le_1",
    "terms": [
      {"column": "x1", "coefficient": 1.0},
      {"column": "x2", "coefficient": 1.0}
    ],
    "sense": "le",
    "rhs": 1.0,
    "enabled": true
  }
]
```

Supported `sense` values are:

```text
le
ge
eq
```

---

## k-sparse JSON example

```json
{
  "enabled": true,
  "columns": ["x1", "x2", "x3", "x4"],
  "k": 2,
  "score": "abs",
  "support_selection": "topk",
  "final_priority": "constraints"
}
```

---

## Current limitations

- Only single-output regression is exposed in the desktop UI.
- Classification, ordinal, hybrid, and multi-objective workflows are intentionally hidden for the first MVP.
- The project is stored only in memory for now. Reloading the app clears fitted models and loaded datasets.
- Constraint repair is best-effort and candidates also report post-generation constraint status.
- The embedded UI is intentionally simple. It can later be replaced by a React / Vue frontend without changing the core service layer.
