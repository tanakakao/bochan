# Python environment

bochan uses uv for reproducible Python environments.

## Files under version control

- `pyproject.toml`: direct dependency requirements and optional dependency groups.
- `uv.lock`: exact resolved dependency graph. Commit this file together with dependency changes.
- `.python-version`: default Python version for the project (`3.12`).

The generated `.venv/` directory is local to each checkout and is not committed.

## Web environment

Install uv, then from the repository root run:

```bash
uv sync --locked --extra web
```

This creates or updates `.venv` from the committed lockfile. `--locked` fails instead of silently changing `uv.lock` when `pyproject.toml` and the lockfile do not agree.

Run commands in the project environment with, for example:

```bash
uv run --locked --extra web python -m uvicorn bochan.serving.webapp.app:app
```

## Other common environments

Tests:

```bash
uv sync --locked --extra test
```

Development tools:

```bash
uv sync --locked --extra dev
```

## Updating dependencies

When dependencies intentionally change:

1. Edit `pyproject.toml`.
2. Run `uv lock`.
3. Review the `uv.lock` diff.
4. Run the relevant `uv sync --locked ...` and tests.
5. Commit `pyproject.toml` and `uv.lock` together.

Do not hand-edit `uv.lock`.

The React frontend is managed separately with pnpm and `web/pnpm-lock.yaml`.
