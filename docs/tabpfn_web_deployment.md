# TabPFN Web deployment

bochan Web treats TabPFN model weights as deployment assets. The public Web
runtime must not perform Prior Labs authentication or download checkpoints while
handling user requests.

The preload CLI intentionally lives at top-level `bochan.tabpfn_preload` rather
than inside `bochan.serving.webapp`. This keeps provisioning independent from
Web-app initialization and the wider BoTorch model stack.

## Local development quick start

The repository includes platform-specific helpers that preload the same TabPFN
v3 classifier/regressor checkpoints used by the Web runtime. The helpers do not
write the Prior Labs API key to disk.

For development with uv, synchronize all bochan optional dependencies into the
repository-local `.venv`:

```bash
uv sync --all-extras
```

The preload helpers automatically prefer the repository-local virtual
environment when it exists, so activation is not required just to run preload.
The Python selection order is:

1. explicit `BOCHAN_PYTHON`
2. repository `.venv` (`.venv\Scripts\python.exe` on Windows,
   `.venv/bin/python` on Linux)
3. `python` on `PATH`
4. Linux only: `python3` on `PATH`

If uv is not being used, install or refresh the current checkout's Web
requirements with the selected Python instead:

```bash
python -m pip install -e ".[web]"
```

The helper scripts verify that the selected Python can import the TabPFN APIs
needed by bochan before asking for an API key. If the dependency is missing or
incompatible, they print the selected Python executable plus both the uv and pip
recovery commands and exit without requesting a secret.

### Windows Command Prompt

From the bochan repository root, run:

```bat
uv sync --all-extras
scripts\preload_tabpfn.bat
```

If `.venv\Scripts\python.exe` exists, the helper uses it automatically even when
the virtual environment has not been activated in the current Command Prompt.
To verify the environment directly, run:

```bat
.venv\Scripts\python.exe -c "import tabpfn; print(tabpfn.__version__)"
```

If `TABPFN_TOKEN` is not already defined, the script prompts for the Prior Labs
API key with hidden input. The prompted token exists only inside the helper
process. The script also prepends the repository `src` directory to `PYTHONPATH`,
so the current checkout is used even when bochan is not installed editable.

By default, the helper uses TabPFN's upstream default cache directory. Because
the bochan Web runtime resolves the same default cache, no cache environment
variable is required for normal local development after preload succeeds.

To use an explicit local cache instead:

```bat
set "TABPFN_MODEL_CACHE_DIR=C:\path\to\tabpfn-cache"
scripts\preload_tabpfn.bat
```

Set the same `TABPFN_MODEL_CACHE_DIR` before starting bochan Web in later Command
Prompt sessions.

### Linux

From the bochan repository root, run:

```bash
uv sync --all-extras
bash scripts/preload_tabpfn.sh
```

If `.venv/bin/python` exists, the helper uses it automatically even when the
virtual environment has not been activated. To verify the environment directly,
run:

```bash
.venv/bin/python -c "import tabpfn; print(tabpfn.__version__)"
```

If `TABPFN_TOKEN` is not already defined, the script prompts for the Prior Labs
API key with hidden terminal input. The prompted token exists only inside the
helper process. The helper also prepends the repository `src` directory to
`PYTHONPATH`.

As on Windows, the upstream TabPFN cache is used by default. To use a specific
cache directory:

```bash
export TABPFN_MODEL_CACHE_DIR="/path/to/tabpfn-cache"
bash scripts/preload_tabpfn.sh
```

Export the same `TABPFN_MODEL_CACHE_DIR` before starting bochan Web in later
shell sessions.

Both helpers pass extra arguments through to the Python preload command. For
example, a developer who intentionally wants the upstream interactive browser
login can run `scripts\preload_tabpfn.bat --allow-browser-auth` on Windows or
`bash scripts/preload_tabpfn.sh --allow-browser-auth` on Linux.

## Production provisioning

1. Accept the required TabPFN license in the Prior Labs account used for deployment.
2. Store `TABPFN_TOKEN` in the deployment platform's secret manager.
3. Mount a persistent directory for TabPFN checkpoints and expose it as
   `TABPFN_MODEL_CACHE_DIR`.
4. Run the preload command in a deployment job or init container:

```bash
export TABPFN_TOKEN="..."
export TABPFN_MODEL_CACHE_DIR="/var/lib/bochan/tabpfn"
python -m bochan.tabpfn_preload
```

The preload command downloads the official default TabPFN v3 classifier and
regressor checkpoints only. It does not download all experimental variants.
Browser authentication is disabled by default, so an automated deployment fails
clearly when the token/license is unavailable.

5. Start the normal bochan Web container with the same checkpoint directory
   mounted, preferably read-only. Do not expose `TABPFN_TOKEN` to the runtime
   container.

```bash
export TABPFN_MODEL_CACHE_DIR="/var/lib/bochan/tabpfn"
# start bochan Web normally
```

For stronger isolation, deny runtime egress to Prior Labs/Hugging Face at the
infrastructure or network-policy layer as well.

## Local interactive provisioning

For a developer workstation only, the Python preload command can explicitly
allow the TabPFN library's browser login flow:

```bash
python -m bochan.tabpfn_preload --allow-browser-auth
```

The platform helper scripts expose the same option as described above.

## Runtime contract

When a real bochan Web request selects TabPFN, bochan checks that both required
default checkpoints are already present and non-empty before constructing the
estimator. If either checkpoint is missing, the request fails with a provisioning
error that points to the preload command. The Web process forces
`TABPFN_NO_BROWSER=1`, so browser authentication is never used as a runtime
fallback.

Explicitly injected, already-constructed estimators used by tests or advanced
programmatic integrations are not subject to the deployment asset check.
