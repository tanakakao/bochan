#!/usr/bin/env bash
set -euo pipefail

# Preload the TabPFN weights required by the bochan Web runtime.
# The API key is kept only in this process environment and is never written to disk.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SHOW_HELP=0
TOKEN_OPTIONAL=0
for arg in "$@"; do
    case "${arg}" in
        --allow-browser-auth)
            TOKEN_OPTIONAL=1
            ;;
        --help|-h)
            SHOW_HELP=1
            ;;
    esac
done

if [[ "${SHOW_HELP}" == "1" ]]; then
    cat <<'EOF'
Usage: bash scripts/preload_tabpfn.sh [options]

Preload the TabPFN v3 classifier and regressor checkpoints used by bochan Web.
If TABPFN_TOKEN is not set, the Prior Labs API key is requested with hidden input.

Common options passed to the Python preload command:
  --cache-dir PATH          Use an explicit TabPFN checkpoint directory.
  --allow-browser-auth      Allow Prior Labs browser authentication for local setup.
  --help, -h                Show this help.

Environment variables:
  TABPFN_TOKEN              Prior Labs API key. Optional when prompted interactively.
  TABPFN_MODEL_CACHE_DIR    Persistent checkpoint directory. Upstream default if unset.
  BOCHAN_PYTHON             Python executable path. Defaults to python, then python3.
EOF
    exit 0
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${BOCHAN_PYTHON:-}" ]]; then
    PYTHON_CMD="${BOCHAN_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check the exact Python environment before asking for a secret.
if ! "${PYTHON_CMD}" -c 'import tabpfn; from tabpfn.constants import ModelVersion; from tabpfn.model_loading import download_model, get_cache_dir, resolve_model_path' >/dev/null 2>&1; then
    echo "[ERROR] A compatible TabPFN package is not installed in the selected Python environment." >&2
    echo "Python executable:" >&2
    "${PYTHON_CMD}" -c 'import sys; print(sys.executable)' >&2 || true
    echo >&2
    echo "Install or refresh the bochan Web dependencies from this repository:" >&2
    printf '  %q -m pip install -e ".[web]"\n' "${PYTHON_CMD}" >&2
    echo >&2
    echo "Then rerun:" >&2
    echo "  bash scripts/preload_tabpfn.sh" >&2
    exit 2
fi

TOKEN_WAS_PROMPTED=0
if [[ -z "${TABPFN_TOKEN:-}" && "${TOKEN_OPTIONAL}" == "0" ]]; then
    if [[ ! -t 0 ]]; then
        echo "[ERROR] TABPFN_TOKEN is not set and no interactive terminal is available." >&2
        exit 1
    fi
    printf '%s' "Prior Labs API Key: "
    IFS= read -r -s TABPFN_TOKEN
    printf '\n'
    export TABPFN_TOKEN
    TOKEN_WAS_PROMPTED=1
fi

if [[ -z "${TABPFN_TOKEN:-}" && "${TOKEN_OPTIONAL}" == "0" ]]; then
    echo "[ERROR] TABPFN_TOKEN is empty. Preload was not started." >&2
    exit 1
fi

cleanup() {
    if [[ "${TOKEN_WAS_PROMPTED}" == "1" ]]; then
        unset TABPFN_TOKEN
    fi
}
trap cleanup EXIT

if [[ -n "${TABPFN_MODEL_CACHE_DIR:-}" ]]; then
    printf 'TabPFN cache: %s\n' "${TABPFN_MODEL_CACHE_DIR}"
else
    echo "TabPFN cache: upstream default cache directory"
fi

"${PYTHON_CMD}" -m bochan.tabpfn_preload "$@"
if [[ "${TOKEN_OPTIONAL}" == "0" ]]; then
    echo "TabPFN preload completed successfully."
fi
