#!/usr/bin/env bash
set -euo pipefail

# Preload the TabPFN weights required by the bochan Web runtime.
# The API key is kept only in this process environment and is never written to disk.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${BOCHAN_PYTHON:-}" ]]; then
    PYTHON_CMD="${BOCHAN_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

TOKEN_OPTIONAL=0
for arg in "$@"; do
    case "${arg}" in
        --allow-browser-auth|--help|-h)
            TOKEN_OPTIONAL=1
            ;;
    esac
done

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

"${PYTHON_CMD}" -m bochan.serving.webapp.tabpfn_preload "$@"
if [[ "${TOKEN_OPTIONAL}" == "0" ]]; then
    echo "TabPFN preload completed successfully."
fi
