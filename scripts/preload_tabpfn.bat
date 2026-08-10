@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Preload the TabPFN weights required by the bochan Web runtime.
rem The API key is never written to disk by this script.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

pushd "%REPO_ROOT%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to enter the bochan repository root: "%REPO_ROOT%"
    exit /b 1
)

set "SHOW_HELP=0"
set "TOKEN_OPTIONAL=0"
for %%A in (%*) do (
    if /I "%%~A"=="--allow-browser-auth" set "TOKEN_OPTIONAL=1"
    if /I "%%~A"=="--help" set "SHOW_HELP=1"
    if /I "%%~A"=="-h" set "SHOW_HELP=1"
)

if "%SHOW_HELP%"=="1" (
    echo Usage: scripts\preload_tabpfn.bat [options]
    echo.
    echo Preload the TabPFN v3 classifier and regressor checkpoints used by bochan Web.
    echo If TABPFN_TOKEN is not set, the Prior Labs API key is requested with hidden input.
    echo.
    echo Python selection order:
    echo   1. BOCHAN_PYTHON
    echo   2. repo-local .venv\Scripts\python.exe
    echo   3. python on PATH
    echo.
    echo Common options passed to the Python preload command:
    echo   --cache-dir PATH          Use an explicit TabPFN checkpoint directory.
    echo   --allow-browser-auth      Allow Prior Labs browser authentication for local setup.
    echo   --help, -h                Show this help.
    echo.
    echo Environment variables:
    echo   TABPFN_TOKEN              Prior Labs API key. Optional when prompted interactively.
    echo   TABPFN_MODEL_CACHE_DIR    Persistent checkpoint directory. Upstream default if unset.
    echo   BOCHAN_PYTHON             Explicit Python executable path.
    popd
    exit /b 0
)

if defined PYTHONPATH (
    set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%REPO_ROOT%\src"
)

if defined BOCHAN_PYTHON (
    set "PYTHON_CMD=%BOCHAN_PYTHON%"
) else if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

rem Check the exact Python environment before asking for a secret.
"%PYTHON_CMD%" -c "import tabpfn; from tabpfn.constants import ModelVersion; from tabpfn.model_loading import download_model, get_cache_dir, resolve_model_path" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] A compatible TabPFN package is not installed in the selected Python environment.
    echo Python executable:
    "%PYTHON_CMD%" -c "import sys; print(sys.executable)"
    echo.
    echo Preferred development setup with uv:
    echo   uv sync --all-extras
    echo.
    echo Or install the bochan Web dependencies with pip:
    echo   "%PYTHON_CMD%" -m pip install -e ".[web]"
    echo.
    echo Then rerun:
    echo   scripts\preload_tabpfn.bat
    popd
    exit /b 2
)

if not defined TABPFN_TOKEN if "%TOKEN_OPTIONAL%"=="0" (
    echo Prior Labs API Key is required only for this preload step.
    for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "$s = Read-Host 'Prior Labs API Key' -AsSecureString; $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b) } finally { if ($b -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) } }"`) do set "TABPFN_TOKEN=%%T"
)

if not defined TABPFN_TOKEN if "%TOKEN_OPTIONAL%"=="0" (
    echo [ERROR] TABPFN_TOKEN is empty. Preload was not started.
    popd
    exit /b 1
)

if defined TABPFN_MODEL_CACHE_DIR (
    echo TabPFN cache: "%TABPFN_MODEL_CACHE_DIR%"
) else (
    echo TabPFN cache: upstream default cache directory
)

"%PYTHON_CMD%" -m bochan.tabpfn_preload %*
set "EXIT_CODE=%ERRORLEVEL%"

rem Remove a token entered by this helper from its local environment.
set "TABPFN_TOKEN="
popd

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] TabPFN preload failed with exit code %EXIT_CODE%.
    exit /b %EXIT_CODE%
)

if "%TOKEN_OPTIONAL%"=="0" echo TabPFN preload completed successfully.
exit /b 0
