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

if defined PYTHONPATH (
    set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%REPO_ROOT%\src"
)

if defined BOCHAN_PYTHON (
    set "PYTHON_CMD=%BOCHAN_PYTHON%"
) else (
    set "PYTHON_CMD=python"
)

set "TOKEN_OPTIONAL=0"
for %%A in (%*) do (
    if /I "%%~A"=="--allow-browser-auth" set "TOKEN_OPTIONAL=1"
    if /I "%%~A"=="--help" set "TOKEN_OPTIONAL=1"
    if /I "%%~A"=="-h" set "TOKEN_OPTIONAL=1"
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

"%PYTHON_CMD%" -m bochan.serving.webapp.tabpfn_preload %*
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
