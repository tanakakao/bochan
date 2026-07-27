@echo off
setlocal EnableExtensions

rem Fixed ports for the bochan development web application.
set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
set "FRONTEND_HOST=127.0.0.1"
set "FRONTEND_PORT=5173"
set "HEALTH_URL=http://%BACKEND_HOST%:%BACKEND_PORT%/api/v1/health"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if /i "%~1"=="backend" goto backend
if /i "%~1"=="frontend" goto frontend

echo ========================================
echo bochan Web launcher
echo ========================================
echo.

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found on PATH.
    echo Install Node.js and make sure npm is available.
    echo.
    pause
    exit /b 1
)

if exist "%VENV_PYTHON%" (
    echo Python: %VENV_PYTHON%
) else (
    where uv >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Neither .venv\Scripts\python.exe nor uv was found.
        echo Create the uv environment in this repository or install uv.
        echo.
        pause
        exit /b 1
    )
    echo Python: uv run --extra web python
)

echo Starting bochan backend at http://%BACKEND_HOST%:%BACKEND_PORT% ...
start "bochan backend" /D "%~dp0" cmd.exe /k ""%~f0" backend"

echo Waiting for FastAPI to become ready...
call :wait_for_backend
if errorlevel 1 (
    echo.
    echo [ERROR] bochan FastAPI did not become ready within 60 seconds.
    echo Check the bochan backend window for the traceback or port error.
    echo The React frontend was not started.
    echo.
    pause
    exit /b 1
)

echo FastAPI is ready.
echo Starting bochan frontend at http://%FRONTEND_HOST%:%FRONTEND_PORT% ...
start "bochan frontend" /D "%~dp0web" cmd.exe /k ""%~f0" frontend"

echo.
echo Startup windows were opened.
echo Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%
echo Backend : http://%BACKEND_HOST%:%BACKEND_PORT%
echo Health  : %HEALTH_URL%
echo.
echo Press any key to close only this launcher window.
pause >nul
exit /b 0

:wait_for_backend
for /L %%I in (1,1,60) do (
    powershell.exe -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 exit /b 0
    timeout /t 1 /nobreak >nul
)
exit /b 1

:backend
cd /d "%~dp0"
echo ========================================
echo bochan FastAPI backend
echo ========================================
echo.

if exist "%VENV_PYTHON%" (
    echo Using uv virtual environment:
    echo %VENV_PYTHON%
    echo.
    "%VENV_PYTHON%" -m uvicorn bochan.serving.webapp.app:app --reload --host %BACKEND_HOST% --port %BACKEND_PORT%
) else (
    echo .venv was not found. Starting through uv run.
    echo.
    uv run --extra web python -m uvicorn bochan.serving.webapp.app:app --reload --host %BACKEND_HOST% --port %BACKEND_PORT%
)

set "SERVER_EXIT=%ERRORLEVEL%"
echo.
echo [ERROR] bochan backend stopped. Exit code: %SERVER_EXIT%
echo Check the error message above. This window will remain open.
pause
exit /b %SERVER_EXIT%

:frontend
cd /d "%~dp0web"
echo ========================================
echo bochan React frontend
echo ========================================
echo.

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found on PATH.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo node_modules was not found. Running npm install...
    call npm install
    if errorlevel 1 (
        echo.
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

call npm run dev -- --host %FRONTEND_HOST% --port %FRONTEND_PORT% --strictPort
set "FRONTEND_EXIT=%ERRORLEVEL%"
echo.
echo [ERROR] bochan frontend stopped. Exit code: %FRONTEND_EXIT%
echo Check the error message above. This window will remain open.
pause
exit /b %FRONTEND_EXIT%
