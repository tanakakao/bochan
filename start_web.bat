@echo off
setlocal

rem Fixed ports for the bochan development web application.
set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
set "FRONTEND_HOST=127.0.0.1"
set "FRONTEND_PORT=5173"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Activate the bochan virtual environment before running this file.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found on PATH.
    echo Install Node.js and make sure npm is available.
    pause
    exit /b 1
)

echo Starting bochan backend at http://%BACKEND_HOST%:%BACKEND_PORT% ...
start "bochan backend" /D "%~dp0" cmd /k python -m uvicorn bochan.serving.webapp.app:app --reload --host %BACKEND_HOST% --port %BACKEND_PORT%

echo Starting bochan frontend at http://%FRONTEND_HOST%:%FRONTEND_PORT% ...
start "bochan frontend" /D "%~dp0web" cmd /k npm run dev -- --host %FRONTEND_HOST% --port %FRONTEND_PORT% --strictPort

echo.
echo bochan startup commands were launched.
echo Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%
echo Backend : http://%BACKEND_HOST%:%BACKEND_PORT%

endlocal
