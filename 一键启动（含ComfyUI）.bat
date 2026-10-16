@echo off
chcp 65001 > nul
title ID Photo Tool - Full Start
cd /d "%~dp0"

set "VENV_PY=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\pythonw.exe"
if not exist "%VENV_PY%" set "VENV_PY=pythonw"
set "COMFY_BAT=F:\bfcomfyui\BFCOMFYUI.bat"

echo ============================================
echo   ID Photo Tool  -  full start
echo   (ComfyUI + App)
echo ============================================
echo.

rem ---------- is ComfyUI already running? ----------
set "COMFY_OK="
curl -s -m 3 --noproxy "127.0.0.1" -o nul "http://127.0.0.1:8166/system_stats"
if not errorlevel 1 set "COMFY_OK=1"
if defined COMFY_OK goto comfy_ready

rem ---------- start it ----------
if not exist "%COMFY_BAT%" (
  echo [WARN] ComfyUI launcher not found: %COMFY_BAT%
  echo        Skipping - AI generative editing will be disabled.
  echo.
  goto launch_app
)

echo [1/2] Starting ComfyUI ... a separate window will open.
start "ComfyUI" cmd /c ""%COMFY_BAT%""
echo [2/2] Waiting for ComfyUI on port 8166 ...

set /a WAIT=0
:wait_loop
timeout /t 3 /nobreak > nul
curl -s -m 3 --noproxy "127.0.0.1" -o nul "http://127.0.0.1:8166/system_stats"
if not errorlevel 1 goto comfy_ready
set /a WAIT+=3
if %WAIT% GEQ 120 (
  echo [WARN] ComfyUI did not respond within 120s. Continuing anyway.
  echo.
  goto launch_app
)
echo        ... still waiting ^(%WAIT%s^)
goto wait_loop

:comfy_ready
echo [OK] ComfyUI is up - AI generative editing is available.
echo.

:launch_app
echo Launching ID Photo Tool ...
start "" "%VENV_PY%" "%~dp0main.py"
exit /b 0
