@echo off
chcp 65001 > nul
title ID Photo Tool
cd /d "%~dp0"

set "VENV_PY=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\pythonw.exe"
if not exist "%VENV_PY%" set "VENV_PY=pythonw"

echo ============================================
echo   ID Photo Tool  -  starting...
echo ============================================
echo.

set "COMFY_OK="
curl -s -m 3 --noproxy "127.0.0.1" -o nul "http://127.0.0.1:8166/system_stats"
if not errorlevel 1 set "COMFY_OK=1"
if not defined COMFY_OK (
  curl -s -m 3 --noproxy "127.0.0.1" -o nul "http://127.0.0.1:8188/system_stats"
  if not errorlevel 1 set "COMFY_OK=1"
)

if not defined COMFY_OK (
  echo [WARN] ComfyUI not detected on port 8166 / 8188.
  echo        The app still works; only "AI generative editing" is disabled.
  echo        To enable it, run:  F:\bfcomfyui\BFCOMFYUI.bat
  echo.
) else (
  echo [OK] ComfyUI detected - AI generative editing is available.
  echo.
)

start "" "%VENV_PY%" "%~dp0main.py"
exit /b 0
