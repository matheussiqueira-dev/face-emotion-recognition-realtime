@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==========================================
echo       EmotionAI - Senior Experience
echo ==========================================

if not exist .venv (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    echo [2/3] Installing dependencies...
    call .\.venv\Scripts\activate.bat
    python -m pip install -r requirements.txt
) else (
    call .\.venv\Scripts\activate.bat
)

echo [3/3] Starting EmotionAI Dashboard...
echo Dashboard will be available at http://127.0.0.1:8000
python run_api.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
