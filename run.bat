@echo off
cd /d "%~dp0"
echo Starting Face Emotion Recognition...

if not exist .venv (
    echo Virtual environment not found. Creating...
    python -m venv .venv
    echo Installing dependencies...
    call .\.venv\Scripts\activate.bat
    python -m pip install -r requirements.txt
) else (
    call .\.venv\Scripts\activate.bat
)

if %errorlevel% neq 0 (
    echo Error activating virtual environment.
    pause
    exit /b %errorlevel%
)

python main.py %*
if %errorlevel% neq 0 (
    echo Application exited with error.
    pause
) else (
    echo Application finished.
)
