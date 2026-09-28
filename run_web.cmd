@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" web_app.py --open
) else (
    python web_app.py --open
)
pause
