@echo off
cd /d "%~dp0"
echo Starting Vendor Demand Studio...
where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or later from python.org.
    pause
    exit /b 1
)
py -m pip install -r requirements-web.txt
if errorlevel 1 (
    echo Could not install dependencies. Check your internet connection.
    pause
    exit /b 1
)
py server.py
pause
