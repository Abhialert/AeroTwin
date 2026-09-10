@echo off
echo ========================================================
echo        AeroTwin SIH 2026 - Digital Twin Dashboard
echo ========================================================
echo.
echo Starting Flask backend on http://localhost:5000
echo.
echo  - Upload a video to start processing
echo  - Results will appear at http://localhost:5000/results.html
echo  - Keep this window open while using the dashboard
echo.
cd /d "%~dp0"
start "" http://localhost:5000
.venv\Scripts\python.exe server.py
