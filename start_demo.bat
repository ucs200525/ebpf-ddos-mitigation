@echo off
rem One-click demo for the review (Windows, simulation mode).
rem Starts the system with the dashboard and opens it in the browser.
rem In the dashboard, press "Guided tour". Close this window to stop.
cd /d "%~dp0"
echo Starting DDoS Shield (simulation mode)...
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://127.0.0.1:5000"
python main.py --interface Ethernet --dashboard
pause
