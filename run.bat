@echo off
rem One-step launcher for Windows. Double-click it, or run from a terminal:
rem   run.bat              dashboard + demo in simulation mode, opens the browser
rem   run.bat test         run the tests
rem   run.bat verify       check every number in the paper/report/slides
rem   run.bat reproduce    rerun every experiment (about 25 min)
rem   run.bat help         all actions
rem The first run creates .venv-windows and installs requirements.txt (needs Python 3.10+).
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py %*
) else (
    python run.py %*
)
set RC=%errorlevel%
rem Keep the window open when started by double-click
if "%~1"=="" pause
exit /b %RC%
