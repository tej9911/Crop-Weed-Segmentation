@echo off
rem Start the development server and open the app in the default browser.
cd /d "%~dp0.."
if "%PORT%"=="" set PORT=5000

rem Open the browser a few seconds after the server starts.
start "" /b cmd /c "ping -n 5 127.0.0.1 >nul & start "" http://127.0.0.1:%PORT%"
python -m backend.app
