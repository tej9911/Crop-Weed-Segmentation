@echo off
cd /d "%~dp0.."
start "" http://localhost:5000
python -m backend.app
