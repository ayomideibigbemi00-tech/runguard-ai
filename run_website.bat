@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  py -m venv .venv
  if errorlevel 1 exit /b 1
  echo Installing dependencies...
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
echo Starting Runguard AI on http://127.0.0.1:8765
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
