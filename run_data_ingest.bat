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
echo Starting controlled CoinGecko ingestion...
.venv\Scripts\python.exe scripts\ingest_data.py %*
if errorlevel 2 (
  echo.
  echo INGESTION INCOMPLETE. Failed jobs remain in data\cache\retry_queue.json.
  exit /b 2
)
echo.
echo DATA INGESTION COMPLETE.
endlocal
