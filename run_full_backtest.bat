@echo off
setlocal
cd /d "%~dp0"
python scripts\run_full_backtest.py %*
if errorlevel 2 (
  echo.
  echo BACKTEST INCOMPLETE. Check the errors above.
  exit /b 2
)
echo.
echo BACKTEST COMPLETE.
endlocal
