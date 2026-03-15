@echo off
setlocal

REM Run full Ryuu-Dev data pipeline with project venv python.
set PY_EXE=Dev\Scripts\python.exe
if not exist "%PY_EXE%" (
  echo [ERR] Missing %PY_EXE%
  exit /b 1
)

"%PY_EXE%" data_pipeline\run_full_pipeline.py --python "%PY_EXE%"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo [ERR] Data pipeline failed with exit code %RC%
  exit /b %RC%
)

echo [OK] Data pipeline completed successfully.
exit /b 0
