@echo off
setlocal

REM One-command Phase 4 key checks/tests
set PY_BIN=Dev\Scripts\python.exe
if not exist "%PY_BIN%" (
  set PY_BIN=python
)

echo ================================================
echo      Running RyuuAI Key Checks and Tests
echo ================================================
echo Python: %PY_BIN%
echo.

%PY_BIN% testing\run_key_checks.py --python %PY_BIN%
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% neq 0 (
  echo One or more checks/tests failed.
) else (
  echo All key checks/tests passed.
)

exit /b %EXIT_CODE%
