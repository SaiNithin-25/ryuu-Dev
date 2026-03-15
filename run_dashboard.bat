@echo off
setlocal

cd /d "%~dp0"

set "STREAMLIT=%~dp0Dev\Scripts\streamlit.exe"
set "PYTHON=%~dp0Dev\Scripts\python.exe"

if exist "%STREAMLIT%" (
    "%STREAMLIT%" run training_dashboard.py %*
    exit /b %errorlevel%
)

if exist "%PYTHON%" (
    "%PYTHON%" -m streamlit run training_dashboard.py %*
    exit /b %errorlevel%
)

python -m streamlit run training_dashboard.py %*
