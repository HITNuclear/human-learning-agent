@echo off
setlocal
echo Starting backend server...
cd /d %~dp0

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv venv 2>nul || python -m venv venv
)

findstr /C:"include-system-site-packages = true" "venv\pyvenv.cfg" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo.
    echo Existing backend\venv was created with system-site-packages enabled.
    echo For a clean open-source setup, delete backend\venv and run this script again:
    echo   rmdir /s /q venv
    echo.
)

echo Installing backend dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if "%HLA_ALLOW_MODEL_DOWNLOAD%"=="1" (
    echo Hugging Face model downloads are enabled for this run.
) else (
    echo Using cached Hugging Face models. Set HLA_ALLOW_MODEL_DOWNLOAD=1 to allow first-run downloads.
    set TRANSFORMERS_OFFLINE=1
    set HF_HUB_OFFLINE=1
)

"venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
