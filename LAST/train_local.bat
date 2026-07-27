@echo off
setlocal

echo =================================================
echo SEM-Net Local Training Script
echo =================================================

cd /d "%~dp0"

:: Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (venv)...
    call "venv\Scripts\activate.bat"
) else (
    echo [WARNING] No virtual environment found. Running with global Python.
)

set CHECKPOINT_DIR=%~1
if "%CHECKPOINT_DIR%"=="" set CHECKPOINT_DIR=.\checkpoints

if not exist "%CHECKPOINT_DIR%" (
    echo [*] Checkpoint directory '%CHECKPOINT_DIR%' not found. Creating it...
    mkdir "%CHECKPOINT_DIR%"
)

if not exist "%CHECKPOINT_DIR%\config.yml" (
    echo [ERROR] 'config.yml' not found in '%CHECKPOINT_DIR%'.
    if exist "config.yml" (
        echo [*] Copying template config.yml to '%CHECKPOINT_DIR%'...
        copy "config.yml" "%CHECKPOINT_DIR%" >nul
    ) else (
        echo Please provide a valid config.yml
        exit /b 1
    )
)

echo [*] Starting local training using %CHECKPOINT_DIR%...

:: Launch training locally. Doesn't pipe to push_logs.py
python -u main.py --model 2 --path "%CHECKPOINT_DIR%"

if %ERRORLEVEL% equ 0 (
    echo [*] Training finished normally.
) else (
    echo [!] Training process exited with an error.
)
