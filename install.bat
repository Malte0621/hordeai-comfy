@echo off
echo Installing AI Horde dependencies...
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not available. Please ensure Python and pip are installed.
    pause
    exit /b 1
)

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Installation complete!
echo You can now restart ComfyUI to use the AI Horde nodes.
echo.
pause
