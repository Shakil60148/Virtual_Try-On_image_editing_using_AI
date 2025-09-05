@echo off
echo Setting up Virtual Try-On application with Python 3.9...

REM Check if Python 3.9 is installed
py -3.9 --version 2>NUL
if %errorlevel% neq 0 (
    echo Python 3.9 is not found.
    echo Please install Python 3.9 from https://www.python.org/downloads/release/python-3913/
    echo After installation, you may need to run this script again.
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment with Python 3.9...
    py -3.9 -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment and install requirements
echo Activating virtual environment and installing requirements...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo.
echo To run the application:
echo 1. Ensure the virtual environment is activated (should show (venv) at the start of the command line)
echo 2. Run: streamlit run virtual_tryon_app.py
echo.
echo Press any key to launch the application...
pause >nul

REM Run the Streamlit app
streamlit run virtual_tryon_app.py

pause
