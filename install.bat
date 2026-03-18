@echo off
echo ============================================
echo OR RAG Bot - Installation Script (Windows)
echo ============================================
echo.

REM Check Python version
echo [1/7] Checking Python version...
python --version
echo.

REM Create virtual environment
echo [2/7] Creating virtual environment...
python -m venv venv
echo Virtual environment created
echo.

REM Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip
echo Pip upgraded
echo.

REM Install dependencies
echo [5/7] Installing dependencies...
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Create directories
echo [6/7] Creating directories...
mkdir data_raw 2>nul
mkdir data_processed 2>nul
mkdir vector_db 2>nul
mkdir frontend\static 2>nul
echo Directories created
echo.

REM Create .env file
echo [7/7] Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env file - please add your API keys
) else (
    echo .env file already exists
)
echo.

echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Edit .env and add your API key
echo 2. Run: python src\main.py
echo 3. Run: python src\app.py
echo 4. Open: http://localhost:5000
echo.
echo For detailed instructions, see QUICKSTART.md
echo ============================================
pause