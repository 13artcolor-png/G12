@echo off
title G12 - BTCUSD Trading Bot
color 0A

echo ============================================================
echo                G12 - BTCUSD Trading Bot
echo ============================================================
echo.

cd /d "%~dp0backend"

echo [0/3] Nettoyage du port 8012...
setlocal
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8012 ^| findstr LISTENING') do (
    echo Fermeture du processus %%a utilisant le port 8012...
    taskkill /F /PID %%a /T 2>nul
)
endlocal
timeout /t 1 /nobreak >nul

echo [1/3] Verification de Python...
python --version
if errorlevel 1 (
    echo ERREUR: Python non trouve!
    pause
    exit /b 1
)

echo.
echo [2/3] Installation des dependances...
pip install -r ../requirements.txt -q

echo.
echo [3/3] Demarrage de G12...
echo.
echo ============================================================
echo   API:       http://localhost:8012
echo   Dashboard: http://localhost:8012/
echo ============================================================
echo.
echo Appuyez sur Ctrl+C pour arreter.
echo.

:: Ouvrir le dashboard dans le navigateur (avec delai)
start "" "http://localhost:8012/"

python main.py

pause
