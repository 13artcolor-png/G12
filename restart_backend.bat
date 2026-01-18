@echo off
echo Redemarrage du backend G12...

REM Tuer le processus Python sur le port 8012
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8012 ^| findstr LISTENING') do (
    echo Arret du processus %%a...
    taskkill /F /PID %%a >nul 2>&1
)

REM Attendre 3 secondes
timeout /t 3 /nobreak >nul

REM Relancer le backend
cd /d "%~dp0"
start "G12_Backend" cmd /k "python backend/main.py"

echo Backend redemarré!
exit
