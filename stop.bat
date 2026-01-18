@echo off
setlocal enabledelayedexpansion

REM Trouver le PID sur le port 8012
set "pid="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8012" ^| findstr "LISTENING"') do set "pid=%%P"

REM Si aucun processus trouve, quitter
if "!pid!"=="" (
    echo Aucun processus G12 trouve.
    timeout /t 2 /nobreak >nul
    exit
)

REM Tuer le processus
taskkill /F /PID !pid! >nul 2>&1

if !errorlevel! equ 0 (
    echo G12 arrete.
) else (
    echo Erreur: Executez en tant qu'Administrateur.
)

timeout /t 2 /nobreak >nul
endlocal
exit
