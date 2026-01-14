@echo off
title G12 - Arret
color 0C

echo ============================================================
echo                G12 - Arret en cours
echo ============================================================
echo.

echo Arret des processus Python G12...
taskkill /F /FI "WINDOWTITLE eq G12*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq G12*" 2>nul

echo.
echo G12 arrete.
echo.

timeout /t 2
