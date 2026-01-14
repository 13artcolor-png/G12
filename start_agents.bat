@echo off
echo ============================================================
echo           G12 - Lancement des 3 Agents
echo ============================================================
echo.

cd /d "%~dp0backend"

echo Lancement de MOMENTUM (247600)...
start "G12 - MOMENTUM" cmd /k "title G12 - MOMENTUM && color 09 && python agent_runner.py momentum"

timeout /t 2 /nobreak >nul

echo Lancement de REVERSAL (247601)...
start "G12 - REVERSAL" cmd /k "title G12 - REVERSAL && color 0A && python agent_runner.py reversal"

timeout /t 2 /nobreak >nul

echo Lancement de LIQUIDATION (247602)...
start "G12 - LIQUIDATION" cmd /k "title G12 - LIQUIDATION && color 0E && python agent_runner.py liquidation"

echo.
echo ============================================================
echo   3 fenetres CMD ouvertes pour les agents
echo   - MOMENTUM (bleu)
echo   - REVERSAL (vert)
echo   - LIQUIDATION (jaune)
echo ============================================================
echo.
echo Cette fenetre peut etre fermee.
pause
