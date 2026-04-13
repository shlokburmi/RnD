@echo off
:: ─────────────────────────────────────────────────────────────────────
::  run_on_pi.bat  —  Run CNN-SPECK batch processor on Raspberry Pi 4
::  Double-click this file from Windows. No VS Code needed.
:: ─────────────────────────────────────────────────────────────────────

:: ── CONFIGURE THESE TWO LINES ─────────────────────────────────────────
set PI_USER=pi
set PI_IP=192.168.x.x
set PI_DIR=/home/pi/speck_cnn
:: ─────────────────────────────────────────────────────────────────────

echo.
echo ============================================================
echo   CNN-SPECK Raspberry Pi 4 Runner
echo   Connecting to %PI_USER%@%PI_IP% ...
echo ============================================================
echo.

ssh %PI_USER%@%PI_IP% "cd %PI_DIR% && python3 batch_process.py"

echo.
echo ============================================================
echo   Done! Fetching results file ...
echo ============================================================

:: Copy cnnresults.txt back to this Windows folder automatically
scp %PI_USER%@%PI_IP%:%PI_DIR%/cnnresults.txt "%~dp0cnnresults.txt"

echo.
echo   Results saved to: %~dp0cnnresults.txt
echo ============================================================
pause
