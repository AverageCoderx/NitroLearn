@echo off
rem Double-click: Start a new training run: pick the settings, name the run, and it starts (its progress page opens in your browser).
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" train
echo.
pause
