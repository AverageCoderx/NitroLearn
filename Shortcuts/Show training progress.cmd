@echo off
rem Double-click: Open a run's progress page in your browser.
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" progress
echo.
pause
