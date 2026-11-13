@echo off
rem Double-click: Check that this PC is ready to train (nitro doctor).
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" doctor
echo.
pause
