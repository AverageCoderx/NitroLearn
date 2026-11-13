@echo off
rem Double-click: Continue a stopped run from its last save.
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" resume
echo.
pause
