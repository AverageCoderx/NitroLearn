@echo off
rem Double-click: Replay a game played by a run's newest bot, in your browser.
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" watch
echo.
pause
