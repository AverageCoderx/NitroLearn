@echo off
rem Double-click: Turn a run's newest bot into an RLBot v5 bot folder (bots\<run>) and open it.
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" export
echo.
pause
