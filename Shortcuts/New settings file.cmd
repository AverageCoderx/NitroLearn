@echo off
rem Double-click: Make your own settings file from a preset (1v1, 2v2 or 3v3) and open it in Notepad.
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" new-settings
echo.
pause
