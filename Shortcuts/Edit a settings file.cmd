@echo off
rem Double-click: Open one of your settings files in Notepad; it is checked for mistakes when you close Notepad.
rem What it runs is in shortcuts.ps1, next to this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" edit-settings
echo.
pause
