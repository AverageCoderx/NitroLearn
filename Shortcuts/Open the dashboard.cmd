@echo off
rem Double-click: open the dashboard, a window to make, train, change, watch and export bots.
rem What it runs is in shortcuts.ps1, next to this file. Closing the dashboard
rem leaves any training running; open it again to see it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0shortcuts.ps1" dashboard
if errorlevel 1 pause
