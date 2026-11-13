@echo off
rem Double-click to uninstall NitroLearn. Everything NitroLearn installed is inside this folder, except the
rem Start menu and desktop shortcuts it made: this removes exactly those (listed in .tools\shortcuts.json),
rem then you delete this folder.
cd /d "%~dp0"
rem Installed with NitroLearn-Setup: its own uninstaller (also in Windows' Settings > Apps) does it all.
if exist "%~dp0unins000.exe" (
    start "" "%~dp0unins000.exe"
    exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\app_shortcuts.ps1" -Remove
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (
    echo NitroLearn's shortcuts are removed.
) else (
    echo Some shortcuts could not be removed: read the messages above and delete them by hand.
)
echo To finish, close the dashboard and any training, then delete this folder:
echo   %~dp0
echo Your bots (runs) and settings files (my-settings) are in it: copy them elsewhere first to keep them.
echo RLBot v5, if you installed it, is its own program and stays: remove it in Windows' Apps list.
pause
exit /b %CODE%
