@echo off
rem Double-click to install NitroLearn, or to update it after downloading a new version into this folder.
rem Everything it downloads stays inside this folder; to remove NitroLearn, delete the folder.
rem Options are passed on to install.ps1, for example:  "Install NitroLearn.cmd" -Rust
cd /d "%~dp0"
if not exist "%~dp0install.ps1" (
    echo This file was started from inside the zip, without the rest of NitroLearn.
    echo.
    echo Extract the whole zip first: right-click the zip file, choose Extract All, then open the
    echo extracted folder and double-click "Install NitroLearn.cmd" there.
    echo.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (
    echo Done. You can close this window.
) else if "%CODE%"=="2" (
    echo NitroLearn is installed, but its final check found something to fix: read the "fix:" lines above.
) else (
    echo Setup did not finish: read the messages above.
)
pause
exit /b %CODE%
