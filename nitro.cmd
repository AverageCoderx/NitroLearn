@echo off
rem Runs the "nitro" command from this folder's private Python environment (made by the installer).
rem Tools, caches and temporary files are kept inside this folder (same settings as tools\env.ps1).
setlocal
set "NITRO_ROOT=%~dp0"
if not exist "%NITRO_ROOT%.venv\Scripts\nitro.exe" (
    echo NitroLearn is not installed yet: double-click "Install NitroLearn.cmd" in this folder.
    exit /b 1
)
set "UV_PYTHON_INSTALL_DIR=%NITRO_ROOT%.tools\python"
set "UV_PYTHON_BIN_DIR=%NITRO_ROOT%.tools\python\bin"
set "UV_CACHE_DIR=%NITRO_ROOT%.cache\uv"
set "UV_TOOL_DIR=%NITRO_ROOT%.tools\uv-tools"
set "UV_TOOL_BIN_DIR=%NITRO_ROOT%.tools\uv-tools\bin"
set "RUSTUP_HOME=%NITRO_ROOT%.tools\rustup"
set "CARGO_HOME=%NITRO_ROOT%.tools\cargo"
set "PIP_CACHE_DIR=%NITRO_ROOT%.cache\pip"
rem NitroLearn's Python makes CUDA_CACHE_PATH readable for NVIDIA's driver, or drops it (folder_env.py).
set "CUDA_CACHE_PATH=%NITRO_ROOT%.cache\cuda"
set "TORCH_HOME=%NITRO_ROOT%.cache\torch"
set "TORCHINDUCTOR_CACHE_DIR=%NITRO_ROOT%.cache\torchinductor"
set "TRITON_CACHE_DIR=%NITRO_ROOT%.cache\triton"
set "WANDB_CONFIG_DIR=%NITRO_ROOT%.cache\wandb\config"
set "WANDB_CACHE_DIR=%NITRO_ROOT%.cache\wandb\cache"
set "WANDB_DATA_DIR=%NITRO_ROOT%.cache\wandb\data"
set "WANDB_ARTIFACT_DIR=%NITRO_ROOT%.cache\wandb\artifacts"
set "TEMP=%NITRO_ROOT%.tmp"
set "TMP=%NITRO_ROOT%.tmp"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_MANAGED_PYTHON=1"
set "UV_NO_CONFIG=1"
set "UV_SYSTEM_CERTS=1"
rem A cargo download that stalls fails (and is tried again) instead of hanging.
set "CARGO_HTTP_TIMEOUT=60"
set "CARGO_HTTP_LOW_SPEED_LIMIT=1024"
set "CARGO_NET_RETRY=3"
set "PATH=%CARGO_HOME%\bin;%NITRO_ROOT%.tools\uv;%PATH%"
if not exist "%TEMP%" mkdir "%TEMP%"
"%NITRO_ROOT%.venv\Scripts\nitro.exe" %*
exit /b %ERRORLEVEL%
