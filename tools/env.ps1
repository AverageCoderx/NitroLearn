# Keeps every tool, cache and temporary file inside this project folder, never elsewhere on the PC.
# Dot-source it from other scripts:  . "$PSScriptRoot\tools\env.ps1"   (or "$PSScriptRoot\env.ps1" from tools\)
# nitro.cmd sets the same variables for the `nitro` command; tools\env.sh holds them for Linux
# (python\nitro\folder_env.py).
$NitroRoot = Split-Path $PSScriptRoot -Parent

$env:UV_PYTHON_INSTALL_DIR = Join-Path $NitroRoot '.tools\python'
$env:UV_PYTHON_BIN_DIR = Join-Path $NitroRoot '.tools\python\bin'
$env:UV_CACHE_DIR = Join-Path $NitroRoot '.cache\uv'
$env:UV_TOOL_DIR = Join-Path $NitroRoot '.tools\uv-tools'
$env:UV_TOOL_BIN_DIR = Join-Path $NitroRoot '.tools\uv-tools\bin'
$env:RUSTUP_HOME = Join-Path $NitroRoot '.tools\rustup'
$env:CARGO_HOME = Join-Path $NitroRoot '.tools\cargo'
$env:PIP_CACHE_DIR = Join-Path $NitroRoot '.cache\pip'
$env:CUDA_CACHE_PATH = Join-Path $NitroRoot '.cache\cuda'
$env:TORCH_HOME = Join-Path $NitroRoot '.cache\torch'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $NitroRoot '.cache\torchinductor'
$env:TRITON_CACHE_DIR = Join-Path $NitroRoot '.cache\triton'
$env:WANDB_CONFIG_DIR = Join-Path $NitroRoot '.cache\wandb\config'
$env:WANDB_CACHE_DIR = Join-Path $NitroRoot '.cache\wandb\cache'
$env:WANDB_DATA_DIR = Join-Path $NitroRoot '.cache\wandb\data'
$env:WANDB_ARTIFACT_DIR = Join-Path $NitroRoot '.cache\wandb\artifacts'
$env:TEMP = Join-Path $NitroRoot '.tmp'
$env:TMP = Join-Path $NitroRoot '.tmp'
$env:UV_PYTHON_INSTALL_BIN = '0'
$env:UV_PYTHON_INSTALL_REGISTRY = '0'
$env:UV_MANAGED_PYTHON = '1'
$env:UV_NO_CONFIG = '1'
$env:UV_SYSTEM_CERTS = '1'
# A cargo download that stalls fails (and is tried again) instead of hanging.
$env:CARGO_HTTP_TIMEOUT = '60'
$env:CARGO_HTTP_LOW_SPEED_LIMIT = '1024'
$env:CARGO_NET_RETRY = '3'
$env:PATH = "$(Join-Path $env:CARGO_HOME 'bin');$(Join-Path $NitroRoot '.tools\uv');$env:PATH"

foreach ($dir in $env:TEMP, $env:PIP_CACHE_DIR, $env:CUDA_CACHE_PATH) {
    New-Item -ItemType Directory -Force $dir | Out-Null
}
# NVIDIA's driver reads CUDA_CACHE_PATH through an ANSI getenv, which turns letters outside the system's code
# page into look-alikes (and the driver makes that other folder). A path that is not plain ASCII is given as
# its 8.3 short form when that is plain ASCII, else left unset: the driver then keeps its cache in its own
# folder under %APPDATA%. python\nitro\folder_env.py (native_safe) does the same for every NitroLearn process.
if ($env:CUDA_CACHE_PATH -match '[^\x00-\x7F]') {
    $CudaShort = ''
    try { $CudaShort = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($env:CUDA_CACHE_PATH).ShortPath } catch {}
    if ($CudaShort -and $CudaShort -notmatch '[^\x00-\x7F]') {
        $env:CUDA_CACHE_PATH = $CudaShort
    } else {
        Remove-Item Env:\CUDA_CACHE_PATH -ErrorAction SilentlyContinue
    }
}
