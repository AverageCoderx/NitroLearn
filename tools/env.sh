# Keeps every tool, cache and temporary file inside NitroLearn's folder, never elsewhere on the computer.
# Source it (nitro.sh and install.sh do):  . tools/env.sh
# For a shell with NitroLearn's commands:  . tools/env.sh && . .venv/bin/activate
# tools/env.ps1 and nitro.cmd hold the same settings for Windows (python/nitro/folder_env.py).
NITRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
export NITRO_ROOT
export UV_PYTHON_INSTALL_DIR="$NITRO_ROOT/.tools/python"
export UV_PYTHON_BIN_DIR="$NITRO_ROOT/.tools/python/bin"
export UV_CACHE_DIR="$NITRO_ROOT/.cache/uv"
export UV_TOOL_DIR="$NITRO_ROOT/.tools/uv-tools"
export UV_TOOL_BIN_DIR="$NITRO_ROOT/.tools/uv-tools/bin"
export RUSTUP_HOME="$NITRO_ROOT/.tools/rustup"
export CARGO_HOME="$NITRO_ROOT/.tools/cargo"
export PIP_CACHE_DIR="$NITRO_ROOT/.cache/pip"
export CUDA_CACHE_PATH="$NITRO_ROOT/.cache/cuda"
export TORCH_HOME="$NITRO_ROOT/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$NITRO_ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$NITRO_ROOT/.cache/triton"
export WANDB_CONFIG_DIR="$NITRO_ROOT/.cache/wandb/config"
export WANDB_CACHE_DIR="$NITRO_ROOT/.cache/wandb/cache"
export WANDB_DATA_DIR="$NITRO_ROOT/.cache/wandb/data"
export WANDB_ARTIFACT_DIR="$NITRO_ROOT/.cache/wandb/artifacts"
export TMPDIR="$NITRO_ROOT/.tmp"
export XDG_CACHE_HOME="$NITRO_ROOT/.cache"
export UV_PYTHON_INSTALL_BIN=0
export UV_PYTHON_INSTALL_REGISTRY=0
export UV_MANAGED_PYTHON=1
export UV_NO_CONFIG=1
export UV_SYSTEM_CERTS=1
# A cargo download that stalls fails (and is tried again) instead of hanging.
export CARGO_HTTP_TIMEOUT=60
export CARGO_HTTP_LOW_SPEED_LIMIT=1024
export CARGO_NET_RETRY=3
export PATH="$CARGO_HOME/bin:$NITRO_ROOT/.tools/uv:$PATH"
mkdir -p "$TMPDIR"
