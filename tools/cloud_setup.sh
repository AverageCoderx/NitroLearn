#!/usr/bin/env bash
# Sets NitroLearn up on a Linux machine with an NVIDIA graphics card (such as a rented cloud machine), from
# inside NitroLearn's folder:
#
#     bash tools/cloud_setup.sh [--cpu] [<bot>.nitropack.zip]
#
# --cpu: a machine without a graphics card (for trying it out). With a pack (`nitro pack` on your PC), it
# also unpacks the bot and the arena meshes the pack carries.
#
# It installs what is missing (a C/C++ compiler, curl, tmux, unzip, nano, the Rust version rust-toolchain.toml
# pins, uv, which fetches Python 3.11 if the machine's is older), builds NitroLearn with the PyTorch build that
# matches the machine's NVIDIA driver and card (CUDA 13.0, 12.8 or 12.6: cards of compute capability 12.0 need 12.8 or
# newer, cards older than compute capability 7.5 need 12.6), checks that the graphics card really computes,
# unpacks the bot (once: run again, it carries on), fetches the arena meshes (nitro fetch-meshes) if the pack brought none, and
# runs `nitro doctor`. Safe to run again.
set -euo pipefail
say() { printf '\n== %s\n' "$*"; }
cpu=0
packed=""
for arg in "$@"; do  # before moving to NitroLearn's folder: a pack's path is from the folder you are in
    case "$arg" in
        --cpu) cpu=1 ;;
        *)
            if [ ! -f "$arg" ]; then
                echo "No pack file at $arg" >&2
                exit 1
            fi
            packed=$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")
            ;;
    esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

sudo=""
if [ "$(id -u)" != 0 ]; then
    if command -v sudo > /dev/null; then sudo=sudo; fi
fi
need() {  # need <command> <apt package>: installs the package when the command is missing
    command -v "$1" > /dev/null && return 0
    if ! command -v apt-get > /dev/null || { [ "$(id -u)" != 0 ] && [ -z "$sudo" ]; }; then
        echo "Needs $1: install the package $2 (this script can only use apt-get, as root or with sudo)." >&2
        exit 1
    fi
    say "Installing $2"
    $sudo apt-get update -qq > /dev/null
    DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq "$2" > /dev/null
}
need c++ build-essential
need curl curl
need tmux tmux
need unzip unzip
need nano nano

# The Rust rust-toolchain.toml pins, through rustup (a system cargo without rustup is passed over).
pin=$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)
if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
if ! command -v rustup > /dev/null; then
    say "Installing Rust $pin"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$pin"
    if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
elif ! rustup toolchain list | grep -q "^$pin-"; then
    say "Installing Rust $pin"
    rustup toolchain install "$pin" --profile minimal --component rustfmt --component clippy
fi
if [ "$({ cargo --version 2> /dev/null || true; } | cut -d' ' -f2)" != "$pin" ]; then
    echo "Rust $pin is needed, and installing it did not work: see the lines above." >&2
    exit 1
fi
if ! command -v uv > /dev/null; then
    [ -x "$HOME/.local/bin/uv" ] || { say "Installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
    export PATH="$HOME/.local/bin:$PATH"
fi

# The PyTorch build for the driver and the card: `nvidia-smi` names the newest CUDA the driver runs and the
# card's generation (compute capability). PyTorch's CUDA 12.8 and 13.0 builds have no code for cards older
# than compute capability 7.5; cards of capability 12.0 need 12.8 or newer.
index=https://download.pytorch.org/whl/cpu
if [ "$cpu" = 0 ]; then
    if ! command -v nvidia-smi > /dev/null; then
        echo "No NVIDIA driver here (nvidia-smi is missing): rent a machine with an NVIDIA card, or use --cpu." >&2
        exit 1
    fi
    if ! smi=$(nvidia-smi 2>&1); then
        printf 'The NVIDIA driver does not work on this machine (nvidia-smi says: %s): pick another machine.\n' "$smi" >&2
        exit 1
    fi
    # "CUDA Version: 13.0" in its header; drivers from 615 on say "CUDA UMD Version: 13.4".
    cuda=$(printf '%s\n' "$smi" | sed -n -E '/CUDA (UMD )?Version/{s/.*CUDA (UMD )?Version: *([0-9]+)\.([0-9]+).*/\2 \3/p;q;}')
    if [ -z "$cuda" ]; then
        echo "nvidia-smi does not say which CUDA version the driver runs: pick another machine." >&2
        exit 1
    fi
    read -r major minor <<< "$cuda"
    driver=$((major * 100 + minor))
    cap=$({ nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2> /dev/null || true; } | tr -d ' ')
    card=75
    if [[ "$cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then card=$((BASH_REMATCH[1] * 10 + BASH_REMATCH[2])); fi
    if [ "$card" -lt 50 ]; then
        echo "This graphics card (generation $cap) is too old for PyTorch: pick another machine." >&2
        exit 1
    elif [ "$card" -ge 100 ] && [ "$driver" -lt 1300 ]; then
        echo "This card needs a driver that runs CUDA 13.0 or newer; this one runs ${major}.${minor}: pick another machine." >&2
        exit 1
    elif [ "$driver" -lt 1206 ]; then
        echo "The NVIDIA driver here runs CUDA ${major}.${minor}; NitroLearn needs 12.6 or newer: pick another machine." >&2
        exit 1
    elif [ "$card" -lt 75 ]; then index=https://download.pytorch.org/whl/cu126
    elif [ "$driver" -ge 1300 ]; then index=https://download.pytorch.org/whl/cu130
    else index=https://download.pytorch.org/whl/cu126
    fi
fi

say "Python and PyTorch ($index)"
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -c 'import torch' 2> /dev/null || uv pip install --quiet torch --index-url "$index" || {
    echo "PyTorch's index could not be reached; installing PyPI's build instead." >&2
    uv pip install --quiet torch
}

say "Building NitroLearn (a few minutes the first time)"
# pip and maturin: `nitro rebuild` builds the core again with a bot's own Rust rewards.
uv pip install --quiet pip maturin
uv pip install --quiet -e '.[wandb,replays]'

if [ -n "$packed" ]; then
    # Already unpacked from this same pack (the script run again): carry on with the checks.
    if python -c 'import sys; from pathlib import Path; from nitro.pack import unpacked_here
sys.exit(0 if unpacked_here(Path(sys.argv[1]), Path("runs")) else 1)' "$packed"; then
        say "$(basename "$packed") is already unpacked here"
    else
        say "Unpacking $(basename "$packed")"
        nitro unpack "$packed"
    fi
fi
if [ ! -d collision_meshes/soccar ]; then
    say "Arena meshes"
    nitro fetch-meshes
fi

say "Checking the machine"
nitro doctor || true
python - "$cpu" << 'EOF'
import os
import sys

import torch

threads = len(os.sched_getaffinity(0))
v1 = ["/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"]
for files in (["/sys/fs/cgroup/cpu.max"], v1):
    try:  # a container's CPU quota: a rented machine may give a share of a big processor
        quota, period = " ".join(open(f).read() for f in files).split()[:2]
    except (OSError, ValueError):
        continue
    if quota not in ("max", "-1"):
        threads = min(threads, max(1, -(-int(quota) // int(period))))
    break
print(f"Processor threads NitroLearn may use: {threads} (of {os.cpu_count()} on the processor)")
if sys.argv[1] == "1":
    print("Graphics card: not used (--cpu)")
    sys.exit(0)
if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the graphics card: this machine cannot train NitroLearn bots on it.")
name = torch.cuda.get_device_name(0)
try:  # a build without code for this card fails here ("no kernel image"), not at import
    torch.ones(4, device="cuda").mul(2).sum().item()
except RuntimeError as err:
    sys.exit(
        f"{name}: PyTorch {torch.__version__} cannot compute on it ({err}). Pick another machine (an NVIDIA "
        "card of compute capability 7.5 or newer); if this one should work, delete .venv and run this again."
    )
print(f"Graphics card: {name}, computing with PyTorch {torch.__version__}")
EOF

cat << EOF

NitroLearn is ready in $root. Next:
  . .venv/bin/activate
  nitro unpack <your pack>.nitropack.zip # if you did not give it to this script
  wandb login                            # once, if the bot uses W&B
  nitro resume <bot>                     # inside tmux, so it keeps going when you disconnect
EOF
