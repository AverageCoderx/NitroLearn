"""The environment that keeps every tool, cache and temporary file inside NitroLearn's folder: uv, Rust,
pip, PyTorch, Weights & Biases and the temporary folder, plus cargo's network limits. The installer
(tools/install_steps.py) and the dashboard's windowless start (nitro.dashboard.winapp) use it; tools/env.ps1
and nitro.cmd (Windows) and tools/env.sh (Linux) set the same, which a test keeps equal.

Standard library only: the installer loads this file before NitroLearn is installed.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path

WINDOWS = sys.platform == "win32"
# Read by NVIDIA's driver through the C library's ANSI getenv: a path with letters outside the system's ANSI
# code page reaches it altered (Windows swaps them for look-alikes) and the driver makes that other folder.
NATIVE_ANSI = ("CUDA_CACHE_PATH",)
# cargo's network limits: a download that stalls (under 1 KB/s for a minute, or no answer for a minute) fails
# after 3 tries instead of hanging.
CARGO_NETWORK = {"CARGO_HTTP_TIMEOUT": "60", "CARGO_HTTP_LOW_SPEED_LIMIT": "1024", "CARGO_NET_RETRY": "3"}


def variables(root: Path, windows: bool = WINDOWS) -> dict[str, str]:
    """Every folder a tool would otherwise write to outside NitroLearn, pointed inside `root`, and cargo's
    network limits."""
    inside = {
        "UV_PYTHON_INSTALL_DIR": ".tools/python",
        "UV_PYTHON_BIN_DIR": ".tools/python/bin",
        "UV_CACHE_DIR": ".cache/uv",
        "UV_TOOL_DIR": ".tools/uv-tools",
        "UV_TOOL_BIN_DIR": ".tools/uv-tools/bin",
        "RUSTUP_HOME": ".tools/rustup",
        "CARGO_HOME": ".tools/cargo",
        "PIP_CACHE_DIR": ".cache/pip",
        "CUDA_CACHE_PATH": ".cache/cuda",
        "TORCH_HOME": ".cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
        "TRITON_CACHE_DIR": ".cache/triton",
        "WANDB_CONFIG_DIR": ".cache/wandb/config",
        "WANDB_CACHE_DIR": ".cache/wandb/cache",
        "WANDB_DATA_DIR": ".cache/wandb/data",
        "WANDB_ARTIFACT_DIR": ".cache/wandb/artifacts",
    }
    if windows:
        inside |= {"TEMP": ".tmp", "TMP": ".tmp"}
    else:
        inside |= {"TMPDIR": ".tmp", "XDG_CACHE_HOME": ".cache"}
    env = {name: str(root.joinpath(*path.split("/"))) for name, path in inside.items()}
    # Python is never added to PATH or the Windows registry; only Pythons uv put here are used.
    env |= {
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_PYTHON_INSTALL_REGISTRY": "0",
        "UV_MANAGED_PYTHON": "1",
        "UV_NO_CONFIG": "1",
        "UV_SYSTEM_CERTS": "1",
    }
    return env | CARGO_NETWORK


def short_path(path: str) -> str | None:
    """Windows' 8.3 short form of an existing file or folder; None where there is none (another system, a
    drive with short names switched off, a missing folder)."""
    if not WINDOWS:
        return None
    import ctypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined, unused-ignore]
        size = kernel32.GetShortPathNameW(path, None, 0)
        if not size:
            return None
        buffer = ctypes.create_unicode_buffer(size)
        if not kernel32.GetShortPathNameW(path, buffer, size):
            return None
    except (AttributeError, OSError):
        return None
    return str(buffer.value)


def native_safe(
    env: MutableMapping[str, str], windows: bool = WINDOWS, short: Callable[[str], str | None] = short_path
) -> None:
    """Makes the NATIVE_ANSI variables in `env` safe for NVIDIA's driver: a path in plain ASCII stays, any
    other becomes its 8.3 short form when that is plain ASCII, else the variable is removed (the driver then
    keeps its cache in its own folder under %APPDATA%, outside NitroLearn's folder). The folder is made first,
    since only an existing folder has a short form. tools/env.ps1 does the same."""
    if not windows:
        return
    for name in NATIVE_ANSI:
        value = env.get(name)
        if value is None or value.isascii():
            continue
        with contextlib.suppress(OSError):
            Path(value).mkdir(parents=True, exist_ok=True)
        shorter = short(value)
        if shorter and shorter.isascii():
            env[name] = shorter
        else:
            del env[name]


def apply(root: Path, env: MutableMapping[str, str] | None = None, windows: bool = WINDOWS) -> None:
    """Sets `variables` in `env` (this process's environment by default), puts the folder's cargo and uv first
    on its PATH and makes the folders that must exist, as tools/env.ps1 and tools/env.sh do; then
    `native_safe`."""
    env = os.environ if env is None else env
    values = variables(root, windows)
    env.update(values)
    first = [str(Path(values["CARGO_HOME"]) / "bin"), str(root / ".tools" / "uv")]
    env["PATH"] = os.pathsep.join([*first, env["PATH"]] if env.get("PATH") else first)
    temp = values["TEMP" if windows else "TMPDIR"]
    for folder in (temp, values["PIP_CACHE_DIR"], values["CUDA_CACHE_PATH"]):
        Path(folder).mkdir(parents=True, exist_ok=True)
    native_safe(env, windows)
