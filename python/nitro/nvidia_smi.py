"""Where NVIDIA's `nvidia-smi` is, and which NVIDIA graphics cards the system lists even when the driver does
not answer: the installer (tools/install_steps.py) and `nitro doctor` use it.

Standard library only: the installer loads this file before NitroLearn is installed.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

WINDOWS = sys.platform == "win32"
NVIDIA_VENDOR = "VEN_10DE"  # NVIDIA's PCI vendor id, in a Windows device id
# Windows' display adapters as "name|device id" lines (also those running Windows' basic driver).
ADAPTERS = (
    "Get-CimInstance Win32_VideoController | "
    "ForEach-Object { [string]$_.Name + '|' + [string]$_.PNPDeviceID }"
)


def places(env: Mapping[str, str], windows: bool = WINDOWS) -> list[Path]:
    """Where NVIDIA's Windows driver puts nvidia-smi.exe, when it is not on the PATH: System32 (current
    drivers), NVSMI under Program Files (older drivers), and the driver's own folder in the DriverStore (where
    System32's copy comes from), newest first."""
    if not windows:
        return []
    system = Path(env.get("SYSTEMROOT") or env.get("WINDIR") or r"C:\Windows")
    found = [system / "System32" / "nvidia-smi.exe"]
    for name in ("ProgramW6432", "ProgramFiles"):
        if env.get(name):
            found.append(Path(env[name]) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")
    store = system / "System32" / "DriverStore" / "FileRepository"
    with contextlib.suppress(OSError):
        found += sorted(store.glob("nv*/nvidia-smi.exe"), key=_age)
    unique: list[Path] = []
    for place in found:
        if place not in unique:
            unique.append(place)
    return unique


def _age(path: Path) -> float:
    try:
        return -path.stat().st_mtime
    except OSError:
        return 0.0


def find(
    env: Mapping[str, str] | None = None,
    windows: bool = WINDOWS,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    """nvidia-smi: on the PATH, else where the Windows driver puts it; None when there is none."""
    on_path = which("nvidia-smi")
    if on_path:
        return on_path
    for place in places(os.environ if env is None else env, windows):
        if place.is_file():
            return str(place)
    return None


def parse_adapters(text: str) -> list[str]:
    """The NVIDIA cards in Windows' "name|device id" lines (ADAPTERS) or in `lspci`'s lines. A card running
    Windows' basic driver is named by its device id only, so it is called "an NVIDIA graphics card"."""
    names = []
    for line in text.splitlines():
        name, bar, device = line.partition("|")
        if bar:
            if NVIDIA_VENDOR in device.upper() or "NVIDIA" in name.upper():
                names.append(name.strip() if "NVIDIA" in name.upper() else "an NVIDIA graphics card")
        elif re.search(r"(VGA compatible|3D) controller", line) and "NVIDIA" in line.upper():
            names.append(line.split(":", 2)[-1].strip())
    return names


def adapters(
    windows: bool = WINDOWS, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> list[str]:
    """The NVIDIA graphics cards the system lists, whether or not NVIDIA's driver answers: Windows' own list
    of display adapters, or `lspci` on Linux. [] when there are none or the list cannot be read."""
    if windows:
        system = Path(os.environ.get("SYSTEMROOT") or r"C:\Windows")
        shell = system / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        command = [str(shell) if shell.is_file() else "powershell", "-NoProfile", "-NonInteractive"]
        command += ["-Command", ADAPTERS]
    elif shutil.which("lspci"):
        command = ["lspci"]
    else:
        return []
    try:
        done = run(command, capture_output=True, text=True, timeout=60, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_adapters(done.stdout or "") if done.returncode == 0 else []
