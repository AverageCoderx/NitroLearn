"""What the dashboard says about this PC without loading PyTorch (Check this PC does that, in its own
process): whether training here can use an NVIDIA graphics card, and which system it runs on."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from functools import cache
from pathlib import Path
from typing import Any

_CUDA = re.compile(r"""^cuda\s*(?::[^=]*)?=\s*['"]([\d.]+)['"]""", re.MULTILINE)


def torch_cuda(version_file: Path | None = None) -> bool | None:
    """Whether the installed PyTorch is built for NVIDIA graphics cards (CUDA), read from its version file;
    None when PyTorch cannot be found."""
    if version_file is None:
        try:
            spec = importlib.util.find_spec("torch")
        except (ImportError, ValueError):
            return None
        if spec is None or not spec.origin:
            return None
        version_file = Path(spec.origin).with_name("version.py")
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError:
        return None
    return _CUDA.search(text) is not None


def nvidia_driver() -> bool:
    """Whether an NVIDIA driver is installed: its `nvidia-smi` program, or on Linux its kernel module."""
    if shutil.which("nvidia-smi"):
        return True
    if sys.platform == "win32":
        system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
        return (system / "nvidia-smi.exe").is_file() or (system / "nvcuda.dll").is_file()
    return Path("/proc/driver/nvidia/version").is_file()


@cache
def graphics_card() -> bool | None:
    """Whether training on this PC runs on a graphics card: PyTorch built for one and an NVIDIA driver
    there. None when PyTorch cannot be found (the page then says nothing either way)."""
    built = torch_cuda()
    if built is None:
        return None
    return built and nvidia_driver()


def facts() -> dict[str, Any]:
    """The page's facts about this PC: {"graphics_card": True, False or None, "windows": bool}."""
    return {"graphics_card": graphics_card(), "windows": sys.platform == "win32"}
