"""Whether the compiled Rust core was built from the Rust code in this folder. After a `git pull` that
changed that code the core must be rebuilt (the installer again, or `pip install --editable .`); an old
core with new Python code fails in confusing ways.

    python -m nitro.build_info      exit code 0 = up to date (or not a source checkout), 1 = rebuild

`crates/nitro_py/build.rs` computes the same fingerprint when the core is built; keep the two in step.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = ("crates/nitro_core", "crates/nitro_py")
FILES = ("Cargo.toml", "Cargo.lock")
# Rust code and its manifests.
SUFFIXES = (".rs", ".toml")
REBUILD = (
    "rebuild it: run the installer again (Install NitroLearn.cmd, or install.sh on Linux), or "
    "`python -m pip install --editable .` in this folder"
)


class CoreOutOfDate(RuntimeError):
    """The compiled core is older than the Rust code next to it."""


def _fnv1a(hash_value: int, data: bytes) -> int:
    for byte in data:
        hash_value = ((hash_value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def source_hash(root: Path = ROOT, dirs: Sequence[str] = DIRS) -> str | None:
    """The fingerprint of the Rust code in `dirs` under `root` (the core's by default); None when `root` is
    not a source checkout."""
    if not (root / dirs[0]).is_dir():
        return None
    files = []
    for folder in dirs:
        # Linked folders are followed, as the build script follows them.
        for here, subfolders, names in os.walk(root / folder, followlinks=True):
            subfolders[:] = [name for name in subfolders if name != "target"]
            files += [Path(here) / name for name in names if Path(name).suffix in SUFFIXES]
    files += [root / name for name in FILES]
    value = 0xCBF29CE484222325
    for name, path in sorted((path.relative_to(root).as_posix(), path) for path in files):
        text = path.read_bytes().replace(b"\r\n", b"\n") if path.is_file() else b""
        value = _fnv1a(value, name.encode() + b"\0" + text + b"\0")
    return f"{value:016x}"


@cache
def core_status() -> tuple[bool, str]:
    """(up to date, a sentence saying so). A core built before the fingerprint existed counts as old."""
    from nitro import _core

    expected = source_hash()
    if expected is None:
        return True, "not a source checkout; nothing to compare"
    built = getattr(_core, "SOURCE_HASH", None)
    if built == expected:
        return True, f"built from the code in this folder ({built})"
    return False, "the compiled core (the Rust part) is older than the code in this folder, " + REBUILD


def check_core() -> None:
    """Raises CoreOutOfDate when the core must be rebuilt first. NITRO_SKIP_CORE_CHECK=1 skips the check,
    for a machine where the fingerprint cannot match whatever is done."""
    if os.environ.get("NITRO_SKIP_CORE_CHECK") == "1":
        return
    current, message = core_status()
    if not current:
        raise CoreOutOfDate(message[0].upper() + message[1:] + ".")


if __name__ == "__main__":
    up_to_date, sentence = core_status()
    print(f"Rust core: {sentence}")
    sys.exit(0 if up_to_date else 1)
