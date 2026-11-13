"""Rust's cargo for the builds NitroLearn starts (the bot program, the replay converters, the core).

The installer puts Rust in NitroLearn's own folder (.tools\\cargo and .tools\\rustup) and changes no PATH, so
cargo is looked for there first: in CARGO_HOME when set (nitro.cmd, tools/env.ps1 and tools/env.sh set it),
else in the folder's .tools, else on the PATH. A cargo from .tools runs with CARGO_HOME and RUSTUP_HOME
pointing there, or its rustup could not find the toolchain. Standard library only: the release's notices
tool uses it before NitroLearn is installed.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
_EXE = ".exe" if sys.platform == "win32" else ""


def rust_homes(project: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """CARGO_HOME and RUSTUP_HOME for a build: as set, else the folder's .tools (`project`, default
    NitroLearn's own) when Rust is there, else none (cargo's own defaults)."""
    env = os.environ if env is None else env
    if env.get("CARGO_HOME"):
        return {name: env[name] for name in ("CARGO_HOME", "RUSTUP_HOME") if env.get(name)}
    tools = (project or PROJECT) / ".tools"
    if (tools / "cargo" / "bin" / f"cargo{_EXE}").is_file():
        return {"CARGO_HOME": str(tools / "cargo"), "RUSTUP_HOME": str(tools / "rustup")}
    return {}


def find_cargo(project: Path | None = None, env: Mapping[str, str] | None = None) -> Path | None:
    """The cargo program a build uses (`rust_homes`' first, else the PATH's), or None without Rust."""
    env = os.environ if env is None else env
    home = rust_homes(project, env).get("CARGO_HOME")
    if home:
        local = Path(home) / "bin" / f"cargo{_EXE}"
        if local.is_file():
            return local
    found = shutil.which("cargo", path=env.get("PATH"))
    return Path(found) if found else None


def cargo_env(project: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """`env` (default: this process's) for running cargo: `rust_homes` set and its programs first on the
    PATH, so cargo, rustc and the build tools that call them by name are the same Rust."""
    out = dict(os.environ if env is None else env)
    homes = rust_homes(project, out)
    out.update(homes)
    if "CARGO_HOME" in homes:
        out["PATH"] = os.pathsep.join([str(Path(homes["CARGO_HOME"]) / "bin"), out.get("PATH", "")])
    return out
