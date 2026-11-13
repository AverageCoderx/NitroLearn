"""The cargo NitroLearn's builds use: the folder's own Rust in .tools (where the installer puts it, on no
PATH) is found and run with its homes, whether or not the launcher set them."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from nitro import export, toolchain

EXE = ".exe" if sys.platform == "win32" else ""


def _project_with_rust(tmp_path: Path) -> Path:
    project = tmp_path / "NitroLearn"
    cargo = project / ".tools" / "cargo" / "bin" / f"cargo{EXE}"
    cargo.parent.mkdir(parents=True)
    cargo.write_text("", encoding="utf-8")
    return project


def test_the_folders_own_rust_is_found_without_the_launchers_settings(tmp_path: Path) -> None:
    project = _project_with_rust(tmp_path)
    tools = project / ".tools"
    bare = {"PATH": str(tmp_path / "empty")}  # `.venv\Scripts\python -m nitro ...`: nothing set
    assert toolchain.find_cargo(project, bare) == tools / "cargo" / "bin" / f"cargo{EXE}"
    env = toolchain.cargo_env(project, bare)
    assert env["CARGO_HOME"] == str(tools / "cargo") and env["RUSTUP_HOME"] == str(tools / "rustup")
    assert env["PATH"].split(os.pathsep)[0] == str(tools / "cargo" / "bin"), "rustc and cargo of that Rust"
    # Set by nitro.cmd or tools/env.*: used as they are.
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "bin").mkdir(parents=True)
    (elsewhere / "bin" / f"cargo{EXE}").write_text("", encoding="utf-8")
    homes = {**bare, "CARGO_HOME": str(elsewhere), "RUSTUP_HOME": str(tmp_path / "rustup")}
    assert toolchain.find_cargo(project, homes) == elsewhere / "bin" / f"cargo{EXE}"
    assert toolchain.cargo_env(project, homes)["RUSTUP_HOME"] == str(tmp_path / "rustup")
    # No Rust in the folder: the PATH's, with cargo's own defaults; none at all: None.
    assert toolchain.rust_homes(tmp_path / "plain", bare) == {}
    assert toolchain.find_cargo(tmp_path / "plain", bare) is None
    assert toolchain.cargo_env(tmp_path / "plain", bare) == bare


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in cargo is a shell script")
def test_export_builds_the_bot_with_the_folders_own_rust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_rust(tmp_path)
    seen = tmp_path / "seen.txt"
    cargo = project / ".tools" / "cargo" / "bin" / "cargo"
    cargo.write_text(f'#!/bin/sh\necho "$RUSTUP_HOME|$CARGO_HOME" > "{seen}"\nexit 3\n', encoding="utf-8")
    cargo.chmod(0o755)
    monkeypatch.setattr(toolchain, "PROJECT", project)
    for name in ("CARGO_HOME", "RUSTUP_HOME", "CARGO_TARGET_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    program, why = export._build_bot(project)
    assert program is None and why.startswith("cargo build failed"), why
    tools = project / ".tools"
    assert seen.read_text(encoding="utf-8").strip() == f"{tools / 'rustup'}|{tools / 'cargo'}"
