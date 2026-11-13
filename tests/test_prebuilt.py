"""The release's ready-made programs (.tools/prebuilt): `nitro export` and the replay converters use them
instead of building with cargo, but only while prebuilt.json says they were built from this folder's code."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nitro import export, prebuilt, replays

PROJECT = Path(__file__).resolve().parents[1]


def _code(root: Path) -> Path:
    """A small NitroLearn folder: the bot program's crates and nitro-replicar's sources."""
    for name, text in (
        ("Cargo.toml", "[workspace]\n"),
        ("Cargo.lock", "# lock\n"),
        ("crates/nitro_bot/src/main.rs", "fn main() {}\n"),
        ("crates/nitro_core/src/lib.rs", "pub fn core() {}\n"),
        ("crates/nitro_replicar/Cargo.toml", "[package]\n"),
        ("crates/nitro_replicar/Cargo.lock", "# lock\n"),
        ("crates/nitro_replicar/src/main.rs", "fn main() {}\n"),
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def _ship(root: Path, programs: dict[str, str]) -> Path:
    folder = root / prebuilt.FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    for name in programs:
        (folder / prebuilt.file_name(name)).write_bytes(f"ready-made {name}".encode())
    (folder / prebuilt.LIST).write_text(json.dumps({"programs": programs}), encoding="utf-8")
    return folder


def test_a_ready_made_program_is_used_only_for_the_code_it_was_built_from(tmp_path: Path) -> None:
    root = _code(tmp_path / "NitroLearn")
    built_from = prebuilt.fingerprints(root)
    assert built_from["replicar"] == replays.REPLICAR_VERSION and len(built_from["nitro_bot"]) == 16
    folder = _ship(root, built_from)
    assert prebuilt.program("nitro_bot", prebuilt.bot_source(root), root) == folder / prebuilt.file_name(
        "nitro_bot"
    )
    # Line endings do not count: a Windows checkout of the same code matches.
    main_rs = root / "crates" / "nitro_replicar" / "src" / "main.rs"
    main_rs.write_bytes(b"fn main() {}\r\n")
    assert prebuilt.nitro_replicar_source(root) == built_from["nitro-replicar"]
    # Changed code: not used.
    (root / "crates" / "nitro_core" / "src" / "lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")
    assert prebuilt.program("nitro_bot", prebuilt.bot_source(root), root) is None
    assert prebuilt.program("replicar", "2.0.4", root) is None
    (folder / prebuilt.LIST).write_text("not json", encoding="utf-8")
    assert prebuilt.program("replicar", replays.REPLICAR_VERSION, root) is None
    assert prebuilt.program("replicar", replays.REPLICAR_VERSION, tmp_path / "elsewhere") is None


def test_export_copies_the_ready_made_bot_program_without_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _code(tmp_path / "NitroLearn")
    _ship(root, prebuilt.fingerprints(root))
    monkeypatch.setattr(export, "_build_bot", lambda project: pytest.fail("no cargo build"))
    exe, message = export._bot_program(root)
    assert exe is not None and exe.read_bytes() == b"ready-made nitro_bot" and message == ""
    # Other code: built with cargo, as before.
    (root / "crates" / "nitro_bot" / "src" / "main.rs").write_text("fn main() { 1; }\n", encoding="utf-8")
    monkeypatch.setattr(export, "_build_bot", lambda project: (None, "built with cargo"))
    assert export._bot_program(root) == (None, "built with cargo")
    # Without a ready-made program and without cargo, the message says what to do.
    monkeypatch.undo()
    monkeypatch.setattr(export, "_cargo", lambda: None)
    exe, message = export._bot_program(root)
    assert exe is None and "Run the installer again" in message
    assert "Tools to build your own Rust rewards" in message


def test_the_replay_converters_use_the_ready_made_ones_without_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "NitroLearn"
    for part in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
        (root / prebuilt.NITRO_REPLICAR / part).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT / prebuilt.NITRO_REPLICAR / part, root / prebuilt.NITRO_REPLICAR / part)
    monkeypatch.setattr(replays, "ROOT", root)
    monkeypatch.setattr(replays, "REPLICAR_ROOT", root / ".scratch" / "replicar")
    monkeypatch.setattr(replays, "NITRO_REPLICAR_SOURCE", root / "crates" / "nitro_replicar")
    monkeypatch.setattr(replays, "NITRO_REPLICAR_ROOT", root / ".scratch" / "nitro-replicar")
    monkeypatch.setattr(replays, "find_cargo", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("nothing is built"))
    with pytest.raises(replays.ReplayError, match=r"cargo is needed.*run the installer again"):
        replays.replicar_tool()
    with pytest.raises(replays.ReplayError, match="cargo is needed"):
        replays.nitro_replicar_tool()
    folder = _ship(
        root,
        {"replicar": replays.REPLICAR_VERSION, "nitro-replicar": prebuilt.nitro_replicar_source(root)},
    )
    assert replays.replicar_tool() == folder / prebuilt.file_name("replicar")
    assert replays.nitro_replicar_tool() == folder / prebuilt.file_name("nitro-replicar")


def test_the_release_build_writes_the_list(tmp_path: Path) -> None:
    """What the release workflow runs: `python -m nitro.prebuilt <folder> <programs>`."""
    done = subprocess.run(
        [sys.executable, "-m", "nitro.prebuilt", str(PROJECT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    listed = json.loads((tmp_path / prebuilt.LIST).read_text(encoding="utf-8"))
    assert listed == {"programs": prebuilt.fingerprints(PROJECT)}
