"""The compiled core's fingerprint of its Rust code (`nitro.build_info`, crates/nitro_py/build.rs): an old
core next to new code is caught before it fails in confusing ways."""

from __future__ import annotations

from pathlib import Path

import pytest

from nitro import _core
from nitro.build_info import ROOT, SUFFIXES, CoreOutOfDate, check_core, core_status, source_hash


def test_the_built_core_carries_the_fingerprint_of_this_folders_code() -> None:
    # The test runs after `tools/check.sh` (or check.ps1) rebuilt the core, so the two must agree.
    assert source_hash() == _core.SOURCE_HASH
    assert core_status()[0]
    check_core()


def tree(root: Path, line_end: str) -> Path:
    for name in ("crates/nitro_core/src/lib.rs", "crates/nitro_py/src/lib.rs"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(f"fn main() {{{line_end}}}{line_end}".encode())
    for name in ("Cargo.toml", "Cargo.lock", "crates/nitro_core/Cargo.toml"):
        (root / name).write_text("[package]\n", encoding="utf-8")
    return root


def test_the_fingerprint_follows_the_code_and_nothing_else(tmp_path: Path) -> None:
    unix, windows = tree(tmp_path / "a", "\n"), tree(tmp_path / "b", "\r\n")
    assert source_hash(unix) == source_hash(windows), "a Windows checkout's line endings do not count"
    before = source_hash(unix)
    (unix / "crates/nitro_core/README.md").write_text("notes", encoding="utf-8")
    (unix / "crates/nitro_core/notes.h").write_text("#pragma once\n", encoding="utf-8")
    (unix / "crates/nitro_core/target").mkdir()
    (unix / "crates/nitro_core/target/built.rs").write_text("x", encoding="utf-8")
    assert source_hash(unix) == before, "only Rust code and manifests outside build folders count"
    (unix / "crates/nitro_core/src/lib.rs").write_text("fn main() { changed }\n", encoding="utf-8")
    assert source_hash(unix) != before
    assert source_hash(tmp_path / "not-a-checkout") is None


def test_an_old_core_is_refused_with_how_to_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    core_status.cache_clear()
    monkeypatch.setattr(_core, "SOURCE_HASH", "0000000000000000")
    try:
        with pytest.raises(CoreOutOfDate, match=r"Install NitroLearn\.cmd"):
            check_core()
    finally:
        core_status.cache_clear()


def test_the_build_script_and_this_module_list_the_same_files() -> None:
    script = (ROOT / "crates/nitro_py/build.rs").read_text(encoding="utf-8")
    assert 'const DIRS: [&str; 2] = ["crates/nitro_core", "crates/nitro_py"];' in script
    assert 'const FILES: [&str; 2] = ["Cargo.toml", "Cargo.lock"];' in script
    assert '.is_some_and(|ext| ext == "rs" || ext == "toml")' in script
    assert SUFFIXES == (".rs", ".toml")


def test_linked_folders_count_as_the_build_script_counts_them(tmp_path: Path) -> None:
    root = tree(tmp_path / "a", "\n")
    before = source_hash(root)
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "extra.rs").write_text("fn extra() {}\n", encoding="utf-8")
    try:
        (root / "crates/nitro_core/src/linked").symlink_to(tmp_path / "shared", target_is_directory=True)
    except OSError:
        pytest.skip("this machine does not allow symbolic links")
    assert source_hash(root) != before, "files in a linked folder are part of the code"


def test_the_check_can_be_skipped_where_the_fingerprint_cannot_match(monkeypatch: pytest.MonkeyPatch) -> None:
    core_status.cache_clear()
    monkeypatch.setattr(_core, "SOURCE_HASH", "0000000000000000")
    monkeypatch.setenv("NITRO_SKIP_CORE_CHECK", "1")
    try:
        check_core()
    finally:
        core_status.cache_clear()
