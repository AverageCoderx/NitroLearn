"""The release's THIRD_PARTY_NOTICES.md: every crate linked into each ready-made program, the licence each is
used under, and the copyright notices, NOTICE files and licence texts those licences ask for."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nitro import notices
from nitro.notices import Component, NoticesError

PROJECT = Path(__file__).resolve().parents[1]
TOOL = PROJECT / "python" / "nitro" / "notices.py"


@pytest.mark.parametrize(
    ("expression", "used"),
    [
        ("MIT OR Apache-2.0", {"MIT"}),
        ("MIT/Apache-2.0", {"MIT"}),
        ("Apache-2.0 / MIT", {"MIT"}),
        ("Unlicense OR MIT", {"MIT"}),
        ("BSD-2-Clause OR Apache-2.0 OR MIT", {"MIT"}),
        ("Apache-2.0 WITH LLVM-exception OR MIT", {"MIT"}),
        ("(Apache-2.0 OR MIT) AND BSD-3-Clause", {"MIT", "BSD-3-CLAUSE"}),
        ("Apache-2.0 AND MIT", {"APACHE-2.0", "MIT"}),
        ("Apache-2.0", {"APACHE-2.0"}),
        ("Zlib", {"ZLIB"}),
        ("Unicode-3.0", None),
        ("(MIT OR Apache-2.0) AND Unicode-3.0", None),
    ],
)
def test_each_crate_is_used_under_the_first_licence_it_offers(expression: str, used: set[str] | None) -> None:
    chosen = notices._chosen(expression)
    assert (set(chosen) if chosen is not None else None) == used


def _rows(tmp_path: Path) -> list[Component]:
    apache_only = tmp_path / "arrow"
    either = tmp_path / "serde"
    for folder in (apache_only, either):
        folder.mkdir()
        (folder / "NOTICE.txt").write_text(f"{folder.name} NOTICE\n", encoding="utf-8")
    return [
        Component("arrow 1.0.0", "Apache-2.0", "https://example.org/arrow", apache_only),
        Component("serde 1.0.0", "MIT OR Apache-2.0", "Copyright (c) Serde", either),
        Component("foldhash 0.2.0", "Zlib", "Copyright (c) 2024 Orson Peters", None),
    ]


def test_the_programs_notices_carry_what_each_licence_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _rows(tmp_path)
    graphs = {"bot": rows[1:], "converter": rows[:2]}
    monkeypatch.setattr(
        notices, "_components", lambda project, select, target, strict=False: graphs[select[0]]
    )
    text = notices.program_notices(PROJECT, {"nitro_bot": ["bot"], "replicar": ["converter"]}, "2.2.1")
    assert "## nitro_bot\n\n- serde 1.0.0: MIT OR Apache-2.0\n- foldhash 0.2.0: Zlib\n" in text
    assert "## replicar\n\n- arrow 1.0.0: Apache-2.0\n- serde 1.0.0: MIT OR Apache-2.0\n" in text
    assert "replicar-cli 2.2.1" in text and next(iter(notices.required_notices(PROJECT))) in text
    assert "- foldhash 0.2.0: Copyright (c) 2024 Orson Peters" in text
    # Apache-2.0 in full, and the NOTICE file of the crate used under it (not of one used under MIT).
    apache = (PROJECT / notices.APACHE_FILE).read_text(encoding="utf-8")
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in apache and apache.strip() in text
    assert "### For arrow 1.0.0\n\n```\narrow NOTICE\n```" in text and "serde NOTICE" not in text
    for heading in ("### MIT License", "### zlib License", "### Apache License, Version 2.0"):
        assert heading in text
    assert "### BSD 3-Clause License" not in text and "The Unlicense" not in text, "only the texts used"
    assert re.search(r"[A-Z]:\\|/home/|/root/", text) is None, "no folder of the building computer"


def test_the_programs_notices_are_complete_or_not_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _rows(tmp_path)
    monkeypatch.setattr(notices, "_components", lambda project, select, target, strict=False: [])
    with pytest.raises(NoticesError, match="could not list the crates linked into nitro_bot"):
        notices.program_notices(PROJECT, {"nitro_bot": ["-p", "nitro_bot"]}, "2.2.1")
    odd = [*rows, Component("icu 2.0.0", "Unicode-3.0", "", None)]
    monkeypatch.setattr(notices, "_components", lambda project, select, target, strict=False: odd)
    with pytest.raises(NoticesError, match=r"no licence text here for: icu 2\.0\.0 \(Unicode-3\.0\)"):
        notices.program_notices(PROJECT, {"nitro_bot": ["-p", "nitro_bot"]}, "2.2.1")
    monkeypatch.setattr(notices, "_components", lambda project, select, target, strict=False: rows)
    with pytest.raises(NoticesError, match=r"Apache-2\.0's text is missing"):
        notices.program_notices(tmp_path, {"nitro_bot": ["-p", "nitro_bot"]}, "2.2.1")
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / "cargo"))
    with pytest.raises(NoticesError, match=r"replicar-cli 2\.2\.1 is not in cargo's cache"):
        notices.programs(PROJECT, "2.2.1")
    found = tmp_path / "cargo" / "registry" / "src" / "index" / "replicar-cli-2.2.1" / "Cargo.toml"
    found.parent.mkdir(parents=True)
    found.write_text("", encoding="utf-8")
    assert notices.programs(PROJECT, "2.2.1")["replicar"] == ["--manifest-path", str(found)]


def _need_cached(select: list[str]) -> None:
    """Skips unless cargo's cache holds every crate `select` may use, on every platform: the listing reads
    them offline, and what a computer happens to have downloaded is not what is tested (the release
    downloads them first, with --fetch)."""
    if not notices._packages(PROJECT, tuple(select)):
        pytest.skip(
            f"not every crate of {select[-1]} is in cargo's cache here (cargo fetch --locked adds them)"
        )


def test_without_every_crates_details_the_release_notices_are_refused_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crate whose source is not downloaded has no licence to read: the release's list stops and says how
    to download them, rather than list crates without licences (a bot folder's notices say less instead)."""
    monkeypatch.setattr(
        notices, "_graph_names", lambda project, select, target, strict=False: {("serde", "1.0.0")}
    )
    monkeypatch.setattr(notices, "_packages", lambda project, select=(): {})
    with pytest.raises(NoticesError, match=r"not all in cargo's cache\. Download them first \(--fetch"):
        notices.program_notices(PROJECT, {"nitro_bot": ["-p", "nitro_bot"]}, "2.2.1")
    assert notices._components(PROJECT)[0].licence == "see the crate's own licence", (
        "lenient for a bot folder"
    )
    ran: list[list[str]] = []

    class Done:
        returncode, stderr = 0, ""

    monkeypatch.setattr(notices.subprocess, "run", lambda command, **kw: ran.append(command) or Done())
    cargo = notices._toolchain().find_cargo(PROJECT)
    if cargo is None:
        pytest.skip("no Rust here")
    notices.fetch(
        PROJECT, {"nitro_bot": ["-p", "nitro_bot"], "replicar": ["--manifest-path", "r/Cargo.toml"]}
    )
    assert ran == [
        [str(cargo), "fetch", "--locked"],
        [str(cargo), "fetch", "--locked", "--manifest-path", "r/Cargo.toml"],
    ], "every platform's crates, for each program"


def test_the_tool_runs_without_nitrolearn_installed(tmp_path: Path) -> None:
    """The release builds the programs where NitroLearn is not installed: the tool is the file alone."""
    done = subprocess.run(
        [sys.executable, "-I", str(TOOL), "programs", "--out", str(tmp_path / "out.md"), "--replicar-version",
         "0.0.0"],
        capture_output=True, text=True, timeout=300, check=False, env={"CARGO_HOME": str(tmp_path)},
    )  # fmt: skip
    assert done.returncode == 1 and "replicar-cli 0.0.0 is not in cargo's cache" in done.stderr, done.stderr
    assert not (tmp_path / "out.md").exists()


@pytest.mark.skipif(
    shutil.which("cargo") is None, reason="no Rust here, so the programs' crates cannot be listed"
)
def test_each_programs_list_is_its_own_build_graph_for_its_target() -> None:
    bot = ["-p", notices.BOT_CRATE]
    _need_cached(bot)
    linux = {row.label.split()[0] for row in notices._components(PROJECT, bot, "x86_64-unknown-linux-gnu")}
    windows = {row.label.split()[0] for row in notices._components(PROJECT, bot, "x86_64-pc-windows-msvc")}
    assert "libc" in linux - windows and "windows-sys" in windows - linux
    assert {"rocketsim", "rlbot", "zstd-sys"} <= linux and not linux & {"nitro_core", "pyo3", "serde_derive"}
    converter = ["--manifest-path", str(PROJECT / "crates" / "nitro_replicar" / "Cargo.toml")]
    _need_cached(converter)
    text = notices.program_notices(
        PROJECT, {"nitro-replicar": converter}, "2.2.1", "x86_64-unknown-linux-gnu"
    )
    assert "- replicar 2.2.1: Copyright (c) 2026 Rolv Arild" in text, "from its repository's LICENSE"
    assert "mimalloc, the C library it builds: Copyright (c) 2018-2025 Microsoft Corporation" in text
    assert "- parquet 60.0.0: Apache-2.0" in text and "Apache Arrow\nCopyright 2016" in text
    assert "### Apache License, Version 2.0" in text and "the x86_64-unknown-linux-gnu target" in text


def test_the_release_ships_the_notices_with_the_programs() -> None:
    workflow = (PROJECT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for target in ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"):
        assert re.search(
            rf"python/nitro/notices\.py programs --fetch --out programs/{notices.PROGRAMS_FILE} [`\\]\n\s+"
            rf'--replicar-version "?\$(env:)?REPLICAR_VERSION"? --target {target}',
            workflow,
        ), target
    assert "path: programs/*.exe" not in workflow, "the Windows artifact keeps the notices too"


@pytest.mark.skipif(
    shutil.which("cargo") is None, reason="no Rust here, so the core's crates cannot be listed"
)
def test_the_cores_notices_list_its_own_build_graph_for_its_target(tmp_path: Path) -> None:
    core = ["--manifest-path", str(PROJECT / notices.CORE_MANIFEST)]
    _need_cached(core)
    linux = {row.label.split()[0] for row in notices._components(PROJECT, core, "x86_64-unknown-linux-gnu")}
    assert {"pyo3", "rocketsim", "zstd-sys"} <= linux and not linux & {
        "nitro_core",
        "nitro_py",
        "pyo3-macros",
    }
    done = subprocess.run(
        [sys.executable, "-I", str(TOOL), "core", "--out", str(tmp_path / "core.md"), "--target",
         "x86_64-pc-windows-msvc"],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    text = (tmp_path / "core.md").read_text(encoding="utf-8")
    assert "nitro._core" in text and next(iter(notices.required_notices(PROJECT))) in text
    assert "## nitro._core\n\n" in text and "- pyo3 " in text and "- rocketsim 0.3.3: MIT" in text
    assert "the x86_64-pc-windows-msvc target" in text and "### MIT License" in text


def test_every_wheel_carries_the_cores_notices() -> None:
    """The release's wheel jobs write them for their target before maturin builds, and maturin puts them in
    the wheel (they are git-ignored, so only `include` takes them); a build from the code writes them too
    (tools/install_steps.py)."""
    pyproject = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'include = [{{ path = "python/nitro/{notices.PROGRAMS_FILE}", format = "wheel" }}]' in pyproject
    workflow = (PROJECT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    core_job = workflow[workflow.index("  core:") : workflow.index("  programs-windows:")]
    assert (
        "python python/nitro/notices.py core --fetch --out python/nitro/THIRD_PARTY_NOTICES.md "
        "--target ${{ matrix.target }}" in core_job
    )
    assert core_job.index("notices.py core") < core_job.index("maturin-action")
    for target in ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"):
        assert f"target: {target}" in core_job
    assert f"python/nitro/{notices.PROGRAMS_FILE}" in (PROJECT / ".gitignore").read_text(encoding="utf-8")
