"""The release's own copies of what an install downloads (tools/bundles.py): the pins agree everywhere, only
files whose licence allows it are carried, every file stays under GitHub's size limit, and the installer
takes them from a folder of the release's files without downloading anything else. Stand-ins only: no
network, no cargo, no uv."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any

import pytest

from nitro import recordings as rec

PROJECT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


bundles = _load("bundles_under_test", PROJECT / "tools" / "bundles.py")
steps = _load("install_steps_for_bundles_test", PROJECT / "tools" / "install_steps.py")


# ---------- the pins ----------


def test_uv_and_python_are_pinned_the_same_in_both_installers() -> None:
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    sh = (PROJECT / "install.sh").read_text(encoding="utf-8")
    win, linux = bundles.UV["windows-x64"], bundles.UV["linux-x64"]
    assert f"$UvVersion = '{bundles.UV_VERSION}'" in ps1 and f"UV_VERSION={bundles.UV_VERSION}" in sh
    assert f"$UvWheel = '{win.asset}'" in ps1 and f"UV_WHEEL={linux.asset}" in sh
    assert f"$UvSha256 = '{win.sha256}'" in ps1 and f"UV_SHA256={linux.sha256}" in sh
    assert win.url.endswith("/" + win.asset) and win.url.replace(win.asset, "$UvWheel") in ps1
    assert linux.url.endswith("/" + linux.asset) and linux.url.replace(linux.asset, "$UV_WHEEL") in sh
    py_win, py_linux = bundles.PYTHON["windows-x64"], bundles.PYTHON["linux-x64"]
    assert f"$PythonAsset = '{py_win.asset}'" in ps1 and f"PYTHON_ASSET={py_linux.asset}" in sh
    assert f"$PythonSha256 = '{py_win.sha256}'" in ps1 and f"PYTHON_SHA256={py_linux.sha256}" in sh
    assert f"$PythonMirrorPath = '{bundles.mirror_path(py_win)}'" in ps1
    assert f"PYTHON_MIRROR_PATH={bundles.mirror_path(py_linux)}" in sh
    # uv looks for a build at <release>/<file name> inside its mirror folder, with the "+" unescaped.
    assert bundles.mirror_path(py_linux) == (
        "20260718/cpython-3.11.15+20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    )
    for item in (win, linux, py_win, py_linux, bundles.ROCKETSIM_LICENSE, bundles.REPLICAR_CLI):
        assert re.fullmatch(r"[0-9a-f]{64}", item.sha256) and item.url.startswith("https://")
    assert bundles.RELEASES == steps.RELEASES and bundles.SUMS == steps.SUMS
    assert bundles.RECORDINGS == rec.RELEASE_ASSET and bundles.RELEASES == rec.RELEASES
    assert steps.RELEASE_FILES_ENV == rec.RELEASE_FILES_ENV


def test_the_release_lists_every_copy_and_says_what_it_does_not_carry() -> None:
    names = bundles.bundled_assets("1.2.0")
    assert names == [
        "uv-0.11.32-py3-none-win_amd64.whl",
        "uv-0.11.32-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "python-3.11.15-windows-x64.tar.gz",
        "python-3.11.15-linux-x64.tar.gz",
        "rocketsim-recordings.zip",
        "nitrolearn-crates-1.2.0.tar.gz",
        "nitrolearn-wheels-1.2.0-windows-x64.zip",
        "nitrolearn-wheels-1.2.0-linux-x64.zip",
        "BUNDLED-NOTICES.md",
    ]
    text = bundles.notices("1.2.0")
    for name in names[:-1]:
        assert name in text
    assert "PyTorch" in text and "arena meshes" in text and "RLBot" in text
    for licence in ("uv-MIT.txt", "Python-PSF.txt", "python-build-standalone.md", "RocketSim-MIT.txt"):
        assert licence in text and (PROJECT / "LICENSES" / licence).is_file()
    listed = (PROJECT / "LICENSES" / "BUNDLED.md").read_text(encoding="utf-8")
    for part in (
        "rocketsim-recordings.zip",
        "nitrolearn-crates-",
        "nitrolearn-wheels-",
        "NitroLearn-Setup.exe",
    ):
        assert part in listed
    assert "Inno-Setup.txt" in listed and (PROJECT / "LICENSES" / "Inno-Setup.txt").is_file()
    assert "2 GiB" in listed and "PyTorch" in listed and "game data" in listed


def test_every_file_must_be_there_and_under_githubs_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in bundles.bundled_assets("1.2.0"):
        (tmp_path / name).write_bytes(b"x" * 10)
    sizes = bundles.check(tmp_path, "1.2.0")
    assert dict(sizes)["BUNDLED-NOTICES.md"] == 10
    monkeypatch.setattr(bundles, "LIMIT", 10)
    with pytest.raises(bundles.BundleError, match="2 GiB"):
        bundles.check(tmp_path, "1.2.0")
    (tmp_path / "rocketsim-recordings.zip").unlink()
    with pytest.raises(bundles.BundleError, match=r"missing from the release: rocketsim-recordings\.zip"):
        bundles.check(tmp_path, "1.2.0")


# ---------- licences ----------


@pytest.mark.parametrize(
    ("expression", "allowed"),
    [
        ("MIT", True),
        ("MIT OR Apache-2.0", True),
        ("MIT/Apache-2.0", True),
        ("(MIT OR Apache-2.0) AND Unicode-3.0", True),
        ("Apache-2.0 WITH LLVM-exception", True),
        ("MIT OR Apache-2.0 OR LGPL-2.1-or-later", True),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", True),
        ("MIT License", True),
        ("3-Clause BSD License", True),
        ("Mozilla Public License 2.0 (MPL 2.0)", True),
        ("GPL-3.0-only", False),
        ("MIT AND GPL-3.0-only", False),
        ("LGPL-3.0-only", False),
        ("", False),
        ("Proprietary", False),
    ],
)
def test_only_licences_that_allow_passing_files_on_are_carried(expression: str, allowed: bool) -> None:
    assert bundles.licence_allows(expression) is allowed


def _wheel(folder: Path, name: str, version: str, metadata: str, licence_file: str | None = None) -> Path:
    path = folder / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr(
            f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.4\nName: {name}\n{metadata}"
        )
        if licence_file is not None:
            out.writestr(f"{name}-{version}.dist-info/licenses/LICENSE", licence_file)
    return path


def test_a_wheels_licence_is_read_from_its_metadata_or_its_licence_file(tmp_path: Path) -> None:
    expression = _wheel(tmp_path, "a", "1", "License-Expression: MIT OR Apache-2.0\n")
    classifier = _wheel(tmp_path, "b", "1", "License: MIT License\n  Copyright (c) someone\n"
                        "Classifier: License :: OSI Approved :: MIT License\n")  # fmt: skip
    text_only = _wheel(tmp_path, "c", "1", "", "Permission is hereby granted, free of charge, to any person")
    gpl = _wheel(tmp_path, "d", "1", "License-Expression: GPL-3.0-only\n")
    assert bundles.wheel_licence(expression) == "MIT OR Apache-2.0"
    assert bundles.wheel_licence(classifier) == "MIT License"
    assert bundles.wheel_licence(text_only) == "MIT"
    listing = bundles.wheels_list([expression, classifier, text_only])
    assert "| a | 1 | MIT OR Apache-2.0 |" in listing and "| c | 1 | MIT |" in listing
    with pytest.raises(bundles.BundleError, match=r"d 1 \(GPL-3\.0-only\)"):
        bundles.wheels_list([expression, gpl])


# ---------- the Python packages ----------


def test_the_wheels_are_what_an_install_takes_from_pypi_except_pytorch() -> None:
    wanted = bundles.wheel_requirements()
    names = {re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0] for line in wanted}
    assert "torch" not in names
    assert {"numpy", "wandb", "pyarrow", "pywebview", "tomlkit", "maturin"} <= names
    assert any(line.startswith("maturin") and "<2" in line for line in wanted), "the build backend's range"
    compiled = (
        "numpy==2.4.6 \\\n    --hash=sha256:" + "a" * 64 + " \\\n    --hash=sha256:" + "b" * 64 + "\n"
        "    # via -r wanted.in\nproxy-tools==0.1.0 \\\n    --hash=sha256:" + "c" * 64 + "\n"
    )
    assert bundles.pinned(compiled) == [
        ("numpy", "2.4.6", ["a" * 64, "b" * 64]),
        ("proxy-tools", "0.1.0", ["c" * 64]),
    ]
    assert bundles.UV_PLATFORM == {
        "windows-x64": "x86_64-pc-windows-msvc",
        "linux-x64": "x86_64-manylinux_2_28",
    }
    assert bundles.PIP_PLATFORMS["linux-x64"][0] == "manylinux_2_28_x86_64"


# ---------- the Rust crates ----------


def _crate(folder: Path, name: str, version: str, licence: str) -> Path:
    """A .crate file as crates.io serves it."""
    path = folder / f"{name}-{version}.crate"
    with tarfile.open(path, "w:gz") as tar:
        for inside, text in (
            ("Cargo.toml", f'[package]\nname = "{name}"\nversion = "{version}"\nlicense = "{licence}"\n'),
            ("src/main.rs", "fn main() {}\n"),
        ):
            data = text.encode()
            info = tarfile.TarInfo(f"{name}-{version}/{inside}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_a_crate_is_vendored_as_cargo_lays_it_out_and_only_permissive_ones_are_listed(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    crate = _crate(tmp_path, "replicar-cli", "2.2.1", "MIT")
    sha = hashlib.sha256(crate.read_bytes()).hexdigest()
    with pytest.raises(bundles.BundleError, match="pinned SHA-256"):
        bundles.add_crate(vendor, crate, "0" * 64)
    placed = bundles.add_crate(vendor, crate, sha)
    assert placed == vendor / "replicar-cli-2.2.1"
    checksum = json.loads((placed / ".cargo-checksum.json").read_text(encoding="utf-8"))
    assert checksum["package"] == sha
    assert checksum["files"]["src/main.rs"] == hashlib.sha256(b"fn main() {}\n").hexdigest()
    rows = bundles.vendored_crates(vendor)
    assert rows == [("replicar-cli", "2.2.1", "MIT", "")]
    assert "| replicar-cli | 2.2.1 | MIT |" in bundles.crates_list(rows)
    with pytest.raises(bundles.BundleError, match=r"gpl-only 1\.0\.0"):
        bundles.crates_list([*rows, ("gpl-only", "1.0.0", "GPL-3.0-only", "")])
    # The archive is the same bytes whenever it is made.
    first = bundles.tar_folder(tmp_path / "a.tar.gz", vendor, [("LIST.md", b"list")]).read_bytes()
    second = bundles.tar_folder(tmp_path / "b.tar.gz", vendor, [("LIST.md", b"list")]).read_bytes()
    assert first == second
    with tarfile.open(tmp_path / "a.tar.gz") as tar:
        assert {"LIST.md", "vendor/replicar-cli-2.2.1/src/main.rs"} <= set(tar.getnames())
    config = bundles.CARGO_CONFIG.format(vendor=bundles.VENDOR)
    assert 'replace-with = "nitrolearn-vendored"' in config and 'directory = "vendor"' in config


def test_the_recordings_archive_holds_the_five_and_rocketsims_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {f"rec_{i}.rlpr.zst": f"recording {i}".encode() for i in range(5)}
    stand_in = type(
        "Recordings",
        (),
        {
            "RECORDINGS": {n: (len(d), hashlib.sha256(d).hexdigest()) for n, d in files.items()},
            "URL": "https://example.invalid/rec/",
        },
    )
    monkeypatch.setattr(bundles, "recordings_module", lambda root=PROJECT: stand_in)
    fetched: list[str] = []

    def download(url: str, target: Path, sha256: str, attempts: int = 4) -> Path:
        fetched.append(url)
        name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
        target.write_bytes(files.get(name, b"MIT License"))
        return target

    monkeypatch.setattr(bundles, "download", download)
    archive = bundles.recordings_archive(tmp_path / bundles.RECORDINGS)
    with zipfile.ZipFile(archive) as zipped:
        assert sorted(zipped.namelist()) == ["LICENSE", *sorted(files)]
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zipped.infolist())
    assert fetched[0] == bundles.ROCKETSIM_LICENSE.url


# ---------- the installer, from a folder of the release's files ----------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_folder(folder: Path, version: str) -> Path:
    """A folder of the release's files for this PC, as the release workflow makes them (stand-ins)."""
    platform = steps.platform_tag()
    folder.mkdir(parents=True)
    wheel = folder / f"nitrolearn-{version}-cp311-abi3-manylinux_2_28_x86_64.whl"
    wheel.write_bytes(b"core")
    programs = folder / f"nitrolearn-programs-{version}-{platform}.tar.gz"
    with tarfile.open(programs, "w:gz") as tar:
        for name, data in (("nitro_bot", b"bot"), ("prebuilt.json", b"{}")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(folder / bundles.wheels_asset(version, platform), "w") as out:
        out.writestr("numpy-2.4.6-cp311-cp311-manylinux_2_28_x86_64.whl", b"numpy")
        out.writestr("requirements.txt", "numpy==2.4.6 --hash=sha256:" + "a" * 64 + "\n")
        out.writestr("NOTICES.md", "list\n")
        out.writestr("../outside.whl", b"never")
    vendor = folder / "build" / "vendor"
    (vendor / "glam-0.34.1").mkdir(parents=True)
    (vendor / "glam-0.34.1" / "Cargo.toml").write_text('[package]\nname = "glam"\n', encoding="utf-8")
    bundles.tar_folder(folder / bundles.crates_asset(version), vendor, [("VENDORED-CRATES.md", b"list")])
    sums = [f"{_sha(p)}  {p.name}" for p in sorted(folder.iterdir()) if p.is_file()]
    (folder / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return folder


class Recorder:
    """Stands in for running programs: records each command and plays the part of uv."""

    def __init__(self, site: Path) -> None:
        self.commands: list[list[str]] = []
        self.site = site
        self.torch: str | None = None

    def __call__(self, command: Any, check: bool = True, minutes: float | None = None, what: str = "") -> int:
        command = [str(part) for part in command]
        self.commands.append(command)
        for part in command:
            if part.startswith("torch=="):
                self.torch = part.removeprefix("torch==")
            if part.startswith("nitrolearn["):
                (self.site / "nitro").mkdir(parents=True, exist_ok=True)
                (self.site / "nitro" / "_core.abi3.so").write_bytes(b"core")
        return 0

    def output(self, command: Any, minutes: float | None = None, env: Any = None, heartbeat: str = "") -> Any:
        """Stands in for run_output (the PyTorch install)."""
        return steps.Ran(self(command, minutes=minutes))


@pytest.mark.skipif(steps.platform_tag() != "linux-x64", reason="the stand-in release is for Linux x86-64")
def test_with_a_folder_of_the_release_files_nothing_else_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "NitroLearn"
    (root / "python" / "nitro").mkdir(parents=True)
    (root / "Cargo.toml").write_text('[workspace.package]\nversion = "1.2.0"\n', encoding="utf-8")
    _release_folder(root / "release-files", "1.2.0")
    site = tmp_path / "site-packages"
    recorder = Recorder(site)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("CARGO_HOME", str(root / ".tools" / "cargo"))
    monkeypatch.setattr(steps, "run", recorder)
    monkeypatch.setattr(steps, "run_output", recorder.output)
    monkeypatch.setattr(steps, "site_packages", lambda: site)
    monkeypatch.setattr(steps, "installed_version", lambda name: recorder.torch)
    monkeypatch.setattr(steps, "core_matches_code", lambda: True)
    monkeypatch.setattr(steps, "fetch", lambda *a, **k: pytest.fail(f"nothing is downloaded: {a}"))
    monkeypatch.setattr(steps, "_ensure_rust", lambda root: None)
    monkeypatch.setattr(steps, "compiler_problem", lambda: None)
    args = steps.parse(["--uv", "uv", "--cpu", "--rust", "--progress"])
    steps.setup(args, root, "file:///nowhere/v{version}/", lambda: None)

    assert os.environ[steps.RELEASE_FILES_ENV] == str(root / "release-files"), "the recordings look there too"
    installs = [c for c in recorder.commands if c[1:3] == ["pip", "install"]]
    wheels = str(root / ".tmp" / "wheels")
    assert installs[1][-6:] == ["--no-index", "--find-links", wheels, "--require-hashes", "-r",
                                str(root / ".tmp" / "wheels" / "requirements.txt")]  # fmt: skip
    assert installs[2][5:8] == ["--no-index", "--find-links", wheels]
    assert installs[2][-1].startswith("nitrolearn[wandb,replays,dashboard] @ file:")
    assert not any("maturin>=1.15,<2" in c for c in recorder.commands), "maturin is in the release's copy"
    assert (root / ".tools" / "prebuilt" / "nitro_bot").read_bytes() == b"bot"
    # Rust's crates from the release's copy, and cargo told to use only them.
    assert (root / ".tools" / "vendor" / "glam-0.34.1" / "Cargo.toml").is_file()
    config = (root / ".tools" / "cargo" / "config.toml").read_text(encoding="utf-8")
    assert config.startswith("# Written by NitroLearn's installer") and 'directory = "vendor"' in config
    assert not (root / ".tmp" / "wheels").exists() and not (root / "outside.whl").exists()


def test_the_crates_copy_is_kept_until_it_changes_and_removed_without_one(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    folder = _release_folder(tmp_path / "release-files", "1.2.0") if steps.platform_tag() else None
    if folder is None:
        pytest.skip("no release files for this kind of computer")
    release = steps.Release("1.2.0", "file:///nowhere/", folder, steps.sums_table(
        (folder / "SHA256SUMS.txt").read_text(encoding="utf-8")))  # fmt: skip
    assert steps.use_vendored_crates(root, release)
    stamp = root / ".tools" / "vendor" / steps.VENDOR_STAMP
    marker = root / ".tools" / "vendor" / "mine.txt"
    marker.write_text("kept", encoding="utf-8")
    assert steps.use_vendored_crates(root, release) and marker.is_file(), "unchanged: not unpacked again"
    assert stamp.read_text(encoding="utf-8").strip() == release.hashes[bundles.crates_asset("1.2.0")]
    # A release without the copy: cargo goes back to crates.io.
    assert not steps.use_vendored_crates(root, steps.Release("1.2.0", "u", None, {}))
    assert not (root / ".tools" / "cargo" / "config.toml").exists() and not stamp.exists()
    # Settings of the user's own in cargo's config are never replaced.
    config = root / ".tools" / "cargo" / "config.toml"
    config.write_text("[net]\noffline = true\n", encoding="utf-8")
    assert not steps.use_vendored_crates(root, release)
    assert config.read_text(encoding="utf-8") == "[net]\noffline = true\n"


def test_the_release_files_beside_the_setup_are_used_and_another_versions_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release's files downloaded together and unzipped: NitroLearn-Setup.exe sits among them and passes
    their folder on (installer/NitroLearn.iss, -Bundles). They are used before anything is downloaded; a
    folder of another version's files is not."""
    root = tmp_path / "NitroLearn"
    root.mkdir()
    (root / "Cargo.toml").write_text('[workspace.package]\nversion = "1.2.0"\n', encoding="utf-8")
    beside = tmp_path / "Downloads" / "release-files"
    beside.mkdir(parents=True)
    names = ["nitrolearn-1.2.0-cp311-abi3-win_amd64.whl", "nitrolearn-programs-1.2.0-windows-x64.zip",
             bundles.crates_asset("1.2.0"), "NitroLearn-Setup.exe"]  # fmt: skip
    for name in names:
        (beside / name).write_bytes(name.encode())
    (beside / steps.SUMS).write_text("".join(f"{_sha(beside / n)}  {n}\n" for n in names), encoding="utf-8")
    monkeypatch.setattr(steps, "fetch", lambda *a, **k: pytest.fail(f"nothing is downloaded: {a}"))
    release = steps.read_release(root, "file:///nowhere/v{version}/", beside)
    assert release is not None and release.folder == beside and set(release.hashes) == set(names)
    found = steps.release_files("\n".join(f"h  {n}" for n in release.hashes), "1.2.0", "windows-x64")
    assert found == (names[0], names[1]), "the names the release workflow gives them"
    assert release.get(names[1], tmp_path / "unused") == beside / names[1]

    asked: list[str] = []

    def offline(url: str, attempts: int = 4) -> bytes:
        asked.append(url)
        raise OSError("offline")

    monkeypatch.setattr(steps, "fetch", offline)
    (root / "Cargo.toml").write_text('[workspace.package]\nversion = "1.3.0"\n', encoding="utf-8")
    assert steps.read_release(root, "file:///nowhere/v{version}/", beside) is None
    assert asked == ["file:///nowhere/v1.3.0/SHA256SUMS.txt"], "another version's files: the release asked"
    assert steps.of_version({"nitrolearn-1.2.0-cp311-abi3-win_amd64.whl": "h"}, "1.2.0")
    assert not steps.of_version({"nitrolearn-1.2.01-cp311-abi3-win_amd64.whl": "h"}, "1.2.0")
    iss = (PROJECT / "installer" / "NitroLearn.iss").read_text(encoding="utf-8")
    assert "FileExists(AddBackslash(Src) + 'SHA256SUMS.txt')" in iss, "the files beside the setup first"
    assert "FileExists(AddBackslash(Src) + 'release-files\\SHA256SUMS.txt')" in iss
    assert "Folder := ReleaseFilesFolder;" in iss and "' -Bundles \"' + Folder + '\"'" in iss
    assert "Result := Result + '.'" in iss, "a drive's backslash would escape the closing quote"


def test_a_release_file_that_does_not_match_its_hash_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_bytes(b"changed")
    release = steps.Release("1.2.0", "file:///nowhere/", tmp_path, {"a.zip": "0" * 64})
    with pytest.raises(steps.Stop, match="does not match the release's SHA-256 hash"):
        release.get("a.zip", tmp_path / "out" / "a.zip")
    assert steps.release_folder(tmp_path, None) is None
    (tmp_path / "release-files").mkdir()
    assert steps.release_folder(tmp_path, None) == tmp_path / "release-files"
    assert steps.release_folder(tmp_path, Path("elsewhere")) == Path("elsewhere").resolve()


def test_the_progress_lines_name_each_step_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "NitroLearn"
    (root / "python" / "nitro").mkdir(parents=True)
    (root / "Cargo.toml").write_text('[workspace.package]\nversion = "1.2.0"\n', encoding="utf-8")
    recorder = Recorder(tmp_path / "site")
    monkeypatch.setattr(steps, "run", recorder)
    monkeypatch.setattr(steps, "run_output", recorder.output)
    monkeypatch.setattr(steps, "installed_version", lambda name: recorder.torch)
    monkeypatch.setattr(steps, "install_from_code", lambda *a: None)
    for progress in (False, True):
        argv = ["--uv", "uv", "--cpu"] + (["--progress"] if progress else [])
        steps.setup(steps.parse(argv), root, "file:///nowhere/v{version}/", lambda: None)
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("@@nitro ")]
        if not progress:
            assert lines == []
    assert lines == [
        f"@@nitro {key} {state}"
        for key in ("pc", "torch", "core", "meshes", "check")
        for state in ("start", "done")
    ]


# ---------- the recordings, from the release's copy ----------


def test_the_recordings_come_from_a_folder_of_the_release_files_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {f"rec_{i}.rlpr.zst": f"recording {i}".encode() for i in range(5)}
    monkeypatch.setattr(
        rec, "RECORDINGS", {n: (len(d), hashlib.sha256(d).hexdigest()) for n, d in files.items()}
    )
    release = tmp_path / "release-files"
    release.mkdir()
    with zipfile.ZipFile(release / rec.RELEASE_ASSET, "w") as out:
        out.writestr("LICENSE", "MIT License")
        for name, data in files.items():
            out.writestr(name, data if name != "rec_4.rlpr.zst" else b"changed")
    monkeypatch.setenv(rec.RELEASE_FILES_ENV, str(release))
    folder = tmp_path / "rocketsim_recordings"
    fetched: list[str] = []

    def github(url: str, target: Path) -> str:
        fetched.append(url)
        data = files[url.removeprefix(rec.URL)]
        target.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    paths = rec.fetch_recordings(folder, say=lambda text: None, download=github)
    assert [p.read_bytes() for p in paths] == list(files.values())
    assert fetched == [rec.URL + "rec_4.rlpr.zst"], "only the one the copy had wrong comes from RocketSim"
    assert sorted(p.name for p in folder.iterdir()) == sorted(files)
