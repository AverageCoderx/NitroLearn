"""The installer: everything stays inside NitroLearn's folder, PyTorch fits the card and comes first, the
release's ready-made files are used only when they check out. install.ps1 itself only runs on Windows; its
steps after Python and .venv are tools/install_steps.py, tested here, and install.sh runs here with stand-ins
for uv and those steps (and once for real against PyPI when it can be reached)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from nitro import folder_env

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_steps", PROJECT / "tools" / "install_steps.py")
assert spec is not None and spec.loader is not None
steps = importlib.util.module_from_spec(spec)
sys.modules["install_steps"] = steps  # its dataclasses look their module up
spec.loader.exec_module(steps)

SMI_HEADER = "| NVIDIA-SMI 572.16   Driver Version: 572.16   CUDA Version: {cuda}   |\n"


@pytest.fixture(autouse=True)
def _room_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Room enough for any install, whatever this computer's disk holds (the space tests set their own)."""
    monkeypatch.setattr(steps, "free_gb", lambda folder: 500.0)


def _settings_in(path: Path, pattern: str) -> dict[str, str]:
    """name -> value, as written after the folder's own part, from one settings file."""
    found = {}
    for name, value in re.findall(pattern, path.read_text(encoding="utf-8"), re.MULTILINE):
        found[name] = value
    return found


def _normalised(values: dict[str, str]) -> dict[str, str]:
    return {name: value.replace("\\", "/").removeprefix("/") for name, value in values.items()}


def test_every_launcher_keeps_the_same_folders_inside_nitrolearn() -> None:
    """tools/env.ps1, nitro.cmd and tools/env.sh set what install_steps.folder_env sets, each pointing
    inside the folder; uv never adds Python to the PATH or the registry."""
    ps1 = _settings_in(PROJECT / "tools" / "env.ps1", r"^\$env:(\w+) = (?:Join-Path \$NitroRoot )?'([^']*)'")
    cmd = _settings_in(PROJECT / "nitro.cmd", r'^set "(\w+)=(?:%NITRO_ROOT%)?([^"]*)"')
    sh = _settings_in(PROJECT / "tools" / "env.sh", r'^export (\w+)="?(?:\$NITRO_ROOT)?([^"\n]*)"?$')
    cmd.pop("NITRO_ROOT")
    cmd.pop("PATH")
    sh.pop("PATH")
    root = Path("/nitro")
    for windows, found in ((True, ps1), (True, cmd), (False, sh)):
        wanted = {
            name: value.removeprefix(str(root)).replace(os.sep, "/").removeprefix("/")
            for name, value in steps.folder_env(root, windows).items()
        }
        assert _normalised(found) == wanted
    expected = steps.folder_env(root, True)
    assert expected["UV_PYTHON_INSTALL_BIN"] == "0" and expected["UV_PYTHON_INSTALL_REGISTRY"] == "0"
    limits = folder_env.CARGO_NETWORK
    for name, value in expected.items():
        assert name in limits or value in ("0", "1") or Path(value).is_relative_to(root), name
    for text in (PROJECT / "nitro.cmd", PROJECT / "tools" / "env.ps1", PROJECT / "tools" / "env.sh"):
        assert ".tools" in text.read_text(encoding="utf-8") and "cargo" in text.read_text(encoding="utf-8")


def test_the_double_click_installer_runs_install_ps1_with_its_options() -> None:
    text = (PROJECT / "Install NitroLearn.cmd").read_text(encoding="utf-8")
    assert '-ExecutionPolicy Bypass -File "%~dp0install.ps1" %*' in text and "pause" in text
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    sh = (PROJECT / "install.sh").read_text(encoding="utf-8")
    # The same uv on both, from PyPI with a pinned hash, never Astral's install script.
    assert "astral.sh" not in ps1 + sh
    assert re.search(r"\$UvVersion = '0\.11\.32'", ps1) and "UV_VERSION=0.11.32" in sh
    assert re.search(r"\$UvSha256 = '[0-9a-f]{64}'", ps1) and re.search(r"UV_SHA256=[0-9a-f]{64}", sh)
    for option in ("--torch", "--rust", "--from-source", "--dev"):
        assert option in ps1 and option in sh


def _facts(cuda: str | None, *cards: tuple[str, str]) -> Any:
    table = SMI_HEADER.format(cuda=cuda) if cuda else "| NVIDIA-SMI |\n"
    return steps.parse_nvidia(table, "\n".join(f"{name}, {cap}" for name, cap in cards))


def test_the_pytorch_build_fits_the_card_and_driver() -> None:
    assert steps.choose_torch(None)[0] == "cpu"
    assert steps.choose_torch(_facts("12.9", ("NVIDIA GeForce RTX 3070", "8.6"))) == ("cu126", [])
    assert steps.choose_torch(_facts("13.0", ("NVIDIA GeForce RTX 5090", "12.0"))) == ("cu128", [])
    assert steps.choose_torch(_facts("12.8", ("NVIDIA GeForce GTX 1080", "6.1")))[0] == "cu126"
    assert steps.choose_torch(_facts("12.4", ("Tesla K40", "3.5")))[0] == "cpu"
    # An old driver: the build the card needs, and a note to update the driver (no reinstall later).
    variant, notes = steps.choose_torch(_facts("12.6", ("NVIDIA GeForce RTX 5080", "12.0")))
    assert variant == "cu128" and "Update the driver" in notes[0]
    old = steps.choose_torch(_facts("12.2", ("NVIDIA GeForce RTX 2060", "7.5")))
    assert old[0] == "cu126" and "12.2" in old[1][0]
    # A driver too old to name the compute capability: an RTX 50 card is known by its name.
    assert steps.choose_torch(_facts("12.8", ("NVIDIA GeForce RTX 5070 Ti", "[N/A]")))[0] == "cu128"
    facts = steps.parse_nvidia(SMI_HEADER.format(cuda="13.0").replace("CUDA", "CUDA UMD"), "A, 8.9\nB, 12.0")
    assert facts.driver_cuda == (13, 0) and [card.capability for card in facts.cards] == [(8, 9), (12, 0)]


def test_a_chosen_pytorch_build_is_remembered(tmp_path: Path) -> None:
    assert steps.torch_choice(tmp_path, "auto") is None
    assert steps.torch_choice(tmp_path, "cu130") == "cu130"
    assert steps.torch_choice(tmp_path, "auto") == "cu130"


SUMS = """\
{wheel_win}  nitrolearn-1.2.0-cp311-abi3-win_amd64.whl
{wheel_linux}  nitrolearn-1.2.0-cp311-abi3-manylinux_2_28_x86_64.whl
{zip}  nitrolearn-programs-1.2.0-windows-x64.zip
{tar}  nitrolearn-programs-1.2.0-linux-x64.tar.gz
"""


def test_the_release_files_for_each_platform() -> None:
    sums = SUMS.format(wheel_win="a" * 64, wheel_linux="b" * 64, zip="c" * 64, tar="d" * 64)
    assert steps.release_files(sums, "1.2.0", "windows-x64") == (
        "nitrolearn-1.2.0-cp311-abi3-win_amd64.whl",
        "nitrolearn-programs-1.2.0-windows-x64.zip",
    )
    assert steps.release_files(sums, "1.2.0", "linux-x64") == (
        "nitrolearn-1.2.0-cp311-abi3-manylinux_2_28_x86_64.whl",
        "nitrolearn-programs-1.2.0-linux-x64.tar.gz",
    )
    assert steps.release_files(sums, "1.3.0", "linux-x64") is None
    assert steps.sums_table(sums)["nitrolearn-programs-1.2.0-linux-x64.tar.gz"] == "d" * 64
    assert (
        steps.project_version(PROJECT)
        == re.search(
            r'^version = "(.+)"', (PROJECT / "Cargo.toml").read_text(encoding="utf-8"), re.MULTILINE
        )[1]
    )


def test_the_ready_made_core_goes_next_to_the_code(tmp_path: Path) -> None:
    """Installed as a plain wheel, NitroLearn would look for its folder (settings, runs, crates) inside
    .venv: its core moves to python/nitro and a .pth points Python there, as an editable install does."""
    site = tmp_path / "site-packages"
    (site / "nitro").mkdir(parents=True)
    (site / "nitro" / "__init__.py").write_text("", encoding="utf-8")
    (site / "nitro" / "_core.abi3.so").write_bytes(b"new core")
    root = tmp_path / "NitroLearn"
    (root / "python" / "nitro").mkdir(parents=True)
    (root / "python" / "nitro" / "_core.abi3.so").write_bytes(b"old core")
    (site / "nitro" / steps.CORE_NOTICES).write_text("# Third-party notices\n", encoding="utf-8")
    placed = steps.use_source_folder(site, root)
    assert placed.read_bytes() == b"new core" and placed.parent == root / "python" / "nitro"
    assert not (site / "nitro").exists()
    assert _pth_paths(site) == [str(root / "python")]
    notices = root / "python" / "nitro" / steps.CORE_NOTICES
    assert notices.read_text(encoding="utf-8") == "# Third-party notices\n", "the core's notices go with it"
    (site / "nitro").mkdir()
    (site / "nitro" / "_core.abi3.so").write_bytes(b"a core without notices")
    steps.use_source_folder(site, root)
    assert not notices.exists(), "no notices left that describe another core"


def _pth_paths(site: Path, encoding: str = "utf-8") -> list[str]:
    """What Python's own site.addpackage adds to sys.path from the .pth files in `site`, reading them in
    `encoding` as Python 3.11 does with the system's code page (cp1252 on most Western Windows PCs)."""
    import site as site_module

    script = (
        "import io, site, sys\n"
        "wrap = io.TextIOWrapper\n"
        "io.TextIOWrapper = lambda raw, encoding=None, **kw: wrap(raw, encoding=sys.argv[2] "
        "if encoding == 'locale' else encoding, **kw)\n"
        "before = list(sys.path)\n"
        "for name in sorted(n for n in __import__('os').listdir(sys.argv[1]) if n.endswith('.pth')):\n"
        "    site.addpackage(sys.argv[1], name, set())\n"
        "print(repr([p for p in sys.path if p not in before]))\n"
    )
    assert site_module.addpackage  # the function the script drives
    done = subprocess.run(
        [sys.executable, "-I", "-c", script, str(site), encoding],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "Error processing line" not in done.stderr, done.stderr
    return list(eval(done.stdout.strip()))  # our own script's repr of a list


@pytest.mark.parametrize("name", ["José", "Łukasz", "Müller Á"])
def test_a_folder_with_a_non_ascii_name_still_reaches_python(tmp_path: Path, name: str) -> None:
    """The default folder holds the Windows user name. Python 3.11 reads .pth files in the system's code page:
    the ready-made core's .pth and an editable install's are written in plain ASCII so the folder is found
    however Windows reads them (a UTF-8 path line read as cp1252 points nowhere, and with Á or Ł it stops
    Python from starting at all)."""
    root = tmp_path / name / "NitroLearn"
    site = root / ".venv" / "Lib" / "site-packages"
    (site / "nitro").mkdir(parents=True)
    (site / "nitro" / "_core.abi3.so").write_bytes(b"core")
    (root / "python" / "nitro").mkdir(parents=True)
    steps.use_source_folder(site, root)
    assert (site / steps.FOLDER_PTH).read_bytes().isascii()
    for encoding in ("cp1252", "cp1251", "utf-8"):
        assert _pth_paths(site, encoding) == [str(root / "python")]

    # An editable install's own .pth (a plain UTF-8 path) is rewritten the same way; other lines stay.
    (site / steps.FOLDER_PTH).unlink()
    (site / "nitrolearn.pth").write_text(str(root / "python"), encoding="utf-8")
    (site / "other.pth").write_text("import os\n", encoding="utf-8")
    assert steps.ascii_pth_files(site) == ["nitrolearn.pth"]
    assert (site / "nitrolearn.pth").read_bytes().isascii()
    assert (site / "other.pth").read_text(encoding="utf-8") == "import os\n"
    assert _pth_paths(site, "cp1252") == [str(root / "python")]
    assert steps.ascii_pth_files(site) == [], "ASCII files are left as they are"


def test_a_plain_utf8_path_line_is_what_breaks(tmp_path: Path) -> None:
    """The reason for the above: the same folder written as a plain line is lost when read as cp1252."""
    folder = tmp_path / "José" / "python"
    folder.mkdir(parents=True)
    site = tmp_path / "site"
    site.mkdir()
    (site / "plain.pth").write_text(str(folder) + "\n", encoding="utf-8")
    assert _pth_paths(site, "utf-8") == [str(folder)]
    assert _pth_paths(site, "cp1252") == []


def test_the_installer_builds_your_own_rust_rewards_in_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The release's core has none of the user's own Rust rewards: with Rust and the build tools they are
    built in again (`nitro rebuild`), else the installer says what is missing; a failed build does not stop
    the install."""
    built: list[str] = []
    monkeypatch.setattr(steps, "compiler_problem", lambda: None)
    assert steps.rebuild_own_rewards(True, lambda *args: built.append(args[0]) or 0)
    assert built == ["rebuild"]
    assert not steps.rebuild_own_rewards(True, lambda *args: 1)
    assert "Build on the dashboard's Resources page" in capsys.readouterr().out
    assert not steps.rebuild_own_rewards(False, lambda *args: built.append("again") or 0)
    assert "Tools to build your own Rust rewards" in capsys.readouterr().out and built == ["rebuild"]
    monkeypatch.setattr(steps, "compiler_problem", lambda: "Install the build tools.")
    assert not steps.rebuild_own_rewards(True, lambda *args: built.append("again") or 0)
    assert "Install the build tools." in capsys.readouterr().out and built == ["rebuild"]
    rebuild = (PROJECT / "python" / "nitro" / "rebuild.py").read_text(encoding="utf-8")
    assert f'LIST = Path("my-settings") / "{steps.OWN_REWARDS.name}"' in rebuild


def test_a_build_from_the_code_writes_the_cores_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(
        steps, "run", lambda command, check=True, minutes=None: ran.append([str(c) for c in command]) or 1
    )
    steps.write_core_notices(tmp_path)
    tool, out = (
        tmp_path / "python" / "nitro" / "notices.py",
        tmp_path / "python" / "nitro" / steps.CORE_NOTICES,
    )
    assert ran == [[sys.executable, str(tool), "core", "--out", str(out)]]
    assert f"python {tool} core --out {out}" in capsys.readouterr().out, "how to write them later"


def test_only_the_programs_are_taken_from_their_archive(tmp_path: Path) -> None:
    archive = tmp_path / "programs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        files = (("./nitro_bot", b"bot"), ("./prebuilt.json", b"{}"), ("./evil.sh", b"x"))
        for name, data in (*files, ("./THIRD_PARTY_NOTICES.md", b"# Third-party notices")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    folder = tmp_path / "prebuilt"
    (folder / "old").mkdir(parents=True)
    if sys.platform != "win32":
        taken = ["THIRD_PARTY_NOTICES.md", "nitro_bot", "prebuilt.json"]
        assert steps.unpack_programs(archive, folder) == taken, "the notices stay beside the programs"
        assert os.access(folder / "nitro_bot", os.X_OK)
    zipped = tmp_path / "programs.zip"
    with zipfile.ZipFile(zipped, "w") as out:
        out.writestr("prebuilt.json", "{}")
        out.writestr("../outside.exe", "x")
    assert steps.unpack_programs(zipped, folder) == ["prebuilt.json"]
    assert (
        sorted(p.name for p in folder.iterdir()) == ["prebuilt.json"]
        and not (tmp_path / "outside.exe").exists()
    )


class Recorder:
    """Stands in for running programs: records each command and plays the part of uv."""

    def __init__(self, site: Path) -> None:
        self.commands: list[list[str]] = []
        self.site = site
        self.torch: str | None = None
        self.limits: list[float | None] = []  # each command's time limit

    def __call__(self, command: Any, check: bool = True, minutes: float | None = None) -> int:
        command = [str(part) for part in command]
        self.commands.append(command)
        self.limits.append(minutes)
        if command[1:3] == ["pip", "install"]:
            for part in command:
                if part.startswith("torch=="):
                    self.torch = part.removeprefix("torch==")
                if part.startswith("nitrolearn["):
                    (self.site / "nitro").mkdir(parents=True, exist_ok=True)
                    (self.site / "nitro" / "_core.abi3.so").write_bytes(b"ready-made core")
        return 0


def _release(folder: Path, version: str, wheel: bytes = b"wheel", tamper: bool = False) -> str:
    """A release's files in `folder` (served as file:// URLs), as the release workflow makes them."""
    platform = steps.platform_tag()
    assert platform is not None
    wheel_name = (
        f"nitrolearn-{version}-cp311-abi3-win_amd64.whl"
        if platform == "windows-x64"
        else f"nitrolearn-{version}-cp311-abi3-manylinux_2_28_x86_64.whl"
    )
    folder.mkdir(parents=True)
    (folder / wheel_name).write_bytes(wheel)
    programs = f"nitrolearn-programs-{version}-{platform}"
    if platform == "windows-x64":
        archive = folder / f"{programs}.zip"
        with zipfile.ZipFile(archive, "w") as out:
            out.writestr("nitro_bot.exe", "bot")
            out.writestr("prebuilt.json", json.dumps({"programs": {"nitro_bot": "x"}}))
    else:
        archive = folder / f"{programs}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in (("nitro_bot", b"bot"), ("prebuilt.json", b'{"programs": {}}')):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    lines = []
    for file in (folder / wheel_name, archive):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        lines.append(f"{'0' * 64 if tamper and file == archive else digest}  {file.name}")
    (folder / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder.as_uri() + "/"


def _project(tmp_path: Path, version: str = "1.2.0") -> Path:
    root = tmp_path / "NitroLearn"
    (root / "python" / "nitro").mkdir(parents=True)
    (root / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{version}"\n', encoding="utf-8")
    return root


@pytest.mark.skipif(steps.platform_tag() is None, reason="no ready-made files for this kind of computer")
def test_setup_installs_pytorch_first_then_the_ready_made_nitrolearn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    base = _release(tmp_path / "release", "1.2.0")
    site = tmp_path / "site-packages"
    recorder = Recorder(site)
    monkeypatch.setattr(steps, "run", recorder)
    monkeypatch.setattr(steps, "site_packages", lambda: site)
    monkeypatch.setattr(steps, "installed_version", lambda name: recorder.torch)
    monkeypatch.setattr(steps, "core_matches_code", lambda: True)
    monkeypatch.setattr(steps, "install_from_code", lambda *a: pytest.fail("the ready-made files are there"))
    args = steps.parse(["--uv", "uv"])
    facts = _facts("12.8", ("NVIDIA GeForce RTX 5090", "12.0"))
    steps.setup(args, root, base, lambda: facts)

    installs = [c for c in recorder.commands if c[1:3] == ["pip", "install"]]
    assert installs[0][-3:] == [
        "torch==2.14.1+cu128",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ]
    assert installs[1][-1].startswith("nitrolearn[wandb,replays,dashboard] @ file:")
    assert (root / "python" / "nitro" / "_core.abi3.so").read_bytes() == b"ready-made core"
    assert (root / ".tools" / "prebuilt" / "prebuilt.json").is_file()
    tail = [c[1:] for c in recorder.commands[-2:]]
    assert tail == [["-m", "nitro", "fetch-meshes"], ["-m", "nitro", "doctor"]]
    limits = dict(zip((" ".join(c) for c in recorder.commands), recorder.limits, strict=True))
    for command, minutes in limits.items():
        downloads = "pip install" in command or "fetch-meshes" in command
        assert (minutes is not None) == downloads, f"a time limit on every download step only: {command}"
    assert not (root / ".tmp" / "release").exists(), "downloads are cleaned up"

    # Run again: PyTorch is not downloaded again.
    recorder.commands.clear()
    steps.setup(args, root, base, lambda: facts)
    assert not any("torch==2.14.1+cu128" in c for c in recorder.commands)


@pytest.mark.skipif(steps.platform_tag() is None, reason="no ready-made files for this kind of computer")
def test_setup_builds_from_the_code_without_ready_made_files_and_refuses_bad_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, "9.9.9")
    site = tmp_path / "site-packages"
    recorder = Recorder(site)
    monkeypatch.setattr(steps, "run", recorder)
    monkeypatch.setattr(steps, "site_packages", lambda: site)
    monkeypatch.setattr(steps, "installed_version", lambda name: recorder.torch)
    built: list[str] = []
    monkeypatch.setattr(steps, "install_from_code", lambda uv, folder, extras, *rest: built.append(extras))
    args = steps.parse(["--uv", "uv"])
    # No release for this version (no SHA256SUMS.txt): built from the code.
    base = (tmp_path / "nothing").as_uri() + "/v{version}/"
    steps.setup(args, root, base, lambda: None)
    assert built == ["wandb,replays,dashboard"]
    assert any(c[-1] == "https://download.pytorch.org/whl/cpu" for c in recorder.commands)
    # A file that does not match SHA256SUMS.txt stops the install; nothing of it is used.
    tampered = _release(tmp_path / "bad", "9.9.9", tamper=True)
    with pytest.raises(steps.Stop, match="did not match its SHA-256 hash"):
        steps.setup(args, root, tampered, lambda: None)
    assert not (root / ".tools" / "prebuilt").exists()


def test_building_from_the_code_says_what_to_install_when_the_compiler_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "compiler_problem", lambda: "install the build tools")
    monkeypatch.setattr(steps, "ensure_rust", lambda root: pytest.fail("nothing installed before that"))
    with pytest.raises(steps.Stop, match="install the build tools"):
        steps.install_from_code(Path("uv"), PROJECT, "dev")
    monkeypatch.undo()
    monkeypatch.setattr(steps, "WINDOWS", True)
    monkeypatch.setattr(steps, "msvc_found", lambda: False)
    text = steps.compiler_problem()
    assert text is not None and "Build Tools for Visual Studio 2022" in text and "never" in text


def test_a_developer_install_adds_rustfmt_and_clippy_to_the_folders_rust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    rustup = tmp_path / ".tools" / "cargo" / "bin" / ("rustup.exe" if steps.WINDOWS else "rustup")
    rustup.parent.mkdir(parents=True)
    rustup.write_text("", encoding="utf-8")
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / ".tools" / "cargo"))
    recorder = Recorder(tmp_path / "site-packages")
    monkeypatch.setattr(steps, "run", recorder)
    monkeypatch.setattr(steps, "installed_version", lambda name: recorder.torch)
    monkeypatch.setattr(steps, "install_from_code", lambda uv, folder, extras, *rest: None)
    args = steps.parse(["--uv", "uv", "--torch", "cpu", "--dev"])
    steps.setup(args, root, "file:///nowhere/", lambda: None)
    pin = steps.rust_pin()[0]
    assert [str(rustup), "component", "add", "--toolchain", pin, "rustfmt", "clippy"] in recorder.commands
    recorder.commands.clear()
    steps.setup(
        Namespace(**{**vars(args), "dev": False}), root, (tmp_path / "none").as_uri() + "/", lambda: None
    )
    assert not any("component" in command for command in recorder.commands), "only a developer install"


def test_rust_goes_into_the_folder_and_rustups_path_note_is_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CARGO_HOME", str(tmp_path / ".tools" / "cargo"))
    monkeypatch.setattr(steps, "fetch", lambda url: b"0" * 64 + b"  rustup-init\n")

    def fetched(url: str, sha256: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"installer")
        return target

    ran: list[tuple[list[str], float | None]] = []
    monkeypatch.setattr(steps, "fetch_checked", fetched)
    monkeypatch.setattr(
        steps,
        "run",
        lambda command, check=True, minutes=None: ran.append(([str(c) for c in command], minutes)),
    )
    steps.ensure_rust(tmp_path)
    ((command, minutes),) = ran
    assert "--no-modify-path" in command and minutes == steps.RUST_MINUTES
    pin, components = steps.rust_pin()
    assert command[command.index("--default-toolchain") + 1] == pin and "stable" not in command
    assert [command[i + 1] for i, part in enumerate(command) if part == "--component"] == components
    out = capsys.readouterr().out
    assert out.rstrip().endswith(steps.RUSTUP_PATH_NOTE) and "finds Rust in its own folder" in out


def test_rust_is_pinned_to_an_exact_version_everywhere_it_is_installed() -> None:
    """rust-toolchain.toml names one exact Rust (a build here is the build that was tested); the release's
    builds and the cloud setup install that one, never whatever 'stable' is that day."""
    with (PROJECT / "rust-toolchain.toml").open("rb") as file:
        toolchain = tomllib.load(file)["toolchain"]
    pin, components = steps.rust_pin()
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin) and toolchain["channel"] == pin
    assert components == ["rustfmt", "clippy"] and toolchain["profile"] == "minimal"
    workflow = (PROJECT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "stable" not in workflow
    assert f'rust-toolchain: "{pin}"' in workflow and f"rustup toolchain install {pin} " in workflow
    assert f"--default-toolchain {pin}" in workflow
    setup = (PROJECT / "tools" / "cloud_setup.sh").read_text(encoding="utf-8")
    assert "stable" not in setup and "rust-toolchain.toml" in setup


def _folder_rustup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: list[str]) -> list[list[str]]:
    """A folder whose rustup has `installed`; returns the commands ensure_rust runs, without running them."""
    cargo_home = tmp_path / ".tools" / "cargo"
    (cargo_home / "bin").mkdir(parents=True)
    (cargo_home / "bin" / ("rustup.exe" if steps.WINDOWS else "rustup")).write_text("", encoding="utf-8")
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))
    monkeypatch.setattr(steps, "rust_toolchains", lambda rustup: list(installed))
    monkeypatch.setattr(steps, "fetch", lambda url: pytest.fail("rustup is already in the folder"))
    ran: list[list[str]] = []
    monkeypatch.setattr(
        steps, "run", lambda command, check=True, minutes=None: ran.append([str(c) for c in command]) or 0
    )
    return ran


def test_an_earlier_install_gets_the_pinned_rust_and_loses_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder set up before the pin (or with a newer Rust whose new lints fail the checks) gets exactly the
    pinned version, as its default, and the Rust nothing builds with any more is removed."""
    pin, _ = steps.rust_pin()
    ran = _folder_rustup(tmp_path, monkeypatch, ["stable-x86_64-pc-windows-msvc"])
    steps.ensure_rust(tmp_path)
    rustup = ran[0][0]
    assert ran == [
        [
            rustup,
            "toolchain",
            "install",
            pin,
            "--profile",
            "minimal",
            "--no-self-update",
            "--component",
            "rustfmt",
            "--component",
            "clippy",
        ],
        [rustup, "default", pin],
        [rustup, "toolchain", "uninstall", "stable-x86_64-pc-windows-msvc"],
    ]


def test_the_pinned_rust_already_in_the_folder_is_kept_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pin, _ = steps.rust_pin()
    ran = _folder_rustup(tmp_path, monkeypatch, [f"{pin}-x86_64-unknown-linux-gnu"])
    steps.ensure_rust(tmp_path)
    assert [command[1:] for command in ran] == [["default", pin]]
    assert f"Rust {pin} is in this folder." in capsys.readouterr().out


def test_rustups_toolchain_list_is_read_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = "stable-x86_64-pc-windows-msvc\n1.97.0-x86_64-pc-windows-msvc (active, default)\n"
    done = subprocess.CompletedProcess([], 0, stdout=listing, stderr="")
    monkeypatch.setattr(steps.subprocess, "run", lambda *args, **kwargs: done)
    assert steps.rust_toolchains(tmp_path / "rustup") == [
        "stable-x86_64-pc-windows-msvc",
        "1.97.0-x86_64-pc-windows-msvc",
    ]
    none = subprocess.CompletedProcess([], 0, stdout="no installed toolchains\n", stderr="")
    monkeypatch.setattr(steps.subprocess, "run", lambda *args, **kwargs: none)
    assert steps.rust_toolchains(tmp_path / "rustup") == []


def test_a_stalled_cargo_download_fails_instead_of_hanging() -> None:
    """Every launcher and the installer give cargo network limits (the launchers' test checks they agree):
    a download under 1 KB/s for the timeout fails, is tried again a few times, and then the build step
    stops with the download hint, well inside the step's time limit."""
    for windows in (True, False):
        env = steps.folder_env(Path("/nitro"), windows)
        assert {name: env[name] for name in folder_env.CARGO_NETWORK} == {
            "CARGO_HTTP_TIMEOUT": "60",
            "CARGO_HTTP_LOW_SPEED_LIMIT": "1024",
            "CARGO_NET_RETRY": "3",
        }
    timeout = int(folder_env.CARGO_NETWORK["CARGO_HTTP_TIMEOUT"])
    tries = 1 + int(folder_env.CARGO_NETWORK["CARGO_NET_RETRY"])
    assert timeout * tries < steps.BUILD_MINUTES * 60 / 10


def _found(**changes: Any) -> Any:
    """A Windows PC with an NVIDIA card, a ready-made core, a person at the console, nothing installed."""
    fields: dict[str, Any] = {
        "windows": True,
        "person": True,
        "card": "NVIDIA GeForce RTX 3070",
        "no_card": "",
        "ready_made": True,
        "from_source": False,
        "first": True,
        "has": {},
    }
    return steps.Found(**(fields | changes))


def _parts(choices: dict[str, bool]) -> set[str]:
    return {part for part, on in choices.items() if on}


def test_the_first_install_starts_from_the_defaults() -> None:
    found = _found()
    assert _parts(steps.effective(found, steps.default_wishes(found, {}))) == {"gpu", "rlbot", "start_menu"}
    # No card: PyTorch for the processor, whatever is wished.
    plain = _found(card="", no_card="no NVIDIA card was found")
    assert not steps.effective(plain, steps.default_wishes(plain, {}) | {"gpu": True})["gpu"]
    # Without a person, nothing outside the folder (no shortcuts, no RLBot) unless a switch asks.
    alone = _found(person=False)
    assert _parts(steps.effective(alone, steps.default_wishes(alone, {}))) == {"gpu"}
    # Off Windows: no RLBot or shortcuts at all.
    linux = _found(windows=False)
    assert steps.shown_parts(linux) == ["gpu", "rust", "dev", "recordings"]
    assert _parts(steps.effective(linux, steps.default_wishes(linux, {}) | {"rlbot": True})) == {"gpu"}
    # Built from the code: Rust is required, not optional.
    for source in (_found(ready_made=False), _found(from_source=True)):
        assert steps.effective(source, steps.default_wishes(source, {}))["rust"]
        assert "needed" in steps.locked(source, {}, "rust")
    assert steps.locked(found, {"dev": True}, "rust"), "developer tools build from the code"
    assert steps.locked(found, {}, "rust") == ""


def test_a_run_again_starts_from_what_is_installed() -> None:
    has = {"torch": True, "gpu": False, "rlbot": True, "start_menu": True, "desktop": True, "rust": True}
    found = _found(first=False, has=has | {"recordings": True})
    wishes = steps.default_wishes(found, {"dev": True})
    assert _parts(steps.effective(found, wishes)) == {
        *("rlbot", "start_menu", "desktop", "rust", "dev", "recordings")
    }, "the CPU PyTorch chosen before stays; dev comes from the record"
    nothing = _found(first=False, has={"torch": True, "gpu": True})
    assert _parts(steps.effective(nothing, steps.default_wishes(nothing, {}))) == {"gpu"}, (
        "shortcuts and RLBot turned down before are not offered again by default"
    )


def test_switching_parts_on_and_off() -> None:
    found = _found()
    wishes = steps.default_wishes(found, {})
    wishes, why = steps.toggle(found, wishes, "desktop")
    assert why == "" and wishes["desktop"] and wishes["start_menu"]
    wishes, _ = steps.toggle(found, wishes, "start_menu")
    assert not wishes["start_menu"] and not wishes["desktop"], "the desktop one comes with the Start menu's"
    wishes, _ = steps.toggle(found, wishes, "dev")
    assert steps.effective(found, wishes)["rust"] and steps.toggle(found, wishes, "rust")[1]
    wishes, _ = steps.toggle(found, wishes, "dev")
    assert not steps.effective(found, wishes)["rust"], "Rust was required only by the developer tools"
    for part in ("gpu", "rust", "recordings"):
        once, _ = steps.toggle(found, wishes, part)
        assert steps.toggle(found, once, part)[0] == wishes
    plain = _found(card="", no_card="no NVIDIA card was found")
    assert steps.toggle(plain, {}, "gpu") == (
        {},
        "no NVIDIA card was found, so PyTorch for the processor is installed.",
    )
    installed = _found(has={"rlbot": True})
    kept, why = steps.toggle(installed, {"rlbot": True}, "rlbot")
    assert kept == {"rlbot": True} and "Apps list" in why


def test_typed_numbers_and_switches() -> None:
    assert steps.read_numbers("", 7) == [] and steps.read_numbers("  ", 7) == []
    assert steps.read_numbers("3", 7) == [3] and steps.read_numbers("3 4", 7) == [3, 4]
    assert steps.read_numbers("3,4", 7) == [3, 4]
    for wrong in ("0", "8", "x", "-1", "3 y", "1.5"):
        assert steps.read_numbers(wrong, 7) is None, wrong

    def switches(*argv: str) -> dict[str, bool]:
        return steps.switched(steps.parse(["--uv", "uv", *argv]))

    assert switches() == {}
    assert switches("--cpu") == {"gpu": False} == switches("--torch", "cpu")
    assert switches("--torch", "cu130") == {"gpu": True}
    assert switches("--rlbot") == {"rlbot": True} and switches("--no-rlbot") == {"rlbot": False}
    assert switches("--desktop") == {"start_menu": True, "desktop": True}
    assert switches("--no-shortcuts") == {"start_menu": False, "desktop": False}
    assert switches("--rust", "--dev", "--recordings") == {"rust": True, "dev": True, "recordings": True}
    assert steps.parse(["--uv", "uv", "--cpu", "--torch", "cu128"]).torch == "cpu"
    assert switches("--no-launch", "--from-source") == {}, "not parts: the menu is still shown"


def test_the_menu_lists_the_parts_and_takes_numbers_until_enter() -> None:
    found = _found(rlbot4="C:\\Users\\me\\AppData\\Local\\RLBotGUIX\\RLBotGUI.exe")
    text = steps.menu_text(found, steps.default_wishes(found, {}))
    lines = text.splitlines()
    assert "Always installed: Python" in text and "(ready-made)" in text and "arena files" in text
    assert "  [x] 1  PyTorch for the graphics card, NVIDIA GeForce RTX 3070 (9 GB on disk" in text
    assert any(line.startswith("  [x] 2  RLBot v5, to play your bots") for line in lines)
    assert "RLBot v4 is at C:\\Users\\me" in text and "need v5" in text
    assert "  [x] 3  Start menu shortcut" in lines
    assert any(line.startswith("  [ ] 4  Desktop shortcut") for line in lines)
    assert any(line.startswith("  [ ] 5  Tools to build your own Rust rewards") for line in lines)
    assert any(line.startswith("  [ ] 6  Developer tools") for line in lines)
    assert any(line.startswith("  [ ] 7  RocketSim's game recordings now (119 MB") for line in lines)
    assert lines[-1] == "Type a number to switch it on/off (or several: 3 4), Enter to install."
    source = steps.menu_text(_found(ready_made=False), {})
    assert "(built from its code)" in source and "[x] 5  Rust in NitroLearn's folder: required" in source
    plain = steps.menu_text(_found(card="", no_card="no NVIDIA card was found", windows=False), {})
    assert (
        "[ ] 1  PyTorch for the graphics card: no NVIDIA card was found, so PyTorch for the processor"
        in plain
    )
    assert "RLBot" not in plain and "shortcut" not in plain

    shown: list[str] = []
    answers = iter(["9", "2 4", "1", ""])
    chosen = steps.decide(found, {}, {}, lambda prompt: next(answers), shown.append)
    assert _parts(chosen) == {"start_menu", "desktop"}
    assert "Type a number from 1 to 7, or press Enter to install." in shown
    assert len([text for text in shown if text.startswith("\nWhat to install")]) == 4

    def ended(prompt: str) -> str:
        raise EOFError

    assert steps.decide(found, {}, {}, ended, shown.append) == steps.effective(
        found, steps.default_wishes(found, {})
    )
    # Switches: no menu, the switched parts on top of the defaults.
    assert _parts(steps.decide(found, {}, {"rlbot": False}, lambda prompt: pytest.fail("asked"))) == {
        "gpu",
        "start_menu",
    }
    alone = _found(person=False)
    assert _parts(steps.decide(alone, {}, {}, lambda prompt: pytest.fail("asked"))) == {"gpu"}


def test_what_is_installed_is_found_and_the_choices_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    monkeypatch.setattr(steps, "installed_version", lambda name: "2.14.1+cpu" if name == "torch" else None)
    found = steps.look_around(root, False, True, "", "no NVIDIA card was found", True, False)
    assert found.first and found.has == {
        "torch": True,
        "gpu": False,
        "start_menu": False,
        "desktop": False,
        "rust": False,
        "recordings": False,
    }
    record = root / steps.SHORTCUTS_RECORD
    record.parent.mkdir(parents=True)
    links = [str(root / "a.lnk"), str(root / "b.lnk")]
    for link in links:
        Path(link).write_bytes(b"lnk")
    record.write_text(json.dumps({"shortcuts": links}), encoding="utf-8")
    assert steps.shortcuts_made(root) == (True, True)
    record.write_text(json.dumps({"shortcuts": links[:1]}), encoding="utf-8")
    assert steps.shortcuts_made(root) == (True, False)
    rustup = root / ".tools" / "cargo" / "bin" / ("rustup.exe" if steps.WINDOWS else "rustup")
    rustup.parent.mkdir(parents=True)
    rustup.write_bytes(b"")
    recordings = steps.nitro_module("recordings")
    assert recordings is not None
    folder = root / "rocketsim_recordings"
    folder.mkdir()
    for name, (size, _) in recordings.RECORDINGS.items():
        (folder / name).write_bytes(bytes(size))
    monkeypatch.setenv("LOCALAPPDATA", str(root / "Local"))
    launcher = root / "Local" / "RLBot5" / "bin" / "launcher.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"")
    (root / "Local" / "RLBotGUIX").mkdir()
    found = steps.look_around(root, True, True, "", "no NVIDIA card was found", True, False)
    assert not found.first and found.rlbot4 == str(root / "Local" / "RLBotGUIX")
    assert {part for part, here in found.has.items() if here} == {
        "torch",
        "start_menu",
        "rust",
        "recordings",
        "rlbot",
    }
    assert steps.read_record(root) == {}
    steps.write_record(root, {"gpu": False, "dev": True})
    assert steps.read_record(root) == {"gpu": False, "dev": True}


def _setup_with(
    root: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str], *argv: str
) -> tuple[Any, Recorder]:
    """setup() in `root` with stand-ins for every download, a person typing `answers` in the menu."""
    recorder = Recorder(root / "site-packages")
    monkeypatch.setattr(steps, "run", recorder)
    monkeypatch.setattr(steps, "installed_version", lambda name: recorder.torch)
    monkeypatch.setattr(steps, "install_from_code", lambda uv, folder, extras, *rest: None)
    monkeypatch.setattr(steps, "read_release", lambda root, base, folder: None)
    release = steps.Release("1.2.0", "u", None, {})
    monkeypatch.setattr(steps, "find_ready_made", lambda release_: steps.ReadyMade(release, "w", "a"))
    monkeypatch.setattr(steps, "install_ready_made", lambda uv, root, found, extras, *rest: True)
    monkeypatch.setattr(steps, "ensure_rust", lambda root: recorder(["ensure_rust"]))
    monkeypatch.setattr(steps, "compiler_problem", lambda: None)
    pending = iter(answers)
    args = steps.parse(["--uv", "uv", *argv])
    facts = _facts("12.8", ("NVIDIA GeForce RTX 3070", "8.6"))
    done = steps.setup(args, root, "unused", lambda: facts, person=True, ask=lambda prompt: next(pending))
    return done, recorder


def test_setup_installs_what_the_menu_says(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Off: graphics-card PyTorch (1); on: Rust (5 on Windows, 2 here) and the recordings.
    rust, recordings = ("5", "7") if steps.WINDOWS else ("2", "4")
    done, recorder = _setup_with(_project(tmp_path), monkeypatch, [f"1 {rust} {recordings}", ""])
    assert done.choices["rust"] and done.choices["recordings"] and not done.choices["gpu"]
    assert recorder.torch == "2.14.1+cpu"
    assert ["ensure_rust"] in recorder.commands
    assert [sys.executable, "-m", "nitro", "fetch-recordings"] in recorder.commands


def test_a_run_again_removes_what_was_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    rustup = root / ".tools" / "cargo" / "bin" / ("rustup.exe" if steps.WINDOWS else "rustup")
    rustup.parent.mkdir(parents=True)
    rustup.write_bytes(b"")
    (root / ".tools" / "rustup").mkdir()
    recordings = steps.nitro_module("recordings")
    assert recordings is not None
    (root / "rocketsim_recordings").mkdir()
    for name, (size, _) in recordings.RECORDINGS.items():
        (root / "rocketsim_recordings" / name).write_bytes(bytes(size))
    (root / "rocketsim_recordings" / "mine.rlpr.zst").write_bytes(b"kept")
    rust, recordings_number = ("5", "7") if steps.WINDOWS else ("2", "4")
    done, recorder = _setup_with(root, monkeypatch, [f"{rust} {recordings_number}", ""])
    assert not done.choices["rust"] and not done.choices["recordings"]
    assert not (root / ".tools" / "cargo").exists() and not (root / ".tools" / "rustup").exists()
    assert sorted(path.name for path in (root / "rocketsim_recordings").iterdir()) == ["mine.rlpr.zst"]
    assert ["ensure_rust"] not in recorder.commands


def _finish(
    tmp_path: Path,
    choices: dict[str, bool],
    has: dict[str, bool] | None = None,
    windows: bool = True,
    person: bool = True,
    no_launch: bool = False,
    rlbot_switch: bool = False,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """after_install with stand-ins: the scripts run (with their arguments) and the `nitro` commands run."""
    ran: list[tuple[str, ...]] = []
    commands: list[tuple[str, ...]] = []

    def run_script(folder: Path, script: str, *args: str) -> int:
        assert folder == tmp_path
        ran.append((script, *args))
        return 0

    def nitro(*args: str) -> int:
        commands.append(args)
        return 0

    found = _found(windows=windows, person=person, has=has or {})
    every = dict.fromkeys(steps.PARTS, False) | choices
    done = steps.Done(0, found, every)
    steps.after_install(tmp_path, no_launch, done, rlbot_switch, run_script, nitro)
    return ran, commands


SHORTCUTS = "tools/app_shortcuts.ps1"
DASHBOARD = ("Shortcuts/shortcuts.ps1", "dashboard")


def test_after_the_install_the_shortcuts_and_rlbot_follow_the_choices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _finish(tmp_path, {"start_menu": True}) == ([(SHORTCUTS,), DASHBOARD], [])
    assert _finish(tmp_path, {"start_menu": True, "desktop": True})[0] == [(SHORTCUTS, "-Desktop"), DASHBOARD]
    assert _finish(tmp_path, {}, no_launch=True) == ([], []), "nothing chosen, nothing made"
    made = {"start_menu": True, "desktop": True}
    assert _finish(tmp_path, made, has=made) == ([DASHBOARD], []), "unchanged: not made again"
    assert _finish(tmp_path, {}, has=made)[0] == [(SHORTCUTS, "-Remove"), DASHBOARD]
    assert _finish(tmp_path, {"start_menu": True}, has=made)[0] == [(SHORTCUTS,), DASHBOARD]
    # RLBot v5 with its own installer when chosen and not installed.
    assert _finish(tmp_path, {"rlbot": True}) == ([DASHBOARD], [("install-rlbot",)])
    assert _finish(tmp_path, {"rlbot": True}, has={"rlbot": True}) == ([DASHBOARD], [])
    # Without a person: what the switches chose, and the dashboard is not opened.
    assert _finish(tmp_path, {"rlbot": True}, person=False) == ([], [("install-rlbot",)])
    # Off Windows: nothing; --rlbot says where to look.
    assert _finish(tmp_path, {}, windows=False, rlbot_switch=True) == ([], [])
    assert "rlbot.org" in capsys.readouterr().out


def test_a_failed_rlbot_install_says_what_to_do_and_the_switches_reach_the_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    done = steps.Done(0, _found(person=False), dict.fromkeys(steps.PARTS, False) | {"rlbot": True})
    steps.after_install(tmp_path, True, done, False, lambda *args: 0, lambda *args: 1)
    assert "nitro install-rlbot later" in capsys.readouterr().out
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    for switch, option in (
        ("RLBot", "--rlbot"),
        ("NoRLBot", "--no-rlbot"),
        ("Desktop", "--desktop"),
        ("NoShortcuts", "--no-shortcuts"),
        ("Dev", "--dev"),
        ("Recordings", "--recordings"),
        ("CPU", "--cpu"),
    ):
        assert f"[switch]${switch}" in ps1 and f"if (${switch}) {{ $steps += '{option}' }}" in ps1, switch
        assert f".PARAMETER {switch}\n" in ps1
        steps.parse(["--uv", "uv", option])
    sh = (PROJECT / "install.sh").read_text(encoding="utf-8")
    assert "--cpu" in sh and "--recordings" in sh


def test_a_redirected_install_has_no_person_to_ask(tmp_path: Path) -> None:
    """Run with its output to a file, the installer is not interactive."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import install_steps as s; print(s.interactive())"
    done = subprocess.run(
        [sys.executable, "-c", code, str(PROJECT / "tools")],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.stdout.strip() == "False", done.stderr


def test_main_keeps_everything_in_the_folder_and_says_why_it_stopped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def stop(args: Namespace, releases: str, person: bool) -> int:
        raise steps.Stop("no network")

    monkeypatch.setattr(steps, "setup", stop)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(os, "chdir", lambda path: None)
    assert steps.main(["--uv", "uv"]) == 1
    assert "SETUP STOPPED: no network" in capsys.readouterr().out
    assert os.environ["UV_CACHE_DIR"] == str(PROJECT / ".cache" / "uv")


def test_too_little_disk_space_stops_before_downloading_and_a_full_disk_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The space this install adds is checked before anything downloads; a step that fails with the disk
    nearly full says the disk is full, not the network or the antivirus."""
    gpu, cpu = f"{steps.TORCH_VERSION}+cu126", f"{steps.TORCH_VERSION}+cpu"
    first = _found()
    assert steps.space_needed_gb(first, {"gpu": True}, "", gpu) == pytest.approx(9.5)
    assert steps.space_needed_gb(first, {"rust": True}, "", cpu) == pytest.approx(3.5)
    again = _found(first=False, has={"torch": True, "gpu": True, "rust": True})
    assert steps.space_needed_gb(again, {"gpu": True, "rust": True}, gpu, gpu) == 0
    root = Path.cwd()
    steps.check_space(root, 9.5, free=20.0)
    steps.check_space(root, 0, free=0.1)
    with pytest.raises(steps.Stop, match=r"needs about 10 GB more on .*which has 4\.0 GB free"):
        steps.check_space(root, 9.5, free=4.0)
    assert steps.failure_hint(root) == steps.DOWNLOAD_HINT
    monkeypatch.setattr(steps, "free_gb", lambda folder: 0.2)
    assert "is full" in steps.failure_hint(root) and "Antivirus" not in steps.failure_hint(root)
    with pytest.raises(steps.Stop, match="is full"):
        steps.run([sys.executable, "-c", "raise SystemExit(1)"], minutes=1)
    assert steps.out_of_space(OSError(28, "No space left on device"))
    assert not steps.out_of_space(OSError(13, "Permission denied"))


def test_the_windows_installer_warns_of_onedrive_and_long_paths_and_checks_the_disk() -> None:
    """install.ps1, before downloading: a folder OneDrive syncs and a very long path are said (a person may
    go on; the setup wizard and updates, which pass -Parts, are never asked), too little room stops it, and a
    uv step that fails with the disk full says so instead of the network hint."""
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        assert f"$env:{name}" in ps1
    assert "$PSScriptRoot.Length -gt 80" in ps1
    assert "-not $PSBoundParameters.ContainsKey('Parts') -and -not [Console]::IsInputRedirected" in ps1
    assert ps1.index("Get-FreeGB\nif ($free -ge 0 -and $free -lt 1)") < ps1.index("Getting uv")
    assert ps1.count("$(Get-FailureHint)") == 3 and '$DownloadHint"' not in ps1
    assert "The disk with NitroLearn's folder is full." in ps1


def test_a_full_disk_mid_install_stops_with_a_plain_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def full(args: Namespace, releases: str, person: bool) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(steps, "setup", full)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(os, "chdir", lambda path: None)
    assert steps.main(["--uv", "uv"]) == 1
    assert "SETUP STOPPED: The disk with NitroLearn's folder" in capsys.readouterr().out


def test_a_failed_final_check_still_makes_the_shortcuts_and_rlbot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The final `nitro doctor` finding a problem does not skip the shortcuts and RLBot chosen: only the
    dashboard does not open by itself, and the installer ends with its own code for "installed, but"."""
    choices = dict.fromkeys(steps.PARTS, False) | {"start_menu": True, "rlbot": True}
    finished: list[tuple[bool, Any]] = []
    for doctor in (0, 1):
        monkeypatch.setattr(
            steps,
            "setup",
            lambda args, releases, person, doctor=doctor: steps.Done(doctor, _found(), choices),
        )
        monkeypatch.setattr(steps, "write_record", lambda root, chosen: None)
        monkeypatch.setattr(
            steps,
            "after_install",
            lambda root, no_launch, done, rlbot: finished.append((no_launch, done.code)),
        )
        monkeypatch.setattr(os, "environ", dict(os.environ))
        monkeypatch.setattr(os, "chdir", lambda path: None)
        code = steps.main(["--uv", "uv"])
        assert code == (0 if doctor == 0 else steps.CHECK_FAILED)
    assert finished == [(False, 0), (True, 1)], "made both times; the dashboard opens only after a good check"
    out = capsys.readouterr().out
    assert "NitroLearn is installed, but its final check" in out, "the setup wizard reads this line"
    cmd = (PROJECT / "Install NitroLearn.cmd").read_text(encoding="utf-8")
    assert f'"%CODE%"=="{steps.CHECK_FAILED}"' in cmd, "the window says what really happened"


def test_a_run_again_after_an_unfinished_install_keeps_its_choices() -> None:
    """Parts chosen at the last install but not made (it stopped first) start ticked on a run again."""
    found = _found(first=False, has={"torch": True, "gpu": True})
    recorded = {"rlbot": True, "start_menu": True, "desktop": False, "recordings": True}
    wishes = steps.default_wishes(found, recorded)
    assert _parts(steps.effective(found, wishes)) == {"gpu", "rlbot", "start_menu", "recordings"}
    alone = _found(first=False, person=False, has={"torch": True, "gpu": True})
    assert steps.default_wishes(alone, recorded)["start_menu"], "also without a person to ask"


def test_a_step_reads_no_input_and_a_stalled_or_failed_download_says_what_holds_it_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    python = sys.executable
    assert steps.run([python, "-c", "import sys; sys.exit(0 if sys.stdin.read() == '' else 3)"]) == 0
    sleeper = [python, "-c", "import time; time.sleep(60)"]
    with pytest.raises(
        steps.Stop, match=r"stopped after 0\.02 minutes without finishing\. Check the internet"
    ):
        steps.run(sleeper, minutes=0.02)
    assert steps.run(sleeper, check=False, minutes=0.02) == steps.TIMED_OUT
    assert "Antivirus web shields" in capsys.readouterr().out
    with pytest.raises(steps.Stop, match="failed \\(see above\\)\\. Check the internet connection"):
        steps.run([python, "-c", "raise SystemExit(2)"], minutes=1)
    with pytest.raises(steps.Stop) as plain:
        steps.run([python, "-c", "raise SystemExit(2)"])
    assert "Antivirus" not in str(plain.value), "only a download step names the usual causes"
    missing = (tmp_path / "missing.whl").as_uri()
    with pytest.raises(steps.Stop, match=r"could not be downloaded .*Antivirus web shields"):
        steps.fetch_checked(missing, "0" * 64, tmp_path / "out.whl")


def test_every_uv_call_has_the_folder_settings_and_the_system_certificates() -> None:
    """uv uses the computer's own certificate store (antivirus and company proxies sign HTTPS with a root
    that is only there), and both installers set the folder's settings before their first uv call, which
    runs with a time limit and nothing to read on its input."""
    assert steps.folder_env(Path("/nitro"), True)["UV_SYSTEM_CERTS"] == "1"
    assert steps.folder_env(Path("/nitro"), False)["UV_SYSTEM_CERTS"] == "1"
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    sh = (PROJECT / "install.sh").read_text(encoding="utf-8")
    assert ps1.index(". (Join-Path $PSScriptRoot 'tools\\env.ps1')") < ps1.index("& $Uv")
    assert sh.index('. "$root/tools/env.sh"') < sh.index("    get_uv\n")
    calls = [line.strip() for line in ps1.splitlines() if re.search(r"(?:& \$Uv|Invoke-Uv) ", line)]
    assert calls == [
        "$said = (& $Uv --version 2>$null) -join ' '",
        """Invoke-Uv 15 "Installing Python $PythonVersion" @('python', 'install', $PythonVersion)""",
        "Invoke-Uv 10 'Making .venv' @('venv', '--seed', '--python', $PythonVersion, '.venv')",
    ]
    assert "RedirectStandardInput = $true" in ps1 and "WaitForExit($Minutes * 60 * 1000)" in ps1
    assert re.findall(r'^\s+(?:limited \d+ "[^"]+" )?"\$uv" (\w+)', sh, re.MULTILINE) == ["python", "venv"]
    assert re.findall(r'limited (\d+) "[^"]+" "\$uv"', sh) == ["15", "10"]
    said = re.search(r"^\$DownloadHint = '(.*)'$", ps1, re.MULTILINE)
    assert said is not None and said[1].replace("''", "'") == steps.DOWNLOAD_HINT
    assert f'DOWNLOAD_HINT="{steps.DOWNLOAD_HINT}"' in sh


# ---------- install.sh ----------

INSTALL_SH = PROJECT / "install.sh"
NO_RELEASE = "file:///nonexistent/v{version}/"  # no release of NitroLearn's to download from


def _no_release() -> dict[str, str]:
    return {**os.environ, "NITRO_RELEASES": NO_RELEASE}


bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="install.sh is for Linux"
)


def _stub(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@bash
def test_install_sh_puts_uv_python_and_the_venv_inside_the_folder(tmp_path: Path) -> None:
    """install.sh with a stand-in uv (it records what it was asked and makes .venv) and stand-in steps:
    every folder uv and the steps use is inside NitroLearn's, and nothing is written to the home folder."""
    root = tmp_path / "Nitro Learn"  # a space in the path
    (root / "tools").mkdir(parents=True)
    shutil.copy(INSTALL_SH, root / "install.sh")
    shutil.copy(PROJECT / "tools" / "env.sh", root / "tools" / "env.sh")
    log = tmp_path / "uv.log"
    uv = _stub(
        tmp_path / "uv",
        f"""echo "$* | $UV_PYTHON_INSTALL_DIR | $UV_CACHE_DIR | $UV_PYTHON_INSTALL_BIN" >> "{log}"
case "$1" in
  --version) echo "uv 0.11.32 (stand-in)" ;;
  venv) mkdir -p .venv/bin && ln -s "{sys.executable}" .venv/bin/python
        echo "home = $UV_PYTHON_INSTALL_DIR/cpython-3.11/bin" > .venv/pyvenv.cfg ;;
esac
""",
    )
    (root / "tools" / "install_steps.py").write_text(
        "import json, os, sys\n"
        "json.dump({'argv': sys.argv[1:], 'env': dict(os.environ)}, open(sys.argv[0] + '.json', 'w'))\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "NITRO_UV": str(uv), "NITRO_RELEASES": NO_RELEASE}
    done = subprocess.run(
        ["bash", str(root / "install.sh"), "--torch", "cpu"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    seen = json.loads((root / "tools" / "install_steps.py.json").read_text(encoding="utf-8"))
    placed = root / ".tools" / "uv" / "uv"
    assert seen["argv"] == ["--uv", str(placed), "--torch", "cpu"] and placed.is_file()
    for name in (
        "UV_PYTHON_INSTALL_DIR",
        "UV_CACHE_DIR",
        "UV_TOOL_DIR",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "TMPDIR",
    ):
        assert Path(seen["env"][name]).is_relative_to(root), name
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith(f"python install 3.11 | {root}/.tools/python | {root}/.cache/uv | 0")
    assert any(call.startswith("venv --seed --python 3.11 .venv") for call in calls)
    assert list(home.iterdir()) == [], "nothing in the home folder"

    # Run again: .venv works, so it is kept.
    log.unlink()
    done = subprocess.run(
        ["bash", str(root / "install.sh")], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0 and not any(c.startswith("venv") for c in log.read_text().splitlines())
    # A .venv made from a Python outside the folder (an earlier, system-Python install) is made again.
    (root / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    done = subprocess.run(
        ["bash", str(root / "install.sh")], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0 and "uses a Python outside this folder" in done.stdout
    assert ".tools/python" in (root / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")


@bash
def test_install_sh_refuses_a_download_that_does_not_match_its_hash(tmp_path: Path) -> None:
    fake = tmp_path / "uv.whl"
    fake.write_bytes(b"not uv")
    script = (
        f'source "{INSTALL_SH}"; root="{tmp_path}"; uv="{tmp_path}/.tools/uv/uv"; UV_URL="{fake.as_uri()}"; '
        "get_uv"
    )
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False, env=_no_release()
    )
    assert done.returncode == 1 and "did not match its SHA-256 hash" in done.stderr
    assert not (tmp_path / ".tools" / "uv" / "uv").exists()


@bash
def test_install_sh_takes_uv_and_python_from_a_folder_of_the_release_files(tmp_path: Path) -> None:
    """uv and Python from `release-files` (or --bundles) before anything is downloaded, each checked against
    its pin; Python is laid out as uv's mirror folder expects."""
    files = tmp_path / "release-files"
    files.mkdir()
    wheel = files / "uv-stand-in.whl"
    with zipfile.ZipFile(wheel, "w") as out:
        out.writestr("uv-0.11.32.data/scripts/uv", "#!/bin/sh\necho uv 0.11.32\n")
    python = files / "python-3.11.15-linux-x64.tar.gz"
    python.write_bytes(b"python")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    py_sha = hashlib.sha256(b"python").hexdigest()
    script = (
        f'source "{INSTALL_SH}"; root="{tmp_path}"; uv="{tmp_path}/.tools/uv/uv"; UV_WHEEL=uv-stand-in.whl; '
        f"UV_SHA256={sha}; PYTHON_SHA256={py_sha}; UV_URL=file:///nonexistent; "
        f'UV_PYTHON_INSTALL_DIR="{tmp_path}/.tools/python"; find_release_files; get_uv; python_mirror; '
        'echo "mirror=${UV_PYTHON_INSTALL_MIRROR:-}"'
    )
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False, env=_no_release()
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / ".tools" / "uv" / "uv").read_text().startswith("#!/bin/sh")
    mirror = tmp_path / ".tmp" / "python-mirror"
    assert f"mirror=file://{mirror}" in done.stdout
    placed = (
        mirror / "20260718" / "cpython-3.11.15+20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    )
    assert placed.read_bytes() == b"python"
    # A copy that does not match its pin is never used; uv then downloads Python itself.
    python.write_bytes(b"changed")
    shutil.rmtree(tmp_path / ".tmp")
    done = subprocess.run(
        ["bash", "-c", script.replace("get_uv; ", "")], capture_output=True, text=True, timeout=60,
        check=False, env=_no_release(),
    )  # fmt: skip
    assert done.returncode == 0 and "mirror=\n" in done.stdout + "\n" and not mirror.exists()
    # --bundles names another folder.
    script = f'source "{INSTALL_SH}"; root="{tmp_path}"; find_release_files --cpu --bundles /else; '
    script += "echo $release_files"
    said = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert said.stdout.strip() == "/else"


@bash
def test_install_sh_stops_a_stalled_or_failed_step_with_a_plain_message(tmp_path: Path) -> None:
    for fake, said in (
        ("timeout() { return 124; }; ", "Thing was stopped after 1 minutes without finishing. Check the"),
        ("", "Thing failed (see above). Check the internet connection. Antivirus web shields"),
    ):
        script = f'source "{INSTALL_SH}"; {fake}limited 1 "Thing" false'
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 1 and said in done.stderr, done.stderr
    script = f'source "{INSTALL_SH}"; limited 1 "Reading" bash -c \'read -r line && exit 3 || exit 0\''
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, "nothing to read on its input"


def _pypi_reachable() -> bool:
    try:
        with urllib.request.urlopen("https://pypi.org/simple/uv/", timeout=10):
            return True
    except OSError:
        return False


@bash
def test_install_sh_downloads_the_pinned_uv_from_pypi(tmp_path: Path) -> None:
    """The real download (27 MB): uv's package on PyPI matches the pinned hash and its uv runs."""
    if not _pypi_reachable():
        pytest.skip("PyPI cannot be reached from here")
    script = f'source "{INSTALL_SH}"; root="{tmp_path}"; uv="{tmp_path}/.tools/uv/uv"; get_uv'
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=600, check=False, env=_no_release()
    )
    assert done.returncode == 0, done.stdout + done.stderr
    said = subprocess.run(
        [str(tmp_path / ".tools" / "uv" / "uv"), "--version"], capture_output=True, text=True
    )
    assert said.stdout.startswith("uv 0.11.32")
    assert not list((tmp_path / ".tmp").iterdir()), "the download is cleaned up"


# ---------- a folder whose name has letters outside the ANSI code page ----------


def _program(subsystem: int) -> bytes:
    """The start of a Windows program: its MZ header pointing at a PE header with this subsystem."""
    head = bytearray(512)
    head[0:2] = b"MZ"
    pe = 0x80
    head[0x3C:0x40] = pe.to_bytes(4, "little")
    head[pe : pe + 4] = b"PE\0\0"
    head[pe + 24 : pe + 26] = (0x20B).to_bytes(2, "little")  # PE32+
    head[pe + 24 + 68 : pe + 24 + 70] = subsystem.to_bytes(2, "little")
    return bytes(head)


def test_the_windowless_python_is_made_a_windowed_program(tmp_path: Path) -> None:
    """uv's .venv\\Scripts\\pythonw.exe is a console program that starts python.exe, which opened an empty
    console window behind the dashboard: it is replaced by Python's own windowed launcher for environments."""
    base = tmp_path / "base"
    launcher = base / "Lib" / "venv" / "scripts" / "nt" / "pythonw.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(_program(steps.GUI_PROGRAM) + b"launcher")
    venv = tmp_path / "José-Łódź" / ".venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    windowless = scripts / "pythonw.exe"
    windowless.write_bytes(_program(steps.CONSOLE_PROGRAM) + b"trampoline")
    assert steps.program_subsystem(windowless) == steps.CONSOLE_PROGRAM
    assert steps.windowless_python(venv, base) == "fixed"
    assert windowless.read_bytes() == launcher.read_bytes()
    assert sorted(path.name for path in scripts.iterdir()) == ["pythonw.exe"], "nothing left beside it"
    assert steps.windowless_python(venv, base) == "fine", "a windowed pythonw.exe stays as it is"
    # A console pythonw.exe with no windowed launcher to take its place stays, and the reason is given.
    windowless.write_bytes(_program(steps.CONSOLE_PROGRAM))
    launcher.write_bytes(_program(steps.CONSOLE_PROGRAM))
    assert "not Python's windowed launcher" in steps.windowless_python(venv, base)
    assert steps.program_subsystem(windowless) == steps.CONSOLE_PROGRAM
    assert steps.program_subsystem(tmp_path / "missing.exe") is None
    (tmp_path / "text.exe").write_bytes(b"MZ not a program")
    assert steps.program_subsystem(tmp_path / "text.exe") is None


def test_a_deleted_shortcut_counts_as_missing_and_is_made_again(tmp_path: Path) -> None:
    """The shortcuts present are the recorded .lnk files still on disk, not only the recorded choice."""
    start, desktop = tmp_path / "Programs" / "NitroLearn.lnk", tmp_path / "Desktop" / "NitroLearn.lnk"
    for link in (start, desktop):
        link.parent.mkdir()
        link.write_bytes(b"lnk")
    record = tmp_path / steps.SHORTCUTS_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"shortcuts": [str(start), str(desktop)]}), encoding="utf-8")
    assert steps.shortcuts_made(tmp_path) == (True, True)
    start.unlink()
    assert steps.shortcuts_made(tmp_path) == (False, True)
    has = dict(zip(("start_menu", "desktop"), steps.shortcuts_made(tmp_path), strict=True))
    wanted = {"start_menu": True, "desktop": True}
    assert _finish(tmp_path, wanted, has=has)[0] == [(SHORTCUTS, "-Desktop"), DASHBOARD]
    desktop.unlink()
    has = dict(zip(("start_menu", "desktop"), steps.shortcuts_made(tmp_path), strict=True))
    assert _finish(tmp_path, {"start_menu": True}, has=has)[0] == [(SHORTCUTS,), DASHBOARD]
    record.write_text(json.dumps({"shortcuts": [7]}), encoding="utf-8")
    assert steps.shortcuts_made(tmp_path) == (False, False)


def test_the_cuda_cache_path_is_one_nvidias_driver_can_read(tmp_path: Path) -> None:
    """NVIDIA's driver reads CUDA_CACHE_PATH through an ANSI getenv: a path that is not plain ASCII becomes
    its 8.3 short form when that is ASCII, else it is not set (the driver's own folder is used)."""
    plain = str(tmp_path / "NitroLearn" / ".cache" / "cuda")
    env = {"CUDA_CACHE_PATH": plain, "TORCH_HOME": "x"}
    folder_env.native_safe(env, windows=True, short=lambda path: pytest.fail("ASCII needs no short form"))
    assert env["CUDA_CACHE_PATH"] == plain
    other = str(tmp_path / "José-Łódź" / "NitroLearn" / ".cache" / "cuda")
    env = {"CUDA_CACHE_PATH": other}
    folder_env.native_safe(
        env, windows=True, short=lambda path: "D:\\NITROT~1\\JOSE-~1\\NITROL~1\\.cache\\cuda"
    )
    assert env == {"CUDA_CACHE_PATH": "D:\\NITROT~1\\JOSE-~1\\NITROL~1\\.cache\\cuda"}
    assert Path(other).is_dir(), "made first: only a folder that exists has a short form"
    for short in (lambda path: None, lambda path: path):  # no short names on the drive, or not ASCII either
        env = {"CUDA_CACHE_PATH": other, "TORCH_HOME": other}
        folder_env.native_safe(env, windows=True, short=short)
        assert env == {"TORCH_HOME": other}, "Python reads the others: they stay"
    env = {"CUDA_CACHE_PATH": other}
    folder_env.native_safe(env, windows=False)
    assert env == {"CUDA_CACHE_PATH": other}, "only Windows has the ANSI code page"
    assert folder_env.short_path(other) is None or sys.platform == "win32"
    # The folder's environment as the installer and the dashboard apply it.
    root = tmp_path / "Łódź"
    env = {"PATH": "elsewhere"}
    folder_env.apply(root, env, windows=True)
    assert "CUDA_CACHE_PATH" not in env or env["CUDA_CACHE_PATH"].isascii()
    assert env["TORCH_HOME"] == str(root / ".cache" / "torch")
    ps1 = (PROJECT / "tools" / "env.ps1").read_text(encoding="utf-8")
    assert "Scripting.FileSystemObject" in ps1 and "Remove-Item Env:\\CUDA_CACHE_PATH" in ps1
    assert ps1.index("Remove-Item Env:\\CUDA_CACHE_PATH") > ps1.index("$env:CUDA_CACHE_PATH = Join-Path")
    init = (PROJECT / "python" / "nitro" / "__init__.py").read_text(encoding="utf-8")
    assert "_folder_env.native_safe(_os.environ)" in init, "every NitroLearn process, before PyTorch"


SHIPPED_SCRIPTS = sorted(
    path
    for pattern in ("*.cmd", "*.ps1", "Shortcuts/*.cmd", "Shortcuts/*.ps1", "tools/*.ps1", "installer/*.iss")
    for path in PROJECT.glob(pattern)
)


def test_scripts_carry_no_letters_a_code_page_could_change() -> None:
    """Windows PowerShell 5.1 reads a .ps1 without a byte-order mark in the ANSI code page and cmd.exe reads a
    .cmd in the console's OEM code page: they are plain ASCII, and every path in a .cmd comes from where the
    file is (%~dp0), never a folder name written into it."""
    assert len(SHIPPED_SCRIPTS) >= 10
    for path in SHIPPED_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert text.isascii(), path
        if path.suffix == ".cmd":
            assert not re.search(r"[A-Za-z]:\\", text), path


def test_paths_reach_windows_powershell_without_an_ansi_round_trip() -> None:
    """Files the PowerShell scripts read are read as UTF-8; the shortcuts are made through IShellLinkW, not
    WScript.Shell (whose shortcut object takes paths through the ANSI code page)."""
    for path in SHIPPED_SCRIPTS:
        if path.suffix != ".ps1":
            continue
        text = path.read_text(encoding="utf-8")
        for line in re.findall(r"^.*Get-Content.*$", text, re.MULTILINE):
            assert "-Encoding UTF8" in line, f"{path.name}: {line.strip()}"
        assert "-ComObject WScript.Shell" not in text, path
    shortcuts = (PROJECT / "tools" / "app_shortcuts.ps1").read_text(encoding="utf-8")
    assert "IShellLinkW" in shortcuts and "MarshalAs(UnmanagedType.LPWStr)" in shortcuts
    install = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    assert "Get-Content -Encoding UTF8 -LiteralPath $config" in install, "pyvenv.cfg as uv writes it"
