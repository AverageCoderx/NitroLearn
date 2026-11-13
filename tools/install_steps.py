"""The installer's steps once uv has put a Python and an empty `.venv` in NitroLearn's folder: install.ps1
(Windows) and install.sh (Linux) run this with `.venv`'s Python. Standard library only, since NitroLearn is
not installed yet when it starts.

    python tools/install_steps.py --uv <uv> [--torch auto|cpu|cu126|cu128|cu130] [--cpu | --gpu] [--rust]
                                  [--from-source] [--dev] [--recordings] [--no-launch] [--rlbot | --no-rlbot]
                                  [--desktop | --start-menu | --no-shortcuts] [--parts LIST]
                                  [--bundles FOLDER] [--progress]

First it looks at the PC (the NVIDIA card, whether the release has a ready-made core for it, what is
installed already) and, with a person at the console and no switch that sets a part, shows a menu of the
optional parts (graphics-card PyTorch, RLBot v5, the shortcuts, Rust for own Rust rewards, developer tools,
RocketSim's recordings), ticked as they are now; the switches set parts without the menu. The choices are
recorded in .tools/install_choices.json, and a run again changes only what was switched.

Then, in order: PyTorch (first, from PyTorch's own index), NitroLearn (the release's ready-made core and
programs when there are some for this version, else built from its code with Rust), the arena meshes (`nitro
fetch-meshes`), the recordings when chosen, then `nitro doctor`. The release also carries its own copies of
the Python packages and the Rust crates (tools/bundles.py): they are used when it has them, so only PyTorch
and the arena meshes come from elsewhere; with a folder of the release's files (`--bundles`, or a
`release-files` folder in NitroLearn's), nothing else is downloaded at all. Everything it downloads or
creates stays in NitroLearn's folder (python/nitro/folder_env.py); safe to run again. Afterwards (Windows)
it makes or removes the shortcuts and opens RLBot v5's own installer when chosen: RLBot is a separate
program that installs itself outside the folder.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSION = "2.14.1"
TORCH_INDEX = "https://download.pytorch.org/whl/"
TORCH_VARIANTS = ("cpu", "cu126", "cu128", "cu130")
# Where the release's ready-made files are: the core (a wheel per platform), the programs, the bundled
# copies (tools/bundles.py), SHA256SUMS.txt.
RELEASES = "https://github.com/AverageCoderx/NitroLearn/releases/download/v{version}/"
SUMS = "SHA256SUMS.txt"
RELEASE_FOLDER = "release-files"  # a folder of the release's files in NitroLearn's folder: used offline
RELEASE_FILES_ENV = "NITRO_RELEASE_FILES"  # tells NitroLearn's own downloads (the recordings) about it
EXTRAS = "wandb,replays,dashboard"
RUSTUP = "https://static.rust-lang.org/rustup/dist/{triple}/rustup-init{exe}"
WINDOWS = sys.platform == "win32"
PROGRAMS = ("nitro_bot", "nitro-replicar", "replicar")
PROGRAMS_NOTICES = "THIRD_PARTY_NOTICES.md"  # the programs' licence notices, kept beside them
CORE_NOTICES = "THIRD_PARTY_NOTICES.md"  # the compiled core's, beside it in python/nitro
FOLDER_PTH = "nitrolearn_folder.pth"  # points Python at NitroLearn's python folder (ready-made core)
BUILD_TOOLS_URL = "https://visualstudio.microsoft.com/downloads/"
# Time limits of the steps that download, generous for a slow connection: past one, the step has stalled.
TORCH_MINUTES = 120  # up to 3 GB
PACKAGES_MINUTES = 30
BUILD_MINUTES = 120  # downloads, then compiles the Rust code
RUST_MINUTES = 60
MESHES_MINUTES = 15


def folder_env(root: Path, windows: bool = WINDOWS) -> dict[str, str]:
    """Every folder a tool would otherwise write to outside NitroLearn, pointed inside `root`, and cargo's
    network limits (a download that stalls fails after 3 tries, so the build step stops with DOWNLOAD_HINT
    instead of waiting out its time limit): python/nitro/folder_env.py, which the dashboard's windowless start
    uses too. tools/env.ps1 and nitro.cmd (Windows) and tools/env.sh (Linux) set the same; a test keeps them
    equal."""
    module = nitro_module("folder_env")
    if module is None:
        raise Stop("python/nitro/folder_env.py is missing: this folder is not a complete NitroLearn")
    return dict(module.variables(root, windows))


def say(text: str) -> None:
    print(f"\n==> {text}", flush=True)


PROGRESS = [False]  # --progress: a line per step for the setup wizard's checklist


def step(key: str, state: str) -> None:
    """With --progress, `@@nitro <key> <state>` (start, done or fail) for the setup wizard; its keys:
    pc, torch, core, rust, meshes, recordings, check, shortcuts, rlbot (install.ps1 adds python)."""
    if PROGRESS[0]:
        print(f"@@nitro {key} {state}", flush=True)


def note(text: str) -> None:
    print(f"    {text}", flush=True)


class Stop(Exception):
    """Setup cannot go on; the message says what to do."""


# Said whenever a download fails or stalls.
DOWNLOAD_HINT = (
    "Check the internet connection. Antivirus web shields (HTTPS scanning) and company proxies often hold "
    "back or block downloads like these: allow this folder in your antivirus (or pause its web shield for "
    "the install), then run the installer again. What was already downloaded is kept."
)
LOW_DISK_GB = 1.0  # less free than this after a failed step: the disk filled up, not the network


def free_gb(folder: Path) -> float | None:
    """The free space on `folder`'s disk in GB, or None when it cannot be read."""
    try:
        return shutil.disk_usage(folder).free / 1024**3
    except OSError:
        return None


def disk_full(folder: Path) -> str:
    """Said when a step stops because the disk with `folder` is full."""
    return (
        f"The disk with NitroLearn's folder ({folder.anchor or folder}) is full. Free up space (NitroLearn "
        "needs about 10 GB with PyTorch for the graphics card, about 2 GB without), or install NitroLearn on "
        "a drive with more room, then run the installer again. What was already downloaded is kept."
    )


def failure_hint(folder: Path | None = None) -> str:
    """Why a download step usually fails: a full disk when there is almost no space left, else
    DOWNLOAD_HINT."""
    free = free_gb(folder or ROOT)
    return disk_full(folder or ROOT) if free is not None and free < LOW_DISK_GB else DOWNLOAD_HINT


def out_of_space(err: OSError) -> bool:
    """Whether `err` says the disk is full (ENOSPC; Windows' ERROR_DISK_FULL or ERROR_HANDLE_DISK_FULL)."""
    return err.errno == errno.ENOSPC or getattr(err, "winerror", None) in (39, 112)


# Said after rustup's own note that Cargo's bin folder is not on the PATH (--no-modify-path is on purpose).
RUSTUP_PATH_NOTE = (
    "Rust is installed in this folder. Ignore rustup's note about adding Cargo's bin directory to the PATH: "
    "NitroLearn finds Rust in its own folder by itself."
)
TIMED_OUT = 124  # run()'s code for a program stopped at its time limit


def run(command: Sequence[str | Path], check: bool = True, minutes: float | None = None) -> int:
    """Runs a program with its output shown and nothing to read on its input; a failure stops setup when
    `check`. With `minutes` (a step that downloads), the program is stopped once it has run that long, and
    a failure or the stop says what usually holds downloads back."""
    what = f"{Path(str(command[0])).name} {command[1] if len(command) > 1 else ''}".strip()
    try:
        code = subprocess.run(
            [str(part) for part in command],
            stdin=subprocess.DEVNULL,
            timeout=None if minutes is None else minutes * 60,
            check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        stuck = f"{what} was stopped after {minutes} minutes without finishing. {failure_hint()}"
        if check:
            raise Stop(stuck) from None
        note(stuck)
        return TIMED_OUT
    if check and code != 0:
        raise Stop(f"{what} failed (see above)." + (f" {failure_hint()}" if minutes is not None else ""))
    return code


# ---------- the PyTorch build for this PC ----------


@dataclass(frozen=True)
class Card:
    name: str
    capability: tuple[int, int] | None


@dataclass(frozen=True)
class NvidiaFacts:
    driver_cuda: tuple[int, int] | None  # the newest CUDA the driver runs
    cards: tuple[Card, ...]


def parse_nvidia(table: str, query: str) -> NvidiaFacts:
    """`nvidia-smi`'s header (the driver's CUDA version) and `--query-gpu=name,compute_cap` lines."""
    found = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", table)
    cards = []
    for line in query.splitlines():
        name, _, cap = line.rpartition(",")
        if not name.strip():
            name, cap = cap, ""
        number = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", cap)
        cards.append(Card(name.strip(), (int(number[1]), int(number[2])) if number else None))
    return NvidiaFacts((int(found[1]), int(found[2])) if found else None, tuple(cards))


def nvidia_facts() -> NvidiaFacts | None:
    """What the NVIDIA driver says, or None without one."""
    smi = shutil.which("nvidia-smi")
    if smi is None and WINDOWS:
        system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(system) if system.is_file() else None
    if smi is None:
        return None
    try:
        table = subprocess.run([smi], capture_output=True, text=True, timeout=60, check=False)
        query = subprocess.run(
            [smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if table.returncode != 0:
        return None
    return parse_nvidia(table.stdout, query.stdout if query.returncode == 0 else "")


def _capability(card: Card) -> tuple[int, int] | None:
    if card.capability is not None:
        return card.capability
    if re.search(r"RTX\s*50\d\d|RTX PRO \d+ Blackwell", card.name, re.IGNORECASE):
        return (12, 0)  # an old driver that cannot say it
    return None


def choose_torch(facts: NvidiaFacts | None) -> tuple[str, list[str]]:
    """The PyTorch build (index name) for this PC, with notes for the user. CUDA 12.6 is the build
    NitroLearn's speed numbers were measured with; cards it has no code for (compute capability 10 and
    newer: RTX 50xx) get CUDA 12.8. No NVIDIA driver, or a card too old for PyTorch: the CPU build."""
    if facts is None:
        return "cpu", [
            "No NVIDIA driver found: installing PyTorch for the processor (training is many times slower).",
            "If this PC has an NVIDIA card, install its driver from https://www.nvidia.com/drivers, restart, "
            "and run the installer again.",
        ]
    caps = [cap for cap in map(_capability, facts.cards) if cap is not None]
    notes = []
    if caps and max(caps) < (5, 0):
        return "cpu", ["This NVIDIA card is too old for PyTorch: installing PyTorch for the processor."]
    variant = "cu128" if caps and max(caps) >= (10, 0) else "cu126"
    if caps and max(caps) >= (10, 0) and min(caps) < (7, 5):
        notes.append(
            "This PC also has an older card that PyTorch's CUDA 12.8 build has no code for: train on the "
            "newer card."
        )
    needed = (12, 8) if variant == "cu128" else (12, 6)
    if facts.driver_cuda is not None and facts.driver_cuda < needed:
        have = ".".join(map(str, facts.driver_cuda))
        notes.append(
            f"The NVIDIA driver runs CUDA {have}; this PyTorch needs {needed[0]}.{needed[1]}. Update the "
            "driver (https://www.nvidia.com/drivers or the NVIDIA app) and restart; nothing needs installing "
            "again."
        )
    return variant, notes


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_torch(uv: Path, variant: str) -> None:
    wanted = f"{TORCH_VERSION}+{variant}"
    if installed_version("torch") == wanted:
        note(f"PyTorch {wanted} is already installed.")
        return
    size = "about 200 MB" if variant == "cpu" else "about 3 GB"
    say(f"Installing PyTorch {wanted} ({size} to download the first time)")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            f"torch=={wanted}",
            "--index-url",
            TORCH_INDEX + variant,
        ],
        minutes=TORCH_MINUTES,
    )


# ---------- NitroLearn itself ----------


def project_version(root: Path) -> str:
    """The version in Cargo.toml (the release's tag is v<version>)."""
    text = (root / "Cargo.toml").read_text(encoding="utf-8")
    found = re.search(
        r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', text, re.MULTILINE | re.DOTALL
    )
    if found is None:
        raise Stop("Cargo.toml names no version")
    return found[1]


def platform_tag() -> str | None:
    """Which of the release's ready-made files fit this PC: 'windows-x64', 'linux-x64', or None."""
    if sys.platform == "win32":
        machine = os.environ.get("PROCESSOR_ARCHITECTURE", "")
        return "windows-x64" if machine.upper() in ("AMD64", "X86_64") else None
    if sys.platform.startswith("linux") and os.uname().machine == "x86_64":
        return "linux-x64"
    return None


def release_files(sums: str, version: str, platform: str) -> tuple[str, str] | None:
    """The core's wheel and the programs' archive for `platform` in SHA256SUMS.txt's lines."""
    names = [line.split()[-1].lstrip("*") for line in sums.splitlines() if len(line.split()) == 2]
    wheel_end = "win_amd64.whl" if platform == "windows-x64" else "x86_64.whl"
    wheels = [
        name
        for name in names
        if name.startswith(f"nitrolearn-{version}-")
        and name.endswith(wheel_end)
        and (platform == "windows-x64" or "linux" in name)
    ]
    archive = f"nitrolearn-programs-{version}-{platform}.{'zip' if platform == 'windows-x64' else 'tar.gz'}"
    if not wheels or archive not in names:
        return None
    return wheels[0], archive


def sums_table(text: str) -> dict[str, str]:
    table = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            table[parts[1].lstrip("*")] = parts[0].lower()
    return table


def fetch(url: str, attempts: int = 4) -> bytes:
    """A download, retried a few times; urllib.error.HTTPError for a missing file (404)."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as answer:
                data: bytes = answer.read()
                return data
        except urllib.error.HTTPError as err:
            if err.code == 404 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_checked(url: str, sha256: str, target: Path) -> Path:
    try:
        data = fetch(url)
    except (urllib.error.URLError, OSError) as err:
        raise Stop(f"{url} could not be downloaded ({err}). {DOWNLOAD_HINT}") from None
    if hashlib.sha256(data).hexdigest() != sha256:
        raise Stop(f"{url} did not match its SHA-256 hash: nothing from it was used. Try again later.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def use_source_folder(site: Path, root: Path) -> Path:
    """After installing the release's wheel: its compiled core moves next to the Python code in `root`'s
    python/nitro, the installed copy of that code goes, and a .pth file points Python at `root`'s python
    folder (as an editable install does), so NitroLearn finds its folder (settings, runs, crates) as when
    built from its code. Returns the core's new place."""
    installed = site / "nitro"
    cores = sorted(installed.glob("_core.*"))
    if not cores:
        raise Stop(f"the ready-made core is not in {installed}")
    target = root / "python" / "nitro"
    for old in target.glob("_core.*"):
        if old.suffix in (".pyd", ".so"):
            try:
                old.unlink()
            except OSError:
                raise Stop(
                    "NitroLearn is running (the dashboard or a training run): close it, then run the "
                    "installer again."
                ) from None
    placed = target / cores[0].name
    shutil.copy2(cores[0], placed)
    # The core's notices travel with it; none in the wheel leaves none that could describe another core.
    notices = installed / CORE_NOTICES
    if notices.is_file():
        shutil.copy2(notices, target / CORE_NOTICES)
    else:
        (target / CORE_NOTICES).unlink(missing_ok=True)
    shutil.rmtree(installed)
    (site / FOLDER_PTH).write_text(pth_line(site, root / "python"), encoding="ascii")
    return placed


def pth_line(site: Path, folder: Path) -> str:
    """A .pth line that puts `folder` on Python's path, in plain ASCII whatever the folder is called.
    Python 3.11 reads .pth files in the system's code page, not UTF-8, so a path line with a letter like é
    or Ł (a user name in the default folder) would point nowhere or stop Python from starting. This line is
    run instead: the path is relative to `sitedir` (the .pth file's folder, as Python names it while it
    reads the file) and any other letter is written as an escape."""
    try:
        relative = os.path.relpath(folder, site)
    except ValueError:  # another drive
        relative = str(folder)
    return (
        f"import os, sys; p = os.path.normpath(os.path.join(sitedir, {relative!a})); "
        "p in sys.path or sys.path.append(p)\n"
    )


def ascii_pth_files(site: Path) -> list[str]:
    """Rewrites the path lines of every .pth file in `site` that is not plain ASCII (an editable install
    writes its folder as UTF-8) as pth_line does; returns the names rewritten."""
    rewritten = []
    for pth in sorted(site.glob("*.pth")):
        try:
            data = pth.read_bytes()
            text = data.decode("utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        if data.isascii():
            continue
        lines = []
        for line in text.splitlines():
            plain = not line.strip() or line.startswith(("#", "import ", "import\t"))
            lines.append(line if plain else pth_line(site, site / line.rstrip()).rstrip("\n"))
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rewritten.append(pth.name)
    return rewritten


def unpack_programs(archive: Path, folder: Path) -> list[str]:
    """The release's programs (with prebuilt.json and their notices) from `archive` into `folder`, emptied
    first; only those names are taken from it."""
    wanted = {f"{name}.exe" if WINDOWS else name for name in PROGRAMS} | {"prebuilt.json", PROGRAMS_NOTICES}
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    taken = []
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                name = info.filename.rsplit("/", 1)[-1]
                if name in wanted and not info.is_dir():
                    (folder / name).write_bytes(zipped.read(info))
                    taken.append(name)
    else:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name.rsplit("/", 1)[-1]
                source = tar.extractfile(member) if member.isfile() else None
                if name in wanted and source is not None:
                    (folder / name).write_bytes(source.read())
                    (folder / name).chmod(0o755)
                    taken.append(name)
    return sorted(taken)


def core_matches_code(python: str = sys.executable) -> bool:
    """Whether the compiled core was built from the Rust code in this folder (nitro.build_info)."""
    return run([python, "-m", "nitro.build_info"], check=False) == 0


def bundles() -> ModuleType:
    """tools/bundles.py: the names of the release's own copies of what an install downloads."""
    module = _load(ROOT / "tools" / "bundles.py", "nitro_bundles_for_installer")
    if module is None:
        raise Stop("tools/bundles.py is missing: this folder is not a complete NitroLearn")
    return module


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Release:
    """This version's release files: SHA256SUMS.txt's table, and where the files are (a folder of them, else
    the release on GitHub)."""

    version: str
    url: str
    folder: Path | None
    hashes: Mapping[str, str]

    def get(self, name: str, target: Path) -> Path:
        """The file, checked against its hash: the folder's copy as it is, else downloaded to `target`."""
        wanted = self.hashes[name]
        local = self.folder / name if self.folder is not None else None
        if local is not None and local.is_file():
            if sha256_of(local) != wanted:
                raise Stop(
                    f"{local} does not match the release's SHA-256 hash for it: nothing from it was used."
                )
            return local
        return fetch_checked(self.url + name, wanted, target)


def release_folder(root: Path, given: Path | None) -> Path | None:
    """The folder of the release's files to install from: --bundles, else `release-files` in the folder."""
    if given is not None:
        return given.resolve()
    found = root / RELEASE_FOLDER
    return found if found.is_dir() else None


def of_version(hashes: Mapping[str, str], version: str) -> bool:
    """Whether SHA256SUMS.txt's table lists NitroLearn's own files of `version` (nitrolearn-<version>-...,
    nitrolearn-programs-<version>-..., ...), not another version's."""
    mark = re.compile(rf"^nitrolearn-(?:[a-z]+-)?{re.escape(version)}[-.]")
    return any(mark.match(name) for name in hashes)


def read_release(root: Path, base: str, folder: Path | None) -> Release | None:
    """This version's release files: their list from `folder` when it has one, else downloaded (only the
    list); None, and why, when there is none."""
    version = project_version(root)
    url = base.format(version=version)
    if folder is not None and (folder / SUMS).is_file():
        hashes = sums_table((folder / SUMS).read_text(encoding="utf-8"))
        if of_version(hashes, version):
            note(f"Using the release's files in {folder}")
            return Release(version, url, folder, hashes)
        note(f"The release files in {folder} are not version {version}'s: they are not used.")
    try:
        sums = fetch(url + SUMS).decode("utf-8")
    except (urllib.error.URLError, OSError) as err:
        note(
            f"No release files for version {version} could be fetched ({err}); a release that is not "
            "published yet has none. NitroLearn is built from its code instead, with its packages from PyPI "
            "and crates.io: about 30 minutes the first time."
        )
        return None
    return Release(version, url, None, sums_table(sums))


@dataclass(frozen=True)
class ReadyMade:
    """The release's ready-made core (wheel) and programs (archive) for this version and PC."""

    release: Release
    wheel: str
    archive: str


def find_ready_made(release: Release | None) -> ReadyMade | None:
    """The release's core and programs for this PC; None, and why, when there are none."""
    if release is None:
        return None
    platform = platform_tag()
    if platform is None:
        note("No ready-made files for this kind of PC: NitroLearn is built from its code.")
        return None
    names = release_files(
        "\n".join(f"{h}  {n}" for n, h in release.hashes.items()), release.version, platform
    )
    if names is None:
        note(
            f"The release of version {release.version} has no ready-made files for {platform}: building from "
            "the code."
        )
        return None
    return ReadyMade(release, names[0], names[1])


def from_wheels(wheels: Path | None) -> list[str]:
    """uv's options to take packages only from the release's copy of them (none without it)."""
    return [] if wheels is None else ["--no-index", "--find-links", str(wheels)]


def unpack_wheels(root: Path, release: Release | None) -> Path | None:
    """The release's copy of the Python packages for this PC (every one an install takes from PyPI, except
    PyTorch) in .tmp/wheels, or None when it has none (they then come from PyPI)."""
    platform = platform_tag()
    if release is None or platform is None:
        return None
    names = bundles()
    name = str(names.wheels_asset(release.version, platform))
    if name not in release.hashes:
        note("The release has no copy of the Python packages for this PC: they come from PyPI.")
        return None
    say("NitroLearn's Python packages, from the release's own copy")
    archive = release.get(name, root / ".tmp" / "release" / name)
    folder = root / ".tmp" / "wheels"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            base = info.filename
            plain = "/" not in base and "\\" not in base and not base.startswith(".")
            if plain and (base.endswith(".whl") or base in (names.REQUIREMENTS, names.WHEELS_LIST)):
                (folder / base).write_bytes(zipped.read(info))
    if not (folder / names.REQUIREMENTS).is_file():
        raise Stop(f"{name} holds no {names.REQUIREMENTS}: it is not one of NitroLearn's release files.")
    return folder


def install_wheels(uv: Path, wheels: Path) -> None:
    """Every package the release's copy holds, each checked against the hash its requirements.txt pins."""
    requirements = wheels / str(bundles().REQUIREMENTS)
    command: list[str | Path] = [uv, "pip", "install", "--python", sys.executable, *from_wheels(wheels)]
    run([*command, "--require-hashes", "-r", requirements], minutes=PACKAGES_MINUTES)


VENDOR_STAMP = ".from-release"  # in .tools/vendor: the hash of the crates archive it was unpacked from


def cargo_config(root: Path) -> Path:
    return root / ".tools" / "cargo" / "config.toml"


def ours(config: Path) -> bool:
    try:
        return config.read_text(encoding="utf-8").startswith("# Written by NitroLearn's installer")
    except OSError:
        return False


def use_vendored_crates(root: Path, release: Release | None) -> bool:
    """With Rust in the folder: the release's copy of every Rust crate in .tools/vendor, and cargo told to
    take crates only from there (.tools/cargo/config.toml), so no build needs crates.io. Without that copy,
    an earlier one and its setting are removed, and cargo uses crates.io."""
    names = bundles()
    config, vendor = cargo_config(root), root / ".tools" / str(names.VENDOR)
    name = str(names.crates_asset(release.version)) if release is not None else ""
    if release is None or name not in release.hashes:
        if ours(config):
            config.unlink()
            shutil.rmtree(vendor, ignore_errors=True)
            note("The release has no copy of the Rust crates: cargo downloads them from crates.io.")
        return False
    if config.exists() and not ours(config):
        note(f"{config} has settings of its own, so cargo keeps them and downloads crates from crates.io.")
        return False
    wanted = release.hashes[name]
    stamp = vendor / VENDOR_STAMP
    if not (stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == wanted):
        say("The Rust crates, from the release's own copy (no crates.io needed)")
        archive = release.get(name, root / ".tmp" / "release" / name)
        unpacked = root / ".tools" / "vendor.new"
        shutil.rmtree(unpacked, ignore_errors=True)
        prefix = f"{names.VENDOR}/"
        with tarfile.open(archive, "r:gz") as tar:
            members = [
                m for m in tar.getmembers() if m.name.startswith(prefix) or m.name == names.VENDOR_LIST
            ]
            tar.extractall(unpacked, members=members, filter="data")
        shutil.rmtree(vendor, ignore_errors=True)
        (unpacked / str(names.VENDOR)).rename(vendor)
        if (unpacked / str(names.VENDOR_LIST)).is_file():
            (unpacked / str(names.VENDOR_LIST)).replace(vendor / str(names.VENDOR_LIST))
        shutil.rmtree(unpacked, ignore_errors=True)
        stamp.write_text(wanted + "\n", encoding="utf-8")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(str(names.CARGO_CONFIG).format(vendor=names.VENDOR), encoding="utf-8")
    return True


def install_ready_made(
    uv: Path, root: Path, found: ReadyMade, extras: str, wheels: Path | None = None
) -> bool:
    """The release's core and programs; False (and why) when the core is for other code than this folder's."""
    say(f"NitroLearn {found.release.version}'s ready-made core and programs")
    downloads = root / ".tmp" / "release"
    wheel = found.release.get(found.wheel, downloads / found.wheel)
    archive = found.release.get(found.archive, downloads / found.archive)
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            *from_wheels(wheels),
            "--reinstall-package",
            "nitrolearn",
            f"nitrolearn[{extras}] @ {wheel.resolve().as_uri()}",
        ],
        minutes=PACKAGES_MINUTES,
    )
    use_source_folder(site_packages(), root)
    taken = unpack_programs(archive, root / ".tools" / "prebuilt")
    note(f"Ready-made programs: {', '.join(taken)}")
    shutil.rmtree(downloads, ignore_errors=True)
    if not core_matches_code():
        note("The ready-made core is for other code than this folder's (a changed or newer copy).")
        return False
    return True


def msvc_found() -> bool:
    """Windows: Microsoft's C++ build tools (Visual Studio or its Build Tools, with the C++ workload)."""
    vswhere = (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.is_file():
        return False
    done = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0 and bool(done.stdout.strip())


def compiler_problem() -> str | None:
    """What is missing to compile NitroLearn's Rust code (which also compiles some C), or None."""
    if WINDOWS:
        if msvc_found():
            return None
        return (
            "Building NitroLearn from its code needs Microsoft's C++ build tools, which this installer never "
            "installs (they change Windows itself). Install 'Build Tools for Visual Studio 2022' from "
            f"{BUILD_TOOLS_URL} (under 'Tools for Visual Studio'), tick 'Desktop development with C++', then "
            "run the installer again. Only building from the code and your own Rust rewards need them."
        )
    if any(shutil.which(name) for name in ("cc", "gcc", "clang")):
        return None
    return (
        "Building NitroLearn from its code needs a C compiler: install your system's build tools (on "
        "Ubuntu or Debian: sudo apt install build-essential), then run the installer again."
    )


def cargo_path() -> Path:
    return Path(os.environ["CARGO_HOME"]) / "bin" / ("cargo.exe" if WINDOWS else "cargo")


def rust_pin() -> tuple[str, list[str]]:
    """The Rust version rust-toolchain.toml pins (every build uses it) and the components it lists."""
    with (ROOT / "rust-toolchain.toml").open("rb") as file:
        toolchain = tomllib.load(file)["toolchain"]
    return str(toolchain["channel"]), [str(name) for name in toolchain.get("components", [])]


def rust_toolchains(rustup: Path) -> list[str]:
    """The toolchains installed in the folder's rustup (names like 1.97.0-x86_64-pc-windows-msvc)."""
    try:
        done = subprocess.run([str(rustup), "toolchain", "list"], capture_output=True, text=True, check=False)
    except OSError:
        return []
    # "no installed toolchains" when there are none
    return [
        line.split()[0] for line in done.stdout.splitlines() if line.strip() and not line.startswith("no ")
    ]


def ensure_rust(root: Path) -> None:
    """The pinned Rust (rust-toolchain.toml) in NitroLearn's folder (.tools; RUSTUP_HOME and CARGO_HOME point
    there, PATH is not changed), as rustup's default there too, so builds outside the folder use it as well.
    Another Rust left in the folder by an earlier install is removed: nothing builds with it."""
    step("rust", "start")
    _ensure_rust(root)
    step("rust", "done")


def _ensure_rust(root: Path) -> None:
    channel, components = rust_pin()
    with_components = [part for name in components for part in ("--component", name)]
    rustup = cargo_path().with_name("rustup.exe" if WINDOWS else "rustup")
    if rustup.exists():
        installed = rust_toolchains(rustup)
        if not any(name.startswith(f"{channel}-") for name in installed):
            say(f"Installing Rust {channel} into this folder")
            run(
                [
                    rustup,
                    "toolchain",
                    "install",
                    channel,
                    "--profile",
                    "minimal",
                    "--no-self-update",
                    *with_components,
                ],
                minutes=RUST_MINUTES,
            )
        else:
            note(f"Rust {channel} is in this folder.")
        run([rustup, "default", channel])
        for name in installed:
            if not name.startswith(f"{channel}-"):
                note(f"Removing {name} from this folder: NitroLearn builds with Rust {channel}.")
                run([rustup, "toolchain", "uninstall", name], check=False)
        return
    say(f"Installing Rust {channel} into this folder (about 1.5 GB)")
    triple = "x86_64-pc-windows-msvc" if WINDOWS else "x86_64-unknown-linux-gnu"
    url = RUSTUP.format(triple=triple, exe=".exe" if WINDOWS else "")
    try:
        expected = fetch(url + ".sha256").decode("utf-8").split()[0].lower()
    except (urllib.error.URLError, OSError) as err:
        raise Stop(f"Rust's installer could not be downloaded ({err}). {DOWNLOAD_HINT}") from None
    installer = fetch_checked(
        url, expected, root / ".tmp" / ("rustup-init.exe" if WINDOWS else "rustup-init")
    )
    installer.chmod(0o755)
    options = ["-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", channel]
    run([installer, *options, *with_components], minutes=RUST_MINUTES)
    installer.unlink()
    note(RUSTUP_PATH_NOTE)


def add_dev_components() -> None:
    """--dev: rustfmt and clippy in the folder's pinned Rust, for the code checks (tools/check.ps1,
    tools/check.sh). rust-toolchain.toml lists them, so this only adds them to a Rust installed without."""
    rustup = cargo_path().with_name("rustup.exe" if WINDOWS else "rustup")
    if not rustup.exists():
        note("Rust is not in this folder, so rustfmt and clippy were not added: the code checks need them.")
        return
    channel, _ = rust_pin()
    say(f"Adding rustfmt and clippy to Rust {channel} in this folder (for the code checks)")
    run([rustup, "component", "add", "--toolchain", channel, "rustfmt", "clippy"], minutes=RUST_MINUTES)


def folder_rust(root: Path) -> bool:
    """Whether Rust is in NitroLearn's folder (its rustup)."""
    return (root / ".tools" / "cargo" / "bin" / ("rustup.exe" if WINDOWS else "rustup")).is_file()


def remove_rust(root: Path) -> None:
    """Rust switched off on an install of the ready-made core: the folder's Rust goes (its rustup and cargo
    folders, inside NitroLearn's folder only)."""
    say("Removing Rust from this folder (switched off; this frees about 1.5 GB)")
    for name in ("rustup", "cargo", "vendor"):
        shutil.rmtree(root / ".tools" / name, ignore_errors=True)


# ---------- the windowless Python ----------

# A Windows program's subsystem (its PE header): 2 is a windowed program, 3 one that needs a console.
GUI_PROGRAM = 2
CONSOLE_PROGRAM = 3


def program_subsystem(path: Path) -> int | None:
    """The subsystem a Windows program (.exe) declares in its header; None when it is not one."""
    try:
        with path.open("rb") as file:
            head = file.read(4096)
    except OSError:
        return None
    if head[:2] != b"MZ" or len(head) < 0x40:
        return None
    pe = int.from_bytes(head[0x3C:0x40], "little")
    field = pe + 24 + 68  # the optional header follows the 4-byte signature and the 20-byte file header
    if head[pe : pe + 4] != b"PE\0\0" or len(head) < field + 2:
        return None
    return int.from_bytes(head[field : field + 2], "little")


def windowless_python(prefix: Path, base_prefix: Path) -> str:
    """Makes the environment's pythonw.exe a windowed program again: "fixed", "fine", or why it stays.

    uv makes .venv's python.exe and pythonw.exe as small console programs that start the base Python's
    python.exe, so a pythonw.exe started from the Start menu or the setup's Finish page opened an empty
    console window behind the dashboard. Python's own windowed launcher for environments (in its
    Lib/venv/scripts/nt) takes its place: it reads pyvenv.cfg's home and starts the base pythonw.exe there."""
    target = prefix / "Scripts" / "pythonw.exe"
    if program_subsystem(target) != CONSOLE_PROGRAM:
        return "fine"
    source = base_prefix / "Lib" / "venv" / "scripts" / "nt" / "pythonw.exe"
    if program_subsystem(source) != GUI_PROGRAM:
        return f"{source} is not Python's windowed launcher"
    new, old = target.with_name("pythonw.exe.new"), target.with_name("pythonw.exe.old")
    try:
        shutil.copyfile(source, new)
        old.unlink(missing_ok=True)
        target.replace(old)  # a program that is running (an open dashboard) can be renamed, not replaced
    except OSError as err:
        new.unlink(missing_ok=True)
        return f"pythonw.exe could not be changed ({err})"
    try:
        new.replace(target)
    except OSError as err:
        old.replace(target)
        return f"pythonw.exe could not be changed ({err})"
    with contextlib.suppress(OSError):
        old.unlink()  # still running: deleted by the next install
    return "fixed"


def install_from_code(
    uv: Path, root: Path, extras: str, release: Release | None = None, wheels: Path | None = None
) -> None:
    problem = compiler_problem()
    if problem is not None:
        raise Stop(problem)
    say(
        "Building NitroLearn from its code: about 30 minutes the first time (Rust is set up in this folder, "
        "then it compiles NitroLearn's core), a few minutes after that"
    )
    ensure_rust(root)
    use_vendored_crates(root, release)
    (site_packages() / FOLDER_PTH).unlink(missing_ok=True)  # left by an install of the ready-made core
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            *from_wheels(wheels),
            "--editable",
            f"{root}[{extras}]",
        ],
        minutes=BUILD_MINUTES,
    )
    ascii_pth_files(site_packages())  # the editable install's .pth names the folder, maybe with é or Ł
    write_core_notices(root)


def write_core_notices(root: Path) -> None:
    """After a build from the code: the core's third-party notices beside it, from the build's own crates."""
    tool = root / "python" / "nitro" / "notices.py"
    out = root / "python" / "nitro" / CORE_NOTICES
    if run([sys.executable, tool, "core", "--out", out], check=False) != 0:
        note(
            f"The core's licence notices could not be written. Write them later with: python {tool} core "
            f"--out {out}"
        )


OWN_REWARDS = Path("my-settings") / "rust-files.txt"  # the user's own Rust rewards (nitro.rebuild's list)


def rebuild_own_rewards(rust: bool, nitro: Callable[..., int] | None = None) -> bool:
    """The release's ready-made core has none of the user's own Rust rewards: with Rust in the folder and
    the build tools on the PC, they are built in again (`nitro rebuild`); otherwise it says how. True when
    they were built in."""
    if not rust:
        note(
            "This is the release's core, without your own Rust rewards. To build them in again, run the "
            'installer again with its part "Tools to build your own Rust rewards" ticked (or with -Rust); '
            "it then builds them in by itself."
        )
        return False
    problem = compiler_problem()
    if problem is not None:
        note(f"Your own Rust rewards were not built in again. {problem}")
        return False
    say("Building your own Rust rewards into NitroLearn again (a few minutes)")
    command = [sys.executable, "-m", "nitro", "rebuild"]
    code = nitro("rebuild") if nitro is not None else run(command, check=False, minutes=BUILD_MINUTES)
    if code != 0:
        note(
            "Your own Rust rewards could not be built in (see above). NitroLearn works without them: fix "
            "what is named above, then press Build on the dashboard's Resources page."
        )
        return False
    return True


def keep_torch(uv: Path, variant: str) -> None:
    """Installing NitroLearn must not have swapped PyTorch for another build; if it did, put it back."""
    if installed_version("torch") != f"{TORCH_VERSION}+{variant}":
        note("PyTorch was changed by another package: installing the right build again.")
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--reinstall-package",
                "torch",
                f"torch=={TORCH_VERSION}+{variant}",
                "--index-url",
                TORCH_INDEX + variant,
            ],
            minutes=TORCH_MINUTES,
        )


# ---------- what to install: the optional parts ----------

CHOICES_RECORD = Path(".tools") / "install_choices.json"  # the parts chosen at the last install
SHORTCUTS_RECORD = Path(".tools") / "shortcuts.json"  # the shortcuts tools/app_shortcuts.ps1 made
# The optional parts in the menu's order (Windows-only ones are left out elsewhere).
PARTS = ("gpu", "rlbot", "start_menu", "desktop", "rust", "dev", "recordings")
WINDOWS_ONLY = frozenset({"rlbot", "start_menu", "desktop"})
RLBOT_SITE = "https://rlbot.org"
RECORDINGS_MINUTES = 30


@dataclass(frozen=True)
class Found:
    """What the installer finds before it downloads anything: what the menu starts from."""

    windows: bool
    person: bool  # someone at the console can answer
    card: str  # the NVIDIA card PyTorch will use ("" when there is none)
    no_card: str  # why PyTorch is for the processor ("" when there is a card)
    ready_made: bool  # the release has NitroLearn's ready-made core for this version and PC
    from_source: bool  # --from-source
    first: bool  # no install has recorded its choices or made shortcuts here yet
    has: Mapping[str, bool]  # the parts here already ("torch": any PyTorch; "gpu": a graphics-card one)
    rlbot4: str = ""  # an RLBot v4 install found ("" when none)


def shown_parts(found: Found) -> list[str]:
    """The parts the menu lists, numbered from 1 in this order."""
    return [part for part in PARTS if found.windows or part not in WINDOWS_ONLY]


def rust_required(found: Found, wishes: Mapping[str, bool]) -> bool:
    """NitroLearn is built from its code (no ready-made core, --from-source, or developer tools): Rust is
    needed then, not optional."""
    return not found.ready_made or found.from_source or wishes.get("dev", False)


def default_wishes(found: Found, recorded: Mapping[str, object]) -> dict[str, bool]:
    """What the menu starts with: the parts installed already, and those chosen at the last install
    (.tools/install_choices.json) that are not there yet (an install that stopped before making them); for
    the rest, the first install's defaults with a person at the console (the Start menu shortcut and RLBot
    v5), and graphics-card PyTorch when there is a card. Without a person nothing outside the folder is made
    unless a switch or an earlier choice asks for it."""
    has = found.has
    offer = found.first and found.person

    def kept(part: str) -> bool:
        return has.get(part, False) or recorded.get(part) is True

    return {
        "gpu": bool(found.card) and (has.get("gpu", False) if has.get("torch") else True),
        "rlbot": kept("rlbot") or offer,
        "start_menu": kept("start_menu") or offer,
        "desktop": kept("desktop"),
        "rust": kept("rust"),
        "dev": recorded.get("dev") is True,
        "recordings": kept("recordings"),
    }


def effective(found: Found, wishes: Mapping[str, bool]) -> dict[str, bool]:
    """The parts that will be there: the wishes, with what cannot be otherwise (no card: no graphics-card
    PyTorch; RLBot v5 installed stays; the desktop shortcut comes with the Start menu one; Rust when
    NitroLearn is built from its code; Windows-only parts only on Windows)."""
    out = {part: bool(wishes.get(part, False)) for part in PARTS}
    out["gpu"] = out["gpu"] and bool(found.card)
    out["rlbot"] = out["rlbot"] or found.has.get("rlbot", False)
    out["start_menu"] = out["start_menu"] or out["desktop"]
    out["rust"] = out["rust"] or rust_required(found, out)
    if not found.windows:
        out |= dict.fromkeys(WINDOWS_ONLY, False)
    return out


def locked(found: Found, wishes: Mapping[str, bool], part: str) -> str:
    """Why `part` cannot be switched now ("" when it can)."""
    if part == "gpu" and not found.card:
        return f"{found.no_card}, so PyTorch for the processor is installed."
    if part == "rlbot" and found.has.get("rlbot"):
        return "RLBot v5 is installed. It is its own program: remove it from Windows' Apps list."
    if part == "rust" and rust_required(found, wishes):
        return "Rust is needed: NitroLearn is built from its code on this PC."
    return ""


def toggle(found: Found, wishes: Mapping[str, bool], part: str) -> tuple[dict[str, bool], str]:
    """The wishes with `part` switched, and "" (or why it cannot be switched, the wishes unchanged)."""
    new = dict(wishes)
    why = locked(found, new, part)
    if why:
        return new, why
    new[part] = not effective(found, new)[part]
    if part == "start_menu" and not new[part]:
        new["desktop"] = False  # the desktop shortcut comes with the Start menu one
    if part == "desktop" and new[part]:
        new["start_menu"] = True
    return new, ""


def switched(args: argparse.Namespace) -> dict[str, bool]:
    """The parts set on the command line; with any of them, the menu is not shown. --parts sets every part
    (those it names on, the rest off), as the setup wizard's choices do."""
    if getattr(args, "parts", None) is not None:
        named = set(parts_list(args.parts))
        return {part: part in named for part in PARTS}
    out = {}
    if args.torch == "cpu":
        out["gpu"] = False
    elif args.torch != "auto":
        out["gpu"] = True
    if args.rlbot:
        out["rlbot"] = True
    if args.no_rlbot:
        out["rlbot"] = False
    if args.no_shortcuts:
        out |= {"start_menu": False, "desktop": False}
    if args.desktop:
        out |= {"start_menu": True, "desktop": True}
    if getattr(args, "start_menu", False):
        out |= {"start_menu": True, "desktop": False}
    if getattr(args, "gpu", False):
        out["gpu"] = True
    for part in ("rust", "dev", "recordings"):
        if getattr(args, part):
            out[part] = True
    return out


def part_text(found: Found, wishes: Mapping[str, bool], part: str) -> str:
    """A part's line in the menu, after its number."""
    if part == "gpu":
        if not found.card:
            return (
                f"PyTorch for the graphics card: {found.no_card}, so PyTorch for the processor (about 1.5 GB "
                "on disk)"
            )
        return f"PyTorch for the graphics card, {found.card} (9 GB on disk; off: for the processor, 1.5 GB)"
    if part == "rlbot":
        if found.has.get("rlbot"):
            return "RLBot v5, to play your bots in Rocket League (installed)"
        text = "RLBot v5, to play your bots in Rocket League (its own installer opens; a few MB)"
        if found.rlbot4:
            text += (
                f"\n         (RLBot v4 is at {found.rlbot4}: NitroLearn's bots need v5, which installs next "
                "to it)"
            )
        return text
    if part == "start_menu":
        return "Start menu shortcut"
    if part == "desktop":
        return "Desktop shortcut (with the Start menu one)"
    if part == "rust":
        if rust_required(found, wishes):
            return "Rust in NitroLearn's folder: required, as NitroLearn is built from its code here (1.5 GB)"
        return "Tools to build your own Rust rewards (Rust in NitroLearn's folder, about 1.5 GB)"
    if part == "dev":
        return "Developer tools: tests, rustfmt and clippy (builds NitroLearn from its code, with Rust)"
    return "RocketSim's game recordings now (119 MB; else downloaded the first time a feature needs them)"


def menu_text(found: Found, wishes: Mapping[str, bool]) -> str:
    on = effective(found, wishes)
    core = "built from its code" if rust_required(found, on) else "ready-made"
    lines = [
        "",
        "What to install",
        "  Always installed: Python and NitroLearn's private environment, NitroLearn's core and programs",
        f"  ({core}), and the arena files.",
        "  PyTorch comes from PyTorch's site and the arena files from RLGym; the rest from NitroLearn's",
        "  release (or from its original site).",
    ]
    if not found.first:
        lines.append("  Ticked parts are here now: switch one off to remove it, or on to add it.")
    lines.append("")
    for number, part in enumerate(shown_parts(found), 1):
        lines.append(f"  [{'x' if on[part] else ' '}] {number}  {part_text(found, on, part)}")
    lines += ["", "Type a number to switch it on/off (or several: 3 4), Enter to install."]
    return "\n".join(lines)


def read_numbers(text: str, count: int) -> list[int] | None:
    """The part numbers typed ("3", "3 4", "3,4"); [] for Enter alone; None for anything else."""
    words = text.replace(",", " ").split()
    if not all(word.isdigit() and 1 <= int(word) <= count for word in words):
        return None
    return [int(word) for word in words]


def choose_parts(
    found: Found,
    wishes: Mapping[str, bool],
    ask: Callable[[str], str],
    show: Callable[[str], None] = print,
) -> dict[str, bool]:
    """The menu: shown again after each change until Enter alone (or the end of the input)."""
    current = dict(wishes)
    parts = shown_parts(found)
    while True:
        show(menu_text(found, current))
        try:
            answer = ask("> ")
        except EOFError:
            answer = ""
        numbers = read_numbers(answer, len(parts))
        if numbers is None:
            show(f"Type a number from 1 to {len(parts)}, or press Enter to install.")
            continue
        if not numbers:
            return current
        for number in numbers:
            current, why = toggle(found, current, parts[number - 1])
            if why:
                show(f"{number}: {why}")


def decide(
    found: Found,
    recorded: Mapping[str, object],
    switches: Mapping[str, bool],
    ask: Callable[[str], str],
    show: Callable[[str], None] = print,
) -> dict[str, bool]:
    """The parts to have after this install: the menu with a person and no switches; else the current state
    (first install: the defaults) with the switches."""
    wishes = default_wishes(found, recorded) | dict(switches)
    if found.person and not switches:
        wishes = choose_parts(found, wishes, ask, show)
    return effective(found, wishes)


def read_record(root: Path) -> dict[str, object]:
    try:
        data = json.loads((root / CHOICES_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_record(root: Path, choices: Mapping[str, bool]) -> None:
    file = root / CHOICES_RECORD
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(dict(choices), indent=1) + "\n", encoding="utf-8")


def nitro_module(name: str) -> ModuleType | None:
    """One of NitroLearn's standard-library-only modules (rlbot_find, recordings, folder_env), loaded from
    this folder's code before NitroLearn is installed."""
    return _load(ROOT / "python" / "nitro" / f"{name}.py", f"nitro_{name}_for_installer")


def _load(path: Path, name: str) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    try:
        spec.loader.exec_module(module)
    except (OSError, ImportError):
        sys.modules.pop(name, None)
        return None
    return module


def shortcuts_made(root: Path) -> tuple[bool, bool]:
    """(Start menu, desktop) shortcuts that tools/app_shortcuts.ps1 recorded (the desktop one is the second)
    and that are still there: one deleted since counts as missing, so the install makes it again."""
    try:
        made = json.loads((root / SHORTCUTS_RECORD).read_text(encoding="utf-8")).get("shortcuts", [])
    except (OSError, ValueError, AttributeError):
        return False, False
    links = made if isinstance(made, list) else []
    there = [isinstance(link, str) and Path(link).is_file() for link in links[:2]]
    there += [False] * (2 - len(there))
    return there[0], there[1]


def look_around(
    root: Path, windows: bool, person: bool, card: str, no_card: str, ready: bool, from_source: bool
) -> Found:
    """What is installed here already, for the menu."""
    torch = installed_version("torch") or ""
    start, desktop = shortcuts_made(root)
    has = {
        "torch": bool(torch),
        "gpu": bool(torch) and not torch.endswith("+cpu"),
        "start_menu": start,
        "desktop": desktop,
        "rust": folder_rust(root),
    }
    recordings = nitro_module("recordings")
    has["recordings"] = bool(recordings and recordings.downloaded(root / "rocketsim_recordings"))
    older = ""
    if windows:
        finder = nitro_module("rlbot_find")
        local = os.environ.get("LOCALAPPDATA")
        local_path = Path(local) if local else None
        if finder is not None:
            has["rlbot"] = finder.find_v5(local_path) is not None
            older = str(finder.find_older(local_path) or "")
    first = not (root / CHOICES_RECORD).is_file() and not (root / SHORTCUTS_RECORD).is_file()
    return Found(windows, person, card, no_card, ready, from_source, first, has, older)


# ---------- room on the disk ----------


def space_needed_gb(found: Found, choices: Mapping[str, bool], torch: str, wanted: str) -> float:
    """About how much more room this install takes (downloads kept in .cache included): PyTorch when it is
    not the build wanted yet (about 9 GB for the graphics card, 1.5 GB for the processor), NitroLearn and
    its packages on a first install, Rust and the recordings when they are added."""
    needed = 0.0 if found.has.get("torch") else 0.5
    if torch != wanted:
        needed += 9.0 if not wanted.endswith("+cpu") else 1.5
    if choices.get("rust") and not found.has.get("rust"):
        needed += 1.5
    if choices.get("recordings") and not found.has.get("recordings"):
        needed += 0.15
    return needed


def check_space(root: Path, needed: float, free: float | None = None) -> None:
    """Stops before anything is downloaded when the disk cannot hold what this install adds."""
    free = free_gb(root) if free is None else free
    if free is None or needed <= 0 or free >= needed + LOW_DISK_GB:
        return
    disk, more = root.anchor or root, max(1, round(needed))
    raise Stop(
        f"Not enough free disk space: this install needs about {more} GB more on {disk}, which has "
        f"{free:.1f} GB free. Free up space, or install NitroLearn on a drive with more room, then run the "
        "installer again."
    )


# ---------- the whole setup ----------


def parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NitroLearn's installer steps (run by install.ps1 / install.sh)"
    )
    parser.add_argument("--uv", type=Path, required=True, help="the uv program in .tools/uv")
    parser.add_argument("--torch", choices=("auto", *TORCH_VARIANTS), default="auto", help="PyTorch build")
    parser.add_argument("--cpu", action="store_true", help="PyTorch for the processor (as --torch cpu)")
    parser.add_argument("--rust", action="store_true", help="also put Rust in the folder (own Rust rewards)")
    parser.add_argument("--from-source", action="store_true", help="build NitroLearn from its code")
    parser.add_argument("--dev", action="store_true", help="developer install (tests and code checkers)")
    parser.add_argument("--recordings", action="store_true", help="download RocketSim's recordings now")
    parser.add_argument("--no-launch", action="store_true", help="do not open the dashboard at the end")
    parser.add_argument(
        "--rlbot", action="store_true", help="also install RLBot v5 with its own installer (Windows)"
    )
    parser.add_argument("--no-rlbot", action="store_true", help="do not install RLBot v5 (Windows)")
    parser.add_argument("--desktop", action="store_true", help="Start menu and desktop shortcuts (Windows)")
    parser.add_argument("--start-menu", action="store_true", help="a Start menu shortcut only (Windows)")
    parser.add_argument("--no-shortcuts", action="store_true", help="no shortcuts; removes earlier ones")
    parser.add_argument(
        "--gpu", action="store_true", help="PyTorch for the graphics card (when there is one)"
    )
    parser.add_argument(
        "--parts",
        help=f"exactly these optional parts, comma-separated, the rest off ({','.join(PARTS)}; none for "
        "none); the setup wizard passes its choices so",
    )
    parser.add_argument(
        "--bundles",
        type=Path,
        help=f"a folder of the release's files to install from (default: {RELEASE_FOLDER})",
    )
    parser.add_argument("--progress", action="store_true", help="a line per step, for the setup wizard")
    args = parser.parse_args(argv)
    if args.cpu:
        args.torch = "cpu"
    if args.parts is not None:
        unknown = sorted(set(parts_list(args.parts)) - set(PARTS))
        if unknown:
            parser.error(f"--parts: no such part: {', '.join(unknown)} (the parts: {', '.join(PARTS)})")
    return args


def parts_list(text: str) -> list[str]:
    """The parts named in --parts ("none" for none)."""
    return [part.strip() for part in text.split(",") if part.strip() not in ("", "none")]


def torch_choice(root: Path, asked: str) -> str | None:
    """A PyTorch build chosen with --torch is remembered, so running the installer again keeps it."""
    remembered = root / ".tools" / "torch-choice.txt"
    if asked != "auto":
        remembered.parent.mkdir(parents=True, exist_ok=True)
        remembered.write_text(asked + "\n", encoding="utf-8")
        return asked
    try:
        text = remembered.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text if text in TORCH_VARIANTS else None


UPDATE_BACKUPS = Path("update") / "replaced"  # in .tmp: an update's copies of files the user had changed


def clear_temporary(folder: Path) -> None:
    """Empties the temporary folder, except the copies an update (nitro.update) kept of changed files."""
    keep = folder / UPDATE_BACKUPS
    found = list(folder.iterdir()) if folder.is_dir() else []
    while found:
        path = found.pop()
        if path in keep.parents:
            found += [inner for inner in path.iterdir() if inner != keep]
        elif path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                path.unlink()
    folder.mkdir(exist_ok=True)


CHECK_FAILED = 2  # main's code when NitroLearn is installed but `nitro doctor` found problems


@dataclass(frozen=True)
class Done:
    code: int  # nitro doctor's
    found: Found
    choices: Mapping[str, bool]


def setup(
    args: argparse.Namespace,
    root: Path = ROOT,
    releases: str = RELEASES,
    facts: Callable[[], NvidiaFacts | None] = nvidia_facts,
    person: bool = False,
    ask: Callable[[str], str] = input,
) -> Done:
    PROGRESS[0] = bool(getattr(args, "progress", False))
    step("pc", "start")
    say(f"Python {sys.version.split()[0]} in {Path(sys.prefix)}")
    if WINDOWS:
        windowless = windowless_python(Path(sys.prefix), Path(sys.base_prefix))
        if windowless not in ("fine", "fixed"):
            note(f"The dashboard may open with an empty console window behind it: {windowless}.")
    say("Looking at this PC")
    nvidia = facts()
    auto, notes = choose_torch(nvidia)
    for text in notes:
        note(text)
    remembered = torch_choice(root, args.torch)
    gpu_variant = remembered if remembered not in (None, "cpu") else auto
    card = ""
    if gpu_variant != "cpu":
        names = [item.name for item in nvidia.cards if item.name] if nvidia else []
        card = names[0] if names else "an NVIDIA card"
    no_card = "no NVIDIA card was found" if nvidia is None else "this NVIDIA card is too old for PyTorch"
    folder = release_folder(root, getattr(args, "bundles", None))
    if folder is not None:
        os.environ[RELEASE_FILES_ENV] = str(folder)
    release = read_release(root, releases, folder)
    ready_made = None
    if args.from_source:
        note("Building NitroLearn from its code (--from-source).")
    else:
        ready_made = find_ready_made(release)
    found = look_around(root, WINDOWS, person, card, no_card, ready_made is not None, args.from_source)
    choices = decide(found, read_record(root), switched(args), ask)

    variant = gpu_variant if choices["gpu"] else "cpu"
    check_space(
        root, space_needed_gb(found, choices, installed_version("torch") or "", f"{TORCH_VERSION}+{variant}")
    )
    step("pc", "done")

    step("torch", "start")
    if remembered is not None and (remembered == "cpu") == choices["gpu"]:
        (root / ".tools" / "torch-choice.txt").unlink(missing_ok=True)  # switched in the menu since
    elif remembered not in (None, "cpu") and args.torch == "auto":
        note(f"Using the PyTorch build chosen earlier with --torch: {variant}")
    install_torch(args.uv, variant)
    step("torch", "done")

    step("core", "start")
    extras = "dev" if choices["dev"] else EXTRAS
    # The release's copy of the packages (not for a developer install: its tools come from PyPI).
    wheels = None if choices["dev"] else unpack_wheels(root, release)
    if wheels is not None:
        install_wheels(args.uv, wheels)
    ready = False
    if ready_made is not None and not choices["dev"]:
        ready = install_ready_made(args.uv, root, ready_made, extras, wheels)
    if not ready:
        install_from_code(args.uv, root, extras, release, wheels)
    elif choices["rust"]:
        problem = compiler_problem()
        ensure_rust(root)
        use_vendored_crates(root, release)
        if problem is not None:
            note(problem)
    elif found.has.get("rust"):
        remove_rust(root)
    if choices["dev"]:
        add_dev_components()
    keep_torch(args.uv, variant)
    # maturin: `nitro rebuild` (own Rust rewards) builds the core in place with it (in the release's copy).
    if wheels is None:
        run(
            [args.uv, "pip", "install", "--python", sys.executable, "maturin>=1.15,<2"],
            minutes=PACKAGES_MINUTES,
        )
    if sys.platform.startswith("linux") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # The dashboard's window on a Linux desktop: Qt is GPL/LGPL software, so it always comes from PyPI.
        qt = [args.uv, "pip", "install", "--python", sys.executable, "pywebview[qt]>=5.3"]
        if run(qt, check=wheels is None, minutes=PACKAGES_MINUTES) != 0:
            note(
                "The dashboard's window (pywebview's Qt part) could not be installed from PyPI: run the "
                "installer again once online."
            )
    if ready and (root / OWN_REWARDS).is_file():
        rebuild_own_rewards(choices["rust"])
    step("core", "done")

    step("meshes", "start")
    say("Arena meshes")
    if run([sys.executable, "-m", "nitro", "fetch-meshes"], check=False, minutes=MESHES_MINUTES) != 0:
        later = ".\\nitro fetch-meshes" if WINDOWS else "./nitro.sh fetch-meshes"
        note(f"The meshes could not be fetched now: run {later} later (see the message above).")
        step("meshes", "fail")
    else:
        step("meshes", "done")
    if choices["recordings"] and not found.has.get("recordings"):
        step("recordings", "start")
        say("RocketSim's game recordings")
        fetched = [sys.executable, "-m", "nitro", "fetch-recordings"]
        if run(fetched, check=False, minutes=RECORDINGS_MINUTES) != 0:
            note("They could not be fetched now: they download the first time a feature needs them.")
            step("recordings", "fail")
        else:
            step("recordings", "done")
    elif not choices["recordings"] and found.has.get("recordings"):
        recordings = nitro_module("recordings")
        if recordings is not None:
            recordings.remove_downloaded(root / "rocketsim_recordings")
            note("RocketSim's game recordings were removed (switched off): they download again when needed.")

    clear_temporary(root / ".tmp")
    step("check", "start")
    say('Checking everything with "nitro doctor"')
    code = run([sys.executable, "-m", "nitro", "doctor"], check=False)
    step("check", "done" if code == 0 else "fail")
    return Done(code, found, choices)


# ---------- after a successful install ----------


def interactive() -> bool:
    """Whether a person can answer here: input and output are a console window, not a file or nothing
    (`Install NitroLearn.cmd < nul > log`). On Windows the NUL device passes for a terminal, so the console
    itself is asked."""
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        mode = ctypes.c_uint32()
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    return True


def powershell(root: Path, script: str, *args: str) -> int:
    """Runs one of NitroLearn's PowerShell scripts (`script`, relative to `root`)."""
    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / script), *args]
    try:
        return subprocess.run(command, cwd=root, check=False).returncode
    except OSError:
        return 1


def nitro_command(*args: str) -> int:
    """`nitro <args>` with this folder's Python (NitroLearn is installed by now), its output shown."""
    try:
        return subprocess.run([sys.executable, "-m", "nitro", *args], check=False).returncode
    except OSError:
        return 1


def after_install(
    root: Path,
    no_launch: bool,
    done: Done,
    rlbot_switch: bool = False,
    run_script: Callable[..., int] = powershell,
    nitro: Callable[..., int] = nitro_command,
) -> None:
    """Windows, once NitroLearn is installed (also when the final check found something to fix): the
    shortcuts as chosen (tools/app_shortcuts.ps1 makes or removes them, only when that changed), RLBot v5
    with its own installer when chosen and not installed (`nitro install-rlbot`), then, with a person at
    the console and unless `no_launch`, the dashboard opens as its in-folder shortcut does. Off Windows,
    --rlbot only says where to get RLBot."""
    found, choices = done.found, done.choices
    if not found.windows:
        if rlbot_switch:
            note(f"RLBot v5's installer is for Windows: see {RLBOT_SITE} for this system.")
        return
    wanted = (choices["start_menu"], choices["desktop"])
    step("shortcuts", "start")
    shortcuts = "done"
    if wanted != (found.has.get("start_menu", False), found.has.get("desktop", False)):
        made = ("-Desktop",) if choices["desktop"] else () if choices["start_menu"] else ("-Remove",)
        if run_script(root, "tools/app_shortcuts.ps1", *made) != 0:
            shortcuts = "fail"
            note(
                "The shortcuts could not be changed (see above); the Shortcuts folder opens the dashboard "
                "too."
            )
    step("shortcuts", shortcuts)
    if choices["rlbot"] and not found.has.get("rlbot"):
        step("rlbot", "start")
        say("RLBot v5")
        if nitro("install-rlbot") != 0:
            step("rlbot", "fail")
            note(f"RLBot is not installed: run .\\nitro install-rlbot later, or get it from {RLBOT_SITE}.")
        else:
            step("rlbot", "done")
    if found.person and not no_launch:
        print("\nOpening the dashboard.", flush=True)
        if run_script(root, "Shortcuts/shortcuts.ps1", "dashboard") != 0:
            note('The dashboard did not open: double-click "Open the dashboard" in the Shortcuts folder.')


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        # Output to a file is in the system's code page: a folder name it has no letter for must not stop
        # the install.
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
    os.environ.update(folder_env(ROOT) if env is None else env)
    module = nitro_module("folder_env")
    if module is not None:
        module.native_safe(os.environ)  # a cache path NVIDIA's driver can read, for the final check
    os.chdir(ROOT)
    try:
        try:
            done = setup(args, releases=os.environ.get("NITRO_RELEASES") or RELEASES, person=interactive())
        except OSError as err:
            if not out_of_space(err):
                raise
            raise Stop(disk_full(ROOT)) from None
    except Stop as err:
        print(f"\nSETUP STOPPED: {err}", flush=True)
        return 1
    write_record(ROOT, done.choices)
    start = (
        'double-click "Open the dashboard" in the Shortcuts folder (or run .\\nitro dashboard)'
        if WINDOWS
        else "run ./nitro.sh dashboard (./nitro.sh --help lists every command)"
    )
    if done.code == 0:
        print(f"\nNitroLearn is installed. To start: {start}.", flush=True)
        after_install(ROOT, args.no_launch, done, args.rlbot)
        return 0
    # The shortcuts and RLBot as chosen all the same; only the dashboard does not open by itself.
    after_install(ROOT, True, done, args.rlbot)
    print(
        '\nNitroLearn is installed, but its final check ("nitro doctor") found problems: follow its "fix:" '
        f"lines above. Then, to start: {start}.",
        flush=True,
    )
    return CHECK_FAILED


if __name__ == "__main__":
    sys.exit(main())
