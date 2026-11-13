"""`nitro install-rlbot`: RLBot v5 (needed to play bots in Rocket League), with RLBot's own installer.

RLBot is a separate program with its own licence: NitroLearn never ships or hosts any of its files. This
downloads RLBot's official installer (rlbot-v5-installer.msi, from the release tagged "installer" of RLBot's
launcher on GitHub, where RLBot publishes the RLBot v5 installer) into NitroLearn's .tmp folder, checks it is
a whole Windows Installer file for RLBot v5, and opens it, so the person goes through RLBot's own installer
windows. RLBot puts itself in %LOCALAPPDATA%\\RLBot5 and is removed from Windows' Apps list, never by deleting
NitroLearn. Only RLBot v5 counts as installed (nitro.rlbot_find): NitroLearn's bots do not run in RLBot v4,
which can stay installed next to it. Windows only, as RLBot's installer is.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nitro import rlbot_find
from nitro.doctor import PROJECT, find_rlbot

SITE = "https://rlbot.org"
# RLBot's launcher repository keeps its installer in the release tagged "installer" (replaced in place on
# each new build, so it is checked against the hash GitHub lists for it rather than a pinned one).
RELEASE_API = "https://api.github.com/repos/RLBot/launcher/releases/tags/installer"
OFFICIAL = "https://github.com/RLBot/launcher/releases/download/"
V5_RELEASE = OFFICIAL + "installer/"  # RLBot v5's installer (RLBot's launcher is v5's)
FALLBACK_URL = V5_RELEASE + "rlbot-v5-installer.msi"
# RLBot's v5 installer names its product "RLBot v5" (its summary and property table say so as plain text).
V5_MARK = b"RLBot v5"
MSI_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")  # every .msi is an OLE compound file
MIN_BYTES = 100_000
MAX_BYTES = 200_000_000
DOWNLOAD_MINUTES = 10
TMP = PROJECT / ".tmp" / "rlbot"  # the download waits here while RLBot's installer runs
# Said whenever a download fails or stalls; the same words as the installer's (tools/install_steps.py).
DOWNLOAD_HINT = (
    "Check the internet connection. Antivirus web shields (HTTPS scanning) and company proxies often hold "
    "back or block downloads like these: allow this folder in your antivirus (or pause its web shield for "
    "the install), then run the installer again. What was already downloaded is kept."
)
NOT_WINDOWS = (
    f"RLBot v5's installer is for Windows. To play bots in Rocket League, see {SITE} for how to run RLBot "
    "on this system."
)
REMOVING = "RLBot is its own program: remove it from Windows' Apps list (deleting NitroLearn leaves it)."
# msiexec's answers that are not plain failures.
MSI_DONE = (0, 3010)  # 3010: installed, Windows wants a restart
MSI_CANCELLED = 1602
MSI_ALREADY = 1638  # another version is installed already


class RLBotInstallError(RuntimeError):
    """The installer could not be fetched, or is not what was expected; the message says what to do."""


@dataclass(frozen=True)
class Installer:
    url: str
    sha256: str | None = None  # as GitHub lists it for the release file
    size: int | None = None


def pick_installer(release: Mapping[str, Any]) -> Installer | None:
    """The .msi file of RLBot's v5 installer release (GitHub's release JSON), from RLBot's own downloads
    only."""
    for asset in release.get("assets") or []:
        if not isinstance(asset, Mapping):
            continue
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if not name.lower().endswith(".msi") or not url.startswith(V5_RELEASE):
            continue
        digest = str(asset.get("digest") or "")
        sha256 = digest.removeprefix("sha256:").lower() if digest.startswith("sha256:") else None
        size = asset.get("size")
        return Installer(url, sha256, size if isinstance(size, int) and size > 0 else None)
    return None


def read_release(url: str = RELEASE_API) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "NitroLearn-installer"}
    )
    with urllib.request.urlopen(request, timeout=30) as answer:
        data: Mapping[str, Any] = json.load(answer)
        return data


def find_installer(release: Callable[[], Mapping[str, Any]] = read_release) -> Installer:
    """The newest installer as RLBot's release lists it; its fixed download link when GitHub's list cannot be
    read (its API allows few requests an hour without an account)."""
    try:
        found = pick_installer(release())
    except (urllib.error.URLError, OSError, ValueError):
        found = None
    return found or Installer(FALLBACK_URL)


def download(
    url: str, minutes: float = DOWNLOAD_MINUTES, opener: Callable[..., Any] = urllib.request.urlopen
) -> bytes:
    """The file's bytes; past `minutes` (a stalled download) or on a network error it stops and says what
    usually holds downloads back."""
    deadline = time.monotonic() + minutes * 60
    parts: list[bytes] = []
    received = 0
    try:
        with opener(url, timeout=60) as answer:
            while chunk := answer.read(1 << 16):
                parts.append(chunk)
                received += len(chunk)
                if received > MAX_BYTES:
                    raise RLBotInstallError(f"{url} is far bigger than RLBot's installer: nothing was run.")
                if time.monotonic() > deadline:
                    raise RLBotInstallError(
                        f"The download of RLBot's installer was stopped after {minutes:g} minutes without "
                        f"finishing. {DOWNLOAD_HINT}"
                    )
    except (urllib.error.URLError, OSError) as err:
        raise RLBotInstallError(f"{url} could not be downloaded ({err}). {DOWNLOAD_HINT}") from None
    return b"".join(parts)


def msi_problem(data: bytes, expected: Installer) -> str | None:
    """Why the downloaded bytes are not the installer RLBot published (None: they are)."""
    if expected.size is not None and len(data) != expected.size:
        return f"it is {len(data)} bytes, RLBot's release lists {expected.size}"
    if not MIN_BYTES <= len(data) <= MAX_BYTES:
        return f"it is {len(data)} bytes, not the size of an installer"
    if not data.startswith(MSI_MAGIC):
        return "it is not a Windows Installer (.msi) file"
    if expected.sha256 is not None and hashlib.sha256(data).hexdigest() != expected.sha256:
        return "it does not match the SHA-256 hash RLBot's release lists"
    if V5_MARK not in data:
        return "it is not the installer of RLBot v5"
    return None


def run_msi(msi: Path) -> int:
    """RLBot's installer with its own windows (no quiet options): the person chooses and confirms there."""
    return subprocess.run(["msiexec", "/i", str(msi)], check=False).returncode


def local_app_data() -> Path | None:
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) if local else None


def launchers(local: Path | None) -> list[Path]:
    """RLBot v5's launchers found (RLBot5\\bin\\launcher.exe, what RLBot's own shortcuts open)."""
    if local is None:
        return []
    found = [base / "RLBot5" / "bin" / "launcher.exe" for base in (local, local / "Programs")]
    return [path for path in found if path.is_file()]


def remember_program(runs: Path, program: Path) -> bool:
    """Sets the dashboard's RLBot program (Open RLBot) unless it already has an RLBot v5 program that still
    exists (a chosen RLBot v4 program is replaced)."""
    from nitro.dashboard import edits

    file = runs / ".dashboard" / "prefs.json"
    try:
        prefs = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prefs = {"theme": "dark"}  # the dashboard's first-run preferences
    if not isinstance(prefs, dict):
        prefs = {"theme": "dark"}
    chosen = prefs.get("rlbot_program")
    if chosen and Path(str(chosen)).is_file() and rlbot_find.is_v5(str(chosen)):
        return False
    prefs["rlbot_program"] = str(program)
    file.parent.mkdir(parents=True, exist_ok=True)
    edits.save(file, json.dumps(prefs, indent=1))
    return True


def install(
    runs: Path,
    tmp: Path = TMP,
    windows: bool = sys.platform == "win32",
    local: Path | None = None,
    installer: Callable[[], Installer] = find_installer,
    fetch: Callable[[str], bytes] = download,
    run: Callable[[Path], int] = run_msi,
) -> int:
    """Downloads, checks and opens RLBot's installer, then says whether RLBot is found: 0 when it is. Nothing
    is downloaded when RLBot is installed already (its launcher keeps it up to date)."""
    if not windows:
        print(NOT_WINDOWS, flush=True)
        return 1
    local = local_app_data() if local is None else local
    program = find_rlbot([runs], local)
    if program is not None:
        print(f"RLBot v5 is installed already: {program} (its launcher keeps it up to date).", flush=True)
        return 0
    older = rlbot_find.find_older(local)
    if older is not None:
        print(f"{older_note(older)} RLBot v5 installs next to it; RLBot v4 stays as it is.", flush=True)
    found = installer()
    print(f"Downloading RLBot's official installer from {found.url}", flush=True)
    try:
        data = fetch(found.url)
    except RLBotInstallError as err:
        print(f"RLBot was not installed: {err}", flush=True)
        return 1
    problem = msi_problem(data, found)
    if problem is not None:
        print(
            f"RLBot was not installed: the download is not RLBot's installer ({problem}). Nothing from it "
            f"was run. Try again later, or get RLBot from {SITE}.",
            flush=True,
        )
        return 1
    tmp.mkdir(parents=True, exist_ok=True)
    msi = tmp / "rlbot-v5-installer.msi"
    msi.write_bytes(data)
    print("Opening RLBot's installer: follow its windows. NitroLearn waits until it is closed.", flush=True)
    try:
        code = run(msi)
    except OSError as err:
        code = -1
        print(f"Windows Installer could not be started ({err}).", flush=True)
    finally:
        msi.unlink(missing_ok=True)
    if code == MSI_CANCELLED:
        print("RLBot's installer was cancelled.", flush=True)
    elif code not in (*MSI_DONE, MSI_ALREADY, -1):  # -1: said above
        print(f"RLBot's installer ended with code {code} (see its message).", flush=True)
    program = find_rlbot([runs], local)
    if program is None:
        print(
            f"RLBot was not found after its installer. Get it from {SITE} if it did not install.", flush=True
        )
        return 1
    print(f"RLBot v5 is installed: {program}", flush=True)
    if code == 3010:
        print("Windows asks for a restart to finish RLBot's install.", flush=True)
    found_launchers = launchers(local)
    if len(found_launchers) == 1 and remember_program(runs, found_launchers[0]):
        print("The dashboard's Open RLBot now opens it.", flush=True)
    print(REMOVING, flush=True)
    return 0


def status(runs: Path, windows: bool = sys.platform == "win32", local: Path | None = None) -> int:
    """Says where RLBot is: 0 when found, 1 when not (or off Windows, where it is not looked for)."""
    if not windows:
        print(NOT_WINDOWS, flush=True)
        return 1
    local = local_app_data() if local is None else local
    program = find_rlbot([runs], local)
    older = rlbot_find.find_older(local)
    if older is not None and program is None:
        print(older_note(older), flush=True)
    print(
        f"RLBot v5: {program}" if program else f"RLBot v5: not found (nitro install-rlbot, or {SITE})",
        flush=True,
    )
    return 0 if program else 1


def older_note(older: Path) -> str:
    return f"RLBot v4 found at {older}; NitroLearn's bots need RLBot v5."
