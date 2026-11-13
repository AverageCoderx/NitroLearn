"""`nitro update`: NitroLearn's newest release, only when asked. Nothing here runs in the background.

The installed version is Cargo.toml's (the installer's `project_version`); releases are the GitHub Releases
of the project, tagged v<version>. Two ways to the new version:

- a git checkout moves to the release's tag (fetched first), only when its tracked files are unchanged;
- any other folder gets the tag's source archive: its files are copied over the folder's, never touching
  what is the user's (the bots, settings, replays, saves and tool folders, read from .gitignore with a fixed
  minimum). A file of the installed release that the new one no longer has is removed (only when the
  installed release's own archive could be read to tell). A file of NitroLearn's that the user changed is
  copied to .tmp/update/replaced/<old version>/ before it is overwritten or removed.

Then the installer runs again (install.sh, or install.ps1 on Windows). The compiled core is never in the
code: the installer replaces it. On Windows a running NitroLearn keeps its core locked, so there the
installer runs in its own window once NitroLearn has closed.

    python -m nitro.update --wait-pid <pid> -- <command ...>   (waits for a process to end, then starts one)
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

REPO = "AverageCoderx/NitroLearn"
LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
ARCHIVE = f"https://github.com/{REPO}/archive/refs/tags/v{{version}}.zip"
AGENT = "NitroLearn-updater"  # the only thing said about the asker: no names, no machine details
CHECK_SECONDS = 10.0
DOWNLOAD_SECONDS = 180.0
GIT_SECONDS = 300.0
WINDOWS = sys.platform == "win32"
WORK = Path(".tmp") / "update"
REPLACED = WORK / "replaced"
NOTE_LINES = 8
NOTE_CHARS = 700
# Never written by an update, whatever .gitignore says: what the user makes and what the installer keeps.
KEPT = (
    "/runs/",
    "/bots/",
    "/saves/",
    "/ab/",
    "/wandb/",
    "/replays/",
    "*.replay",
    "*.rlpr*",
    "/my-settings/",
    "/packs/",
    "*.nitropack.zip",
    "/collision_meshes/",
    "/rocketsim_recordings/",
    "/.venv/",
    "/.tools/",
    "/.cache/",
    "/.tmp/",
    "/.scratch/",
    "/target/",
    "/.git",
)
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

Fetch = Callable[[str, float], bytes]
Say = Callable[[str], None]


class UpdateError(Exception):
    """The update cannot go on; the message says why in plain words."""


# ---------- versions ----------


def _install_steps(root: Path) -> ModuleType:
    """The installer's steps (tools/install_steps.py, standard library only) as a module."""
    path = root / "tools" / "install_steps.py"
    spec = importlib.util.spec_from_file_location("nitro_install_steps", path)
    if spec is None or spec.loader is None:
        raise UpdateError(f"{path} is missing: this folder is not a complete NitroLearn")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # reading leaves no file behind
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes
    return module


def installed_version(root: Path) -> str:
    """The version of the NitroLearn in `root`, as the installer reads it (Cargo.toml)."""
    steps = _install_steps(root)
    try:
        return str(steps.project_version(root))
    except (OSError, steps.Stop) as err:
        raise UpdateError(f"The installed version could not be read: {err}") from None


def version_key(version: str) -> tuple[tuple[int, ...], int]:
    """Orders versions: 1.10.0 after 1.9.2, and 1.1.0-rc1 before 1.1.0."""
    core, _, extra = version.strip().lstrip("v").partition("-")
    numbers = tuple(int(n) for n in re.findall(r"\d+", core.split("+")[0]))
    return (numbers + (0,) * (3 - len(numbers)), 0 if extra else 1)


# ---------- checking ----------


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    date: str  # "1 December 2026", or "" when GitHub gives none
    notes: str  # the release notes' first lines
    page: str  # the release's page on GitHub


@dataclass(frozen=True)
class Check:
    installed: str
    release: Release
    state: str  # "newer" (an update is out), "same", or "older" (this folder is ahead of the release)
    git: bool  # a git checkout: updated with git
    blocked: str  # why the update cannot run in this folder now; "" when it can

    @property
    def line(self) -> str:
        """The check's result in one sentence."""
        out = self.release
        if self.state == "newer":
            when = f" ({out.date})" if out.date else ""
            return f"You have {self.installed}. NitroLearn {out.version} is out{when}."
        if self.state == "older":
            return f"You have {self.installed}, newer than the newest release ({out.version})."
        return f"You have {self.installed}, the newest version."

    def view(self) -> dict[str, object]:
        return {**asdict(self), "line": self.line}


def fetch(url: str, seconds: float) -> bytes:
    """One request, with a short time limit and a generic name for the asker."""
    request = urllib.request.Request(
        url, headers={"User-Agent": AGENT, "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=seconds) as answer:
        data: bytes = answer.read()
    return data


def _reason(err: BaseException) -> str:
    reason = getattr(err, "reason", None) or err
    return str(reason).strip() or type(err).__name__


def _http_problem(err: urllib.error.HTTPError, what: str) -> str:
    if err.code == 404:
        return f"{what}: no release was found (NitroLearn's page on GitHub may not be public yet)."
    limited = err.headers is not None and err.headers.get("X-RateLimit-Remaining") == "0"
    if err.code == 429 or limited:
        return f"{what}: GitHub allows only a few checks an hour from one connection. Try again later."
    if err.code == 403:
        return f"{what}: the request was refused (403). A firewall or proxy on this network may block GitHub."
    return f"{what}: GitHub answered {err.code} ({err.reason})."


def _date(text: object) -> str:
    try:
        day = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return ""
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def first_lines(body: object, lines: int = NOTE_LINES, chars: int = NOTE_CHARS) -> str:
    """The first lines of release notes, plain: headings' marks and empty lines left out."""
    kept: list[str] = []
    for line in str(body or "").replace("\r\n", "\n").split("\n"):
        text = line.strip().lstrip("#").strip()
        if text:
            kept.append(re.sub(r"^[*+] ", "- ", text))
        if len(kept) == lines:
            break
    out = "\n".join(kept)
    return out if len(out) <= chars else out[: chars - 1].rstrip() + "…"


def read_release(data: object) -> Release:
    """GitHub's answer for a release, checked for what is used."""
    if not isinstance(data, dict) or not isinstance(data.get("tag_name"), str):
        raise ValueError("no tag")
    tag = data["tag_name"]
    version = tag.removeprefix("v")
    if not re.fullmatch(r"\d+(\.\d+)*(-[0-9A-Za-z.]+)?", version):
        raise ValueError(f"the tag {tag!r} names no version")
    page = str(data.get("html_url") or "")
    return Release(
        version=version,
        tag=tag,
        date=_date(data.get("published_at") or ""),
        notes=first_lines(data.get("body")),
        page=page if page.startswith("https://github.com/") else "",
    )


def check(root: Path, get: Fetch = fetch) -> Check:
    """Asks GitHub for the newest release and compares it with the folder's version. UpdateError, with a
    message for the user, when it cannot."""
    installed = installed_version(root)
    what = "Could not check for updates"
    try:
        answer = get(LATEST, CHECK_SECONDS)
    except urllib.error.HTTPError as err:
        raise UpdateError(_http_problem(err, what)) from None
    except (urllib.error.URLError, OSError) as err:
        raise UpdateError(
            f"{what}: GitHub cannot be reached ({_reason(err)}). Check the connection."
        ) from None
    try:
        release = read_release(json.loads(answer.decode("utf-8")))
    except (ValueError, UnicodeDecodeError) as err:
        raise UpdateError(f"{what}: GitHub's answer could not be read ({err}). Try again later.") from None
    new, old = version_key(release.version), version_key(installed)
    state = "newer" if new > old else "older" if new < old else "same"
    git = is_git(root)
    return Check(installed, release, state, git, blocked=git_blocked(root) if git else "")


# ---------- what may not run beside an update ----------


def trainers(runs: Path) -> list[str]:
    """The bots in `runs` that are training now."""
    from nitro.dashboard import bots

    return [folder.root.name for folder in bots.bot_folders(runs) if bots.running(folder) is not None]


def busy(runs: Path) -> str:
    """Why an update cannot run now (a bot trains), or ""."""
    names = trainers(runs)
    if not names:
        return ""
    listed = ", ".join(names)
    return f"Stop training first: {listed} {'is' if len(names) == 1 else 'are'} training."


# ---------- a git checkout ----------


def is_git(root: Path) -> bool:
    return (root / ".git").exists()


def _git(root: Path, *args: str, seconds: float = 60.0) -> subprocess.CompletedProcess[str]:
    flags = subprocess.CREATE_NO_WINDOW if WINDOWS else 0  # type: ignore[attr-defined, unused-ignore]
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=seconds,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},  # never waits for a password
            creationflags=flags,
        )
    except FileNotFoundError:
        raise UpdateError(
            "This folder is a git checkout, but git is not installed: install git, or update by hand."
        ) from None
    except subprocess.TimeoutExpired:
        raise UpdateError(
            f"git {args[0]} did not finish in {seconds:.0f} seconds. Try again later."
        ) from None


def _said(done: subprocess.CompletedProcess[str]) -> str:
    lines = [line.strip() for line in (done.stderr + "\n" + done.stdout).splitlines() if line.strip()]
    return lines[-1] if lines else f"git ended with code {done.returncode}"


def git_changes(root: Path) -> list[str]:
    """The tracked files changed in the checkout (files the user added do not count)."""
    done = _git(root, "status", "--porcelain", "--untracked-files=no")
    if done.returncode != 0:
        raise UpdateError(f"git could not say what changed in this folder ({_said(done)}).")
    return [line[3:].strip() for line in done.stdout.splitlines() if line.strip()]


def git_blocked(root: Path) -> str:
    """Why git cannot update this checkout as it is, or ""."""
    try:
        changes = git_changes(root)
    except UpdateError as err:
        return str(err)
    if not changes:
        return ""
    shown = ", ".join(changes[:3]) + (f" and {len(changes) - 3} more" if len(changes) > 3 else "")
    return (
        f"There are changes to the code in this folder ({shown}). NitroLearn never overwrites them: "
        "commit or undo them first, then update."
    )


def update_git(root: Path, release: Release, say: Say = print) -> None:
    """Moves the checkout to the release's tag: a fast-forward of its branch, or a checkout of the tag when
    no branch is checked out. Refuses with changed files, or a branch with commits of its own."""
    if problem := git_blocked(root):
        raise UpdateError(problem)
    say("Fetching the release from git")
    fetched = _git(root, "fetch", "--tags", seconds=GIT_SECONDS)
    if fetched.returncode != 0:
        raise UpdateError(f"git could not fetch the release ({_said(fetched)}). Nothing was changed.")
    tag = release.tag
    if _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}").returncode != 0:
        raise UpdateError(f"The release's tag {tag} is not in this checkout's remote. Nothing was changed.")
    branch = _git(root, "symbolic-ref", "-q", "--short", "HEAD")
    if branch.returncode == 0:
        name = branch.stdout.strip()
        if _git(root, "merge-base", "--is-ancestor", "HEAD", tag).returncode != 0:
            raise UpdateError(
                f"The branch {name} has commits the release does not have. NitroLearn does not merge them "
                f"for you: run `git merge {tag}` yourself. Nothing was changed."
            )
        say(f"Moving {name} to {tag}")
        moved = _git(root, "merge", "--ff-only", tag, seconds=GIT_SECONDS)
    else:
        say(f"Checking out {tag}")
        moved = _git(root, "checkout", "--quiet", tag, seconds=GIT_SECONDS)
    if moved.returncode != 0:
        raise UpdateError(f"git could not move to {tag} ({_said(moved)}).")


# ---------- a folder from the release's archive ----------


@dataclass
class Kept:
    """Which paths are the user's: .gitignore's patterns (only these forms: `name`, `/name`, `name/`,
    `dir/name`, with * and ?) and the fixed minimum. Matched without regard to case."""

    patterns: list[tuple[str, bool, bool]] = field(default_factory=list)  # (glob, anchored, folders only)

    @classmethod
    def read(cls, *ignore_files: Path) -> Kept:
        lines = list(KEPT)
        for file in ignore_files:
            with contextlib.suppress(OSError, UnicodeDecodeError):
                lines += file.read_text(encoding="utf-8").splitlines()
        kept = cls()
        for line in lines:
            text = line.strip()
            if not text or text.startswith(("#", "!")):
                continue
            folder = text.endswith("/")
            text = text.strip("/")
            kept.patterns.append((text.lower(), line.strip().startswith("/") or "/" in text, folder))
        return kept

    def covers(self, path: str) -> bool:
        """Whether `path` (relative, with /) is the user's: it, or a folder it is in, matches."""
        parts = path.lower().split("/")
        for pattern, anchored, folders_only in self.patterns:
            for end in range(1, len(parts) + 1):
                if folders_only and end == len(parts):
                    continue
                name = "/".join(parts[:end]) if anchored else parts[end - 1]
                if fnmatch.fnmatchcase(name, pattern):
                    return True
        return False


@dataclass
class Copied:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)  # the user's changed copies, kept in `backup`
    removed: list[str] = field(default_factory=list)  # the old release's files the new one no longer has
    kept: list[str] = field(default_factory=list)  # in the release, but the user's: never written
    backup: str = ""


def unpack(data: bytes, target: Path) -> Path:
    """The source archive `data` into `target`, without its one top folder. Refuses an archive with a path
    that would land outside `target`; links are left out."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UpdateError("The download is not a zip archive. Nothing was changed.") from None
    members = [info for info in archive.infolist() if not info.is_dir()]
    tops = {PurePosixPath(info.filename).parts[0] for info in members if info.filename}
    if len(tops) != 1:
        raise UpdateError("The download is not NitroLearn's source archive. Nothing was changed.")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for info in members:
        parts = PurePosixPath(info.filename).parts[1:]
        if (
            not parts
            or any(p in ("", "..") or ":" in p or "\\" in p for p in parts)
            or info.filename[0] == "/"
        ):
            raise UpdateError(f"The download holds a strange path ({info.filename}). Nothing was changed.")
        mode = info.external_attr >> 16
        if mode & 0o170000 == 0o120000:
            continue  # a link
        file = target.joinpath(*parts)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(archive.read(info))
        if mode & 0o111 and not WINDOWS:
            file.chmod(0o755)
    return target


def _write(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".update-part")
    shutil.copyfile(source, part)
    with contextlib.suppress(OSError):
        shutil.copymode(source, part)
    os.replace(part, target)


def copy_release(root: Path, new: Path, old: Path | None, backup: Path, say: Say = print) -> Copied:
    """Copies the release's files (`new`) over `root`, and removes the old release's (`old`) files that the
    new one no longer has. The user's paths (`Kept`) are left alone. A file that differs from the old
    release (or from the new one when the old one is not at hand) is the user's change: it is copied into
    `backup` first. Without `old` nothing is removed."""
    kept = Kept.read(root / ".gitignore", new / ".gitignore")
    out = Copied(backup=str(backup))
    for file in sorted(path for path in new.rglob("*") if path.is_file()):
        rel = file.relative_to(new).as_posix()
        if kept.covers(rel):
            out.kept.append(rel)
            continue
        target = root / rel
        if target.is_dir() or any(parent.is_file() for parent in target.parents if root in parent.parents):
            out.kept.append(rel)  # a file or folder of the user's stands where the release has one
            continue
        if target.is_file():
            current = target.read_bytes()
            if current == file.read_bytes():
                continue
            base = None if old is None else old / rel
            if base is None or not base.is_file() or base.read_bytes() != current:
                saved = backup / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
                out.replaced.append(rel)
            out.changed.append(rel)
        else:
            out.added.append(rel)
        _write(target, file)
    if old is not None:
        remove_dropped(root, new, old, kept, backup, out)
    gone = f"removed {len(out.removed)} the new version no longer has" if out.removed else "removed none"
    say(f"Copied {len(out.changed)} changed and {len(out.added)} new files; {gone}.")
    return out


def remove_dropped(root: Path, new: Path, old: Path, kept: Kept, backup: Path, out: Copied) -> None:
    """Removes from `root` the files the old release shipped and the new one does not (names compared
    without regard to case, as Windows does), never the user's paths; a removed file the user had changed
    is copied into `backup` first. Folders left empty by it go too."""
    shipped = {path.relative_to(new).as_posix().lower() for path in new.rglob("*") if path.is_file()}
    for file in sorted(path for path in old.rglob("*") if path.is_file()):
        rel = file.relative_to(old).as_posix()
        target = root / rel
        if rel.lower() in shipped or kept.covers(rel) or target.is_symlink() or not target.is_file():
            continue
        try:
            if target.read_bytes() != file.read_bytes():
                saved = backup / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
                out.replaced.append(rel)
            target.unlink()
        except OSError:
            continue  # in use: it stays, as before
        out.removed.append(rel)
        for folder in target.parents:
            if folder == root or root not in folder.parents:
                break
            try:
                folder.rmdir()  # only an empty one
            except OSError:
                break


def update_files(
    root: Path, release: Release, installed: str, get: Fetch = fetch, say: Say = print
) -> Copied:
    """Downloads the release's source archive (and the installed version's, to tell the user's changes
    from the release's) into .tmp/update and copies the release's files over the folder."""
    work = root / WORK
    say(f"Downloading NitroLearn {release.version}")
    try:
        new = unpack(get(ARCHIVE.format(version=release.version), DOWNLOAD_SECONDS), work / "new")
    except urllib.error.HTTPError as err:
        raise UpdateError(_http_problem(err, "The release could not be downloaded")) from None
    except (urllib.error.URLError, OSError) as err:
        raise UpdateError(
            f"The release could not be downloaded ({_reason(err)}). Check the connection. Nothing was "
            "changed."
        ) from None
    old: Path | None
    try:
        old = unpack(get(ARCHIVE.format(version=installed), DOWNLOAD_SECONDS), work / "old")
    except (urllib.error.URLError, OSError, UpdateError):
        old = None
        say(
            f"The files of {installed} could not be downloaded to compare with: every file of NitroLearn's "
            "that differs is kept in the backup folder first."
        )
    try:
        return copy_release(root, new, old, root / REPLACED / installed, say)
    finally:
        for folder in (work / "new", work / "old"):
            shutil.rmtree(folder, ignore_errors=True)


# ---------- Windows' Apps list ----------

# NitroLearn-Setup's own entry in Windows' Apps list (installer/NitroLearn.iss: its AppId, with "_is1").
APPS_LIST_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{8C1E5A42-3B7D-4F2A-9E61-6D0B7C2F4A19}_is1"
)


def show_version_in_apps_list(root: Path, version: str, registry: Any = None) -> bool:
    """Windows, after an update of a folder NitroLearn-Setup installed: the version Windows' Apps list shows
    follows (DisplayVersion in Setup's own per-user entry, only when that entry is for this folder). True
    when it was changed."""
    if registry is None:
        if not WINDOWS:
            return False
        import winreg as registry
    try:
        access = registry.KEY_QUERY_VALUE | registry.KEY_SET_VALUE
        with registry.OpenKey(registry.HKEY_CURRENT_USER, APPS_LIST_KEY, 0, access) as key:
            location = str(registry.QueryValueEx(key, "InstallLocation")[0])
            if _same_folder(location, str(root)):
                registry.SetValueEx(key, "DisplayVersion", 0, registry.REG_SZ, version)
                return True
    except OSError:
        pass
    return False


def _same_folder(one: str, other: str) -> bool:
    def plain(path: str) -> str:
        return os.path.normcase(os.path.abspath(path.rstrip("\\/") or path))

    return plain(one) == plain(other)


# ---------- the installer ----------


def developer_install() -> bool:
    """Whether this Python has the developer tools (an install made with --dev)."""
    return importlib.util.find_spec("pytest") is not None and importlib.util.find_spec("maturin") is not None


CHOICES = Path(".tools") / "install_choices.json"  # the installer's record of the parts chosen
# The installer's optional parts (tools/install_steps.py's PARTS), as --parts names them.
PARTS = ("gpu", "rlbot", "start_menu", "desktop", "rust", "dev", "recordings")


def recorded_parts(root: Path, rust: bool = False, dev: bool = False) -> str | None:
    """The parts the last install chose, from its record, as the installer's --parts takes them ("none" for
    none); None without a record. RLBot v5 is left out: installed, it stays whatever --parts says, and an
    update never opens its installer again. Rust in the folder and a developer install stay."""
    try:
        data = json.loads((root / CHOICES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    chosen = {part for part in PARTS if data.get(part) is True and part != "rlbot"}
    chosen |= {"rust"} if rust else set()
    chosen |= {"dev"} if dev else set()
    return ",".join(part for part in PARTS if part in chosen) or "none"


def installer_command(root: Path, windows: bool = WINDOWS, dev: bool = False) -> list[str]:
    """The installer, as `Install NitroLearn.cmd` or `bash install.sh` runs it, without questions: the parts
    the last install chose (so its menu does not wait for Enter), and the dashboard not opened by it (the
    update opens it again itself). Without a record of the parts: Rust stays when the folder has it, and a
    developer install stays one."""
    rust = (root / ".tools" / "cargo" / "bin" / ("cargo.exe" if windows else "cargo")).is_file()
    parts = recorded_parts(root, rust, dev)
    if windows:
        flags = ["-Parts", parts] if parts is not None else ["-Dev"] if dev else ["-Rust"] if rust else []
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "install.ps1"),
            *flags,
            "-NoLaunch",
        ]
    flags = [f"--parts={parts}"] if parts is not None else ["--dev"] if dev else ["--rust"] if rust else []
    return ["bash", str(root / "install.sh"), *flags, "--no-launch"]


def _quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def finisher_command(
    wait_pid: int, installer: Sequence[str] | None, reopen: Sequence[str] | None, cwd: Path
) -> list[str]:
    """Windows: a PowerShell window that waits for process `wait_pid` (the dashboard, or `nitro update`) to
    end, so nothing holds the compiled core any more, then runs the installer and opens the dashboard
    again. It stays open to be read when the installer did not finish cleanly."""
    lines = [
        "$Host.UI.RawUI.WindowTitle = 'NitroLearn update'",
        "Write-Host 'Finishing the NitroLearn update. Waiting for NitroLearn to close...'",
        f"Wait-Process -Id {int(wait_pid)} -ErrorAction SilentlyContinue",
        "Start-Sleep -Seconds 2",
        "$code = 0",
    ]
    if installer:
        lines += [f"& {' '.join(_quoted(part) for part in installer)}", "$code = $LASTEXITCODE"]
    if reopen:
        arguments = ", ".join(_quoted(part) for part in reopen[1:])
        lines.append(
            f"Start-Process -FilePath {_quoted(reopen[0])} -ArgumentList {arguments} "
            f"-WorkingDirectory {_quoted(str(cwd))}"
        )
    lines.append(
        "if ($code -eq 0) { Write-Host 'The update is finished.'; Start-Sleep -Seconds 4 } else { "
        "Write-Host 'The installer did not finish cleanly: read the messages above.'; "
        "Read-Host 'Press Enter to close' }"
    )
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "; ".join(lines)]


def start_detached(command: Sequence[str], cwd: Path, console: bool) -> None:
    """Starts `command` on its own, living on when this process ends; `console`: in a window of its own
    (Windows)."""
    if WINDOWS:
        flags = subprocess.CREATE_NEW_CONSOLE if console else subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined, unused-ignore]
        subprocess.Popen(list(command), cwd=cwd, creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(
            list(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def reopen_command(argv: Sequence[str], executable: str = sys.executable) -> list[str] | None:
    """How the dashboard started (`nitro dashboard ...`) as a command that starts it again, windowless on
    Windows; None when this process is not the dashboard started that way."""
    if list(argv[:1]) != ["dashboard"]:
        return None
    python = Path(executable)
    if WINDOWS and python.name.lower() == "python.exe" and python.with_name("pythonw.exe").is_file():
        python = python.with_name("pythonw.exe")
    return [str(python), "-m", "nitro", *argv]


def relaunch_command(wait_pid: int, command: Sequence[str]) -> list[str]:
    """A process that waits for `wait_pid` to end, then starts `command` (this module's `main`)."""
    return [sys.executable, "-m", "nitro.update", "--wait-pid", str(int(wait_pid)), "--", *command]


# ---------- the whole update ----------


@dataclass
class Done:
    old: str
    new: str
    how: str  # "git" or "files"
    copied: Copied = field(default_factory=Copied)


def apply(root: Path, found: Check, get: Fetch = fetch, say: Say = print) -> Done:
    """Brings the folder's code to `found`'s release (installer not included)."""
    if found.state != "newer":
        raise UpdateError(f"{found.line} There is nothing to update.")
    if found.git:
        update_git(root, found.release, say)
        done = Done(found.installed, installed_version(root), "git")
    else:
        copied = update_files(root, found.release, found.installed, get, say)
        done = Done(found.installed, installed_version(root), "files", copied)
    if done.copied.replaced:
        say(
            f"Files of NitroLearn's that you had changed (kept in {done.copied.backup} before they were "
            "replaced):"
        )
        for rel in done.copied.replaced:
            say(f"  {rel}")
    return done


def command(
    root: Path,
    runs: Path,
    *,
    check_only: bool = False,
    yes: bool = False,
    installer: str = "",
    result: Path | None = None,
    get: Fetch = fetch,
    ask: Callable[[str], str] = input,
    say: Say = print,
    run_installer: Callable[[list[str], Path], int] | None = None,
) -> int:
    """`nitro update` (`check_only`: `nitro update --check`). `installer`: "now" runs it here, "window" in
    its own window once this process has ended (the default on Windows), "later" leaves it to the
    dashboard. `result` gets what happened as JSON, for the dashboard."""

    def keep(data: dict[str, object]) -> None:
        if result is not None:
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps(data, indent=1), encoding="utf-8")

    try:
        found = check(root, get)
    except UpdateError as err:
        say(str(err))
        keep({"error": str(err)})
        return 1
    say(found.line)
    if found.state == "newer" and found.release.notes:
        say(found.release.notes)
    if check_only or found.state != "newer":
        return 0
    try:
        if problem := found.blocked or busy(runs):
            raise UpdateError(problem)
        if not yes and ask(
            f"Update to {found.release.version} now? Your bots, settings and saves are kept. [y/N] "
        ).strip().lower() not in ("y", "yes"):
            say("Nothing was changed.")
            return 0
        done = apply(root, found, get, say)
    except UpdateError as err:
        say(str(err))
        keep({"error": str(err)})
        return 1
    show_version_in_apps_list(root, done.new)
    mode = installer or ("window" if WINDOWS else "now")
    data: dict[str, object] = {"old": done.old, "new": done.new, "how": done.how, **asdict(done.copied)}
    steps = installer_command(root, dev=developer_install())
    if mode == "later":
        say(f"Updated the code from {done.old} to {done.new}. The installer runs once NitroLearn has closed.")
        keep({**data, "installer": "later"})
        return 0
    if mode == "window":
        start_detached(finisher_command(os.getpid(), steps, None, root), root, console=True)
        say(
            f"Updated the code from {done.old} to {done.new}. A new window runs the installer once this ends."
        )
        keep({**data, "installer": "window"})
        return 0
    say("Running the installer again (a few minutes; longer when PyTorch changes)")
    runner = run_installer or (
        lambda cmd, cwd: subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL).returncode
    )
    code = runner(steps, root)
    if code != 0:
        say(
            f"Updated from {done.old} to {done.new}, but the installer did not finish cleanly: read its "
            "messages above."
        )
        keep({**data, "installer": "failed"})
        return 1
    say(f"Updated from {done.old} to {done.new}. Restart the dashboard to use it.")
    keep({**data, "installer": "done"})
    return 0


# ---------- waiting for a process, then starting one ----------


def _alive(pid: int) -> bool:
    from nitro.dashboard.bots import pid_alive

    return pid_alive(pid)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] != "--wait-pid" or args[2] != "--":
        print("usage: python -m nitro.update --wait-pid <pid> -- <command ...>", file=sys.stderr)
        return 2
    pid, then = int(args[1]), args[3:]
    deadline = time.monotonic() + 120
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.3)
    start_detached(then, Path.cwd(), console=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
