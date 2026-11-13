"""`nitro update`: the installed version, the check against GitHub (answered here, never asked for real), the
update of a folder from the release's archive (the release's files are written and those it dropped removed,
nothing of the user's is touched, files the user changed are copied first), the update of a git checkout
(refused with local changes), and the refusal while a bot trains."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from nitro import update
from nitro.cli import build_parser, main
from nitro.dashboard import bots
from test_dashboard import make_bot

PROJECT = Path(__file__).resolve().parents[1]
CARGO = '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "{version}"\nedition = "2024"\n'


def release_answer(version: str, body: str = "## Faster replays\n\n* Twice as fast\n* Fewer files") -> bytes:
    return json.dumps(
        {
            "tag_name": f"v{version}",
            "published_at": "2026-12-01T10:00:00Z",
            "body": body,
            "html_url": f"https://github.com/{update.REPO}/releases/tag/v{version}",
        }
    ).encode()


def http_error(code: int, remaining: str = "") -> urllib.error.HTTPError:
    headers = Message()
    if remaining:
        headers["X-RateLimit-Remaining"] = remaining
    return urllib.error.HTTPError(
        update.LATEST, code, "Not Found" if code == 404 else "Forbidden", headers, None
    )


class GitHub:
    """Stands in for GitHub: each URL's answer (bytes, or an exception to raise), and the URLs asked."""

    def __init__(self, answers: dict[str, bytes | Exception]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, url: str, seconds: float) -> bytes:
        self.asked.append(url)
        assert seconds <= update.DOWNLOAD_SECONDS
        answer = self.answers.get(url, http_error(404))
        if isinstance(answer, Exception):
            raise answer
        return answer


def archive(version: str, files: dict[str, str]) -> bytes:
    """A source archive as GitHub makes one: every file under one top folder."""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as zipped:
        zipped.writestr(f"NitroLearn-{version}/", "")
        for name, text in files.items():
            zipped.writestr(f"NitroLearn-{version}/{name}", text)
    return data.getvalue()


def release_files(version: str, changes: dict[str, str] | None = None) -> dict[str, str]:
    """What a release ships (a few of NitroLearn's files), with `changes` on top."""
    files = {
        "Cargo.toml": CARGO.format(version=version),
        "tools/install_steps.py": (PROJECT / "tools" / "install_steps.py").read_text(encoding="utf-8"),
        ".gitignore": (PROJECT / ".gitignore").read_text(encoding="utf-8"),
        "README.md": "NitroLearn\n",
        "python/nitro/trainer.py": f"VERSION = {version!r}\n",
        "python/nitro/removed_later.py": "OLD = True\n",
        "configs/presets/starter.toml": "[run]\nseed = 1\n",
    }
    files.update(changes or {})
    return files


def installed(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


# ---------- versions and the check ----------
def test_the_installed_version_is_cargo_tomls_as_the_installer_reads_it(tmp_path: Path) -> None:
    root = installed(tmp_path, release_files("1.0.0"))
    assert update.installed_version(root) == "1.0.0"
    assert update.installed_version(PROJECT) == update._install_steps(PROJECT).project_version(PROJECT)
    (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    with pytest.raises(update.UpdateError, match="installed version could not be read"):
        update.installed_version(root)
    order = ["0.9.9", "1.0.0-rc1", "1.0.0", "1.0.1", "1.9.2", "1.10.0", "2.0"]
    assert sorted(order, key=update.version_key) == order


@pytest.mark.parametrize(
    ("latest", "state", "line"),
    [
        ("1.1.0", "newer", "You have 1.0.0. NitroLearn 1.1.0 is out (1 December 2026)."),
        ("1.0.0", "same", "You have 1.0.0, the newest version."),
        ("0.9.0", "older", "You have 1.0.0, newer than the newest release (0.9.0)."),
    ],
)
def test_the_check_compares_the_newest_release_with_this_folder(
    tmp_path: Path, latest: str, state: str, line: str
) -> None:
    root = installed(tmp_path, release_files("1.0.0"))
    github = GitHub({update.LATEST: release_answer(latest)})
    found = update.check(root, github)
    assert (found.state, found.line, found.installed) == (state, line, "1.0.0")
    assert github.asked == [update.LATEST], "one request, to GitHub's API"
    assert found.release.notes == "Faster replays\n- Twice as fast\n- Fewer files", (
        "the notes' first lines, plain"
    )
    assert found.release.page.startswith("https://github.com/") and not found.git and found.blocked == ""
    assert found.view()["line"] == line and found.view()["release"]["version"] == latest


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (http_error(404), "no release was found"),
        (http_error(403, remaining="0"), "only a few checks an hour"),
        (http_error(429), "only a few checks an hour"),
        (http_error(403), "A firewall or proxy on this network may block GitHub"),
        (http_error(500), "GitHub answered 500"),
        (urllib.error.URLError(OSError("Name or service not known")), "GitHub cannot be reached"),
        (TimeoutError("timed out"), "GitHub cannot be reached (timed out)"),
        (b"<html>not json", "GitHub's answer could not be read"),
        (b'{"tag_name": "nightly"}', "GitHub's answer could not be read"),
    ],
)
def test_a_check_that_cannot_be_answered_says_so_plainly(
    tmp_path: Path, answer: bytes | Exception, said: str
) -> None:
    root = installed(tmp_path, release_files("1.0.0"))
    with pytest.raises(update.UpdateError) as caught:
        update.check(root, GitHub({update.LATEST: answer}))
    assert str(caught.value).startswith("Could not check for updates: ") and said in str(caught.value)


def test_a_request_names_no_one_and_has_a_short_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Answer(io.BytesIO):
        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def urlopen(request: Any, timeout: float) -> Answer:
        seen.update(url=request.full_url, headers=dict(request.header_items()), timeout=timeout)
        return Answer(b"{}")

    monkeypatch.setattr(update.urllib.request, "urlopen", urlopen)
    assert update.fetch(update.LATEST, update.CHECK_SECONDS) == b"{}"
    assert seen["url"] == "https://api.github.com/repos/AverageCoderx/NitroLearn/releases/latest"
    assert seen["headers"]["User-agent"] == "NitroLearn-updater" and seen["timeout"] <= 10
    assert set(seen["headers"]) == {"User-agent", "Accept"}, "nothing else about the asker"


# ---------- a folder from the release's archive ----------
def user_things(root: Path) -> dict[str, bytes]:
    """Files of the user's in a NitroLearn folder, from every kind the update must keep."""
    things = {
        "runs/my-bot/run.json": b"{}",
        "runs/my-bot/checkpoints/000000001000/state.pt": b"save",
        "bots/my-bot/bot.toml": b"[settings]",
        "saves/old.pt": b"save",
        "ab/a-vs-b/results.csv": b"1",
        "wandb/run-1/files": b"w",
        "replays/match.replay": b"r",
        "elsewhere/match.replay": b"r",
        "recordings/game.rlpr.zst": b"z",
        "my-settings/my-bot.toml": b"[run]",
        "packs/my-bot.nitropack.zip": b"pk",
        "my-bot.nitropack.zip": b"pk",
        "collision_meshes/soccar/mesh_0.cmf": b"m",
        ".venv/pyvenv.cfg": b"home = x",
        ".tools/shortcuts.json": b"{}",
        ".tools/prebuilt/nitro_bot": b"exe",
        ".cache/uv/x": b"c",
        ".scratch/notes.txt": b"n",
        "target/release/x": b"t",
        "python/nitro/_core.abi3.so": b"core",
        "python/nitro/_core.pyd": b"core",
        "notes-of-mine.txt": b"mine",
        "configs/presets/my-own-preset.toml": b"[run]\nseed = 9\n",
    }
    for name, data in things.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return things


def test_an_update_from_the_archive_writes_only_the_releases_files_and_keeps_everything_of_the_users(
    tmp_path: Path,
) -> None:
    root = installed(tmp_path / "NitroLearn", release_files("1.0.0"))
    mine = user_things(root)
    (root / "configs/presets/starter.toml").write_text("[run]\nseed = 2  # mine\n", encoding="utf-8")
    new = release_files(
        "1.1.0",
        {
            "python/nitro/trainer.py": "VERSION = '1.1.0'\n",
            "python/nitro/added.py": "NEW = True\n",
            "configs/presets/starter.toml": "[run]\nseed = 5\n",
            # A release never holds these (git ignores them); if one did, the user's copy would stay.
            "runs/my-bot/run.json": '{"from": "release"}',
            "my-settings/my-bot.toml": "[release]",
            "notes-of-mine.txt": "the release's notes",
        },
    )
    del new["python/nitro/removed_later.py"]
    github = GitHub(
        {
            update.LATEST: release_answer("1.1.0"),
            update.ARCHIVE.format(version="1.1.0"): archive("1.1.0", new),
            update.ARCHIVE.format(version="1.0.0"): archive("1.0.0", release_files("1.0.0")),
        }
    )
    said: list[str] = []
    result = tmp_path / "result.json"
    ran: list[list[str]] = []
    code = update.command(
        root,
        root / "runs",
        yes=True,
        installer="now",
        result=result,
        get=github,
        say=said.append,
        run_installer=lambda command, cwd: ran.append(command) or 0,
    )
    assert code == 0, said
    assert update.installed_version(root) == "1.1.0"
    assert (root / "python/nitro/trainer.py").read_text(encoding="utf-8") == "VERSION = '1.1.0'\n"
    assert (root / "python/nitro/added.py").read_text(encoding="utf-8") == "NEW = True\n"
    assert (root / "configs/presets/starter.toml").read_text(encoding="utf-8") == "[run]\nseed = 5\n"
    assert not (root / "python/nitro/removed_later.py").exists(), "the new version no longer has it"
    assert json.loads(result.read_text(encoding="utf-8"))["removed"] == ["python/nitro/removed_later.py"]
    for name, data in mine.items():
        if name != "notes-of-mine.txt":
            assert (root / name).read_bytes() == data, f"{name} is the user's"
    backup = root / ".tmp/update/replaced/1.0.0"
    assert (backup / "configs/presets/starter.toml").read_text(
        encoding="utf-8"
    ) == "[run]\nseed = 2  # mine\n"
    assert (backup / "notes-of-mine.txt").read_bytes() == b"mine", (
        "a file the release also ships: copied first"
    )
    assert (root / "notes-of-mine.txt").read_text(encoding="utf-8") == "the release's notes"
    assert not (backup / "python/nitro/trainer.py").exists(), "changed by the release only: no copy"
    data = json.loads(result.read_text(encoding="utf-8"))
    assert (data["old"], data["new"], data["how"], data["installer"]) == ("1.0.0", "1.1.0", "files", "done")
    assert sorted(data["replaced"]) == ["configs/presets/starter.toml", "notes-of-mine.txt"]
    assert "runs/my-bot/run.json" in data["kept"] and "my-settings/my-bot.toml" in data["kept"]
    assert data["added"] == ["python/nitro/added.py"]
    assert ran == [update.installer_command(root, dev=update.developer_install())]
    assert ran[0][:2] == ["bash", str(root / "install.sh")] or sys.platform == "win32"
    assert not (root / ".tmp/update/new").exists() and not (root / ".tmp/update/old").exists()
    assert said[-1] == "Updated from 1.0.0 to 1.1.0. Restart the dashboard to use it."


def test_files_the_new_version_dropped_go_but_never_the_users(tmp_path: Path) -> None:
    """Only files the installed release shipped and the new one does not are removed: one the user changed
    is kept in the backup first, a name that only changed case stays (one file on Windows), the user's
    folders are never touched, and empty folders left behind go."""
    old = release_files("1.0.0", {"python/nitro/gone/old.py": "X = 1\n", "docs/Guide.md": "guide\n"})
    old |= {"configs/presets/dropped.toml": "[run]\n", "my-settings/from-release.toml": "[run]\n"}
    root = installed(tmp_path / "NitroLearn", old)
    (root / "configs/presets/dropped.toml").write_text("[run]\nseed = 7\n", encoding="utf-8")
    (root / "my-settings/from-release.toml").write_text("mine\n", encoding="utf-8")
    new = release_files("1.1.0", {"docs/guide.md": "guide\n"})
    del new["python/nitro/removed_later.py"]
    folders = {}
    for name, files in (("old", old), ("new", new)):
        folders[name] = installed(tmp_path / name, files)
    said: list[str] = []
    backup = tmp_path / "backup"
    copied = update.copy_release(root, folders["new"], folders["old"], backup, said.append)
    assert sorted(copied.removed) == [
        "configs/presets/dropped.toml",
        "python/nitro/gone/old.py",
        "python/nitro/removed_later.py",
    ]
    assert not (root / "python/nitro/gone").exists(), "an emptied folder goes"
    assert (backup / "configs/presets/dropped.toml").read_text(encoding="utf-8") == "[run]\nseed = 7\n"
    assert "configs/presets/dropped.toml" in copied.replaced
    assert (root / "my-settings/from-release.toml").read_text(encoding="utf-8") == "mine\n"
    assert any((root / "docs").glob("[Gg]uide.md")), "a change of case only is not a removal"
    assert said[-1].endswith("removed 3 the new version no longer has.")
    nothing = update.copy_release(root, folders["new"], None, backup, said.append)
    assert nothing.removed == [] and said[-1].endswith("removed none.")


def test_without_the_old_release_every_differing_file_is_copied_first(tmp_path: Path) -> None:
    root = installed(tmp_path, release_files("1.0.0"))
    new = release_files("1.1.0", {"README.md": "NitroLearn 1.1\n"})
    github = GitHub(
        {
            update.LATEST: release_answer("1.1.0"),
            update.ARCHIVE.format(version="1.1.0"): archive("1.1.0", new),
        }
    )
    found = update.check(root, github)
    said: list[str] = []
    done = update.apply(root, found, github, said.append)
    assert sorted(done.copied.replaced) == ["Cargo.toml", "README.md", "python/nitro/trainer.py"]
    assert (root / ".tmp/update/replaced/1.0.0/README.md").read_text(encoding="utf-8") == "NitroLearn\n"
    assert any("could not be downloaded to compare" in line for line in said)


def test_a_strange_or_missing_download_changes_nothing(tmp_path: Path) -> None:
    root = installed(tmp_path / "NitroLearn", release_files("1.0.0"))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zipped:
        zipped.writestr("NitroLearn-1.1.0/README.md", "x")
        zipped.writestr("NitroLearn-1.1.0/../../outside.txt", "x")
    for answer, said in (
        (bad.getvalue(), "strange path"),
        (b"not a zip", "not a zip archive"),
        (urllib.error.URLError("no route"), "could not be downloaded"),
    ):
        github = GitHub(
            {update.LATEST: release_answer("1.1.0"), update.ARCHIVE.format(version="1.1.0"): answer}
        )
        with pytest.raises(update.UpdateError, match=said):
            update.apply(root, update.check(root, github), github, lambda _: None)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and ".tmp" not in p.parts} == before
    assert not (tmp_path / "outside.txt").exists()


def test_the_kept_paths_come_from_gitignore_and_the_fixed_minimum(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("# mine\n/private/\n*.secret\n!keep.secret\n", encoding="utf-8")
    kept = update.Kept.read(tmp_path / ".gitignore", tmp_path / "missing")
    for path in (
        "private/a.txt",
        "deep/x.secret",
        "runs/b/run.json",
        "RUNS/b/run.json",
        "a/b.replay",
        "x.rlpr.zst",
        ".tools/shortcuts.json",
        ".git/config",
        "packs/p.zip",
    ):
        assert kept.covers(path), path
    for path in ("python/nitro/runs.py", "docs/private/a.md", "configs/presets/a.toml", "README.md"):
        assert not kept.covers(path), path
    real = update.Kept.read(PROJECT / ".gitignore")
    assert real.covers("python/nitro/_core.cp311-win_amd64.pyd") and real.covers("crates/x/target/a")
    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT, capture_output=True, text=True, check=False)
    if tracked.returncode == 0 and tracked.stdout:
        assert not [p for p in tracked.stdout.splitlines() if real.covers(p)], (
            "every file NitroLearn ships is written"
        )


# ---------- a git checkout ----------
GIT = [
    "git",
    "-c",
    "user.name=NitroLearn test",
    "-c",
    "user.email=test@example.invalid",
    "-c",
    "init.defaultBranch=main",
]


def git(folder: Path, *args: str) -> str:
    done = subprocess.run([*GIT, *args], cwd=folder, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def commit(folder: Path, version: str, message: str) -> None:
    installed(folder, release_files(version))
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", message)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_git_checkout_moves_to_the_release_only_when_its_code_is_unchanged(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    commit(origin, "1.0.0", "Release 1.0.0")
    git(origin, "tag", "v1.0.0")
    work = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(origin), str(work))
    commit(origin, "1.1.0", "Release 1.1.0")
    git(origin, "tag", "v1.1.0")
    (work / "notes-of-mine.txt").write_text("mine", encoding="utf-8")
    github = GitHub({update.LATEST: release_answer("1.1.0")})

    (work / "README.md").write_text("my change\n", encoding="utf-8")
    found = update.check(work, github)
    assert found.git and found.state == "newer"
    assert "There are changes to the code in this folder (README.md)" in found.blocked
    with pytest.raises(update.UpdateError, match="changes to the code"):
        update.apply(work, found, github, lambda _: None)
    assert (work / "README.md").read_text(encoding="utf-8") == "my change\n", "never overwritten"
    assert update.installed_version(work) == "1.0.0" and git(work, "stash", "list") == "", "nothing stashed"

    git(work, "checkout", "--", "README.md")
    found = update.check(work, github)
    assert found.blocked == ""
    done = update.apply(work, found, github, lambda _: None)
    assert (done.old, done.new, done.how) == ("1.0.0", "1.1.0", "git")
    assert (
        git(work, "describe", "--tags") == "v1.1.0" and git(work, "symbolic-ref", "--short", "HEAD") == "main"
    )
    assert (work / "notes-of-mine.txt").read_text(encoding="utf-8") == "mine"
    assert github.asked == [update.LATEST, update.LATEST], "git fetched the code: no archive downloaded"

    # A branch with commits of its own is never merged for the user.
    commit(origin, "1.2.0", "Release 1.2.0")
    git(origin, "tag", "v1.2.0")
    (work / "mine.py").write_text("x = 1\n", encoding="utf-8")
    git(work, "add", "mine.py")
    git(work, "commit", "-q", "-m", "Mine")
    github = GitHub({update.LATEST: release_answer("1.2.0")})
    with pytest.raises(update.UpdateError, match="commits the release does not have"):
        update.apply(work, update.check(work, github), github, lambda _: None)
    assert update.installed_version(work) == "1.1.0"

    # No branch checked out (a tag): the newer tag is checked out.
    git(work, "checkout", "-q", "v1.1.0")
    update.apply(work, update.check(work, github), github, lambda _: None)
    assert update.installed_version(work) == "1.2.0"


# ---------- what may not run beside it, and the installer ----------
def test_an_update_is_refused_while_a_bot_trains(tmp_path: Path) -> None:
    root = installed(tmp_path / "NitroLearn", release_files("1.0.0"))
    runs = root / "runs"
    folder = make_bot(runs, "racer")
    (folder.root / bots.RUNNING).write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    github = GitHub({update.LATEST: release_answer("1.1.0")})
    said: list[str] = []
    result = tmp_path / "result.json"
    assert (
        update.command(root, runs, yes=True, installer="now", result=result, get=github, say=said.append) == 1
    )
    assert said[-1] == "Stop training first: racer is training."
    assert github.asked == [update.LATEST], "nothing downloaded"
    assert json.loads(result.read_text(encoding="utf-8")) == {
        "error": "Stop training first: racer is training."
    }
    (folder.root / bots.RUNNING).unlink()
    assert update.busy(runs) == ""


def test_nothing_changes_without_a_yes_or_a_newer_release(tmp_path: Path) -> None:
    root = installed(tmp_path, release_files("1.0.0"))
    github = GitHub({update.LATEST: release_answer("1.1.0")})
    said: list[str] = []
    assert update.command(root, root / "runs", get=github, ask=lambda _: "n", say=said.append) == 0
    assert said[-1] == "Nothing was changed." and update.installed_version(root) == "1.0.0"
    assert update.command(root, root / "runs", check_only=True, get=github, say=said.append) == 0
    assert said[-2:] == [
        "You have 1.0.0. NitroLearn 1.1.0 is out (1 December 2026).",
        "Faster replays\n- Twice as fast\n- Fewer files",
    ]
    same = GitHub({update.LATEST: release_answer("1.0.0")})
    assert update.command(root, root / "runs", yes=True, get=same, say=said.append) == 0
    assert said[-1] == "You have 1.0.0, the newest version." and same.asked == [update.LATEST]
    failing = GitHub({update.LATEST: http_error(404)})
    assert update.command(root, root / "runs", check_only=True, get=failing, say=said.append) == 1
    assert said[-1].startswith("Could not check for updates: no release was found")


def test_the_installer_runs_as_the_install_scripts_run_it(tmp_path: Path) -> None:
    assert update.installer_command(tmp_path, windows=False) == [
        "bash",
        str(tmp_path / "install.sh"),
        "--no-launch",
    ]
    windows = update.installer_command(tmp_path, windows=True)
    assert windows == [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(tmp_path / "install.ps1"),
        "-NoLaunch",
    ]
    (tmp_path / ".tools/cargo/bin").mkdir(parents=True)
    (tmp_path / ".tools/cargo/bin/cargo").write_text("")
    assert update.installer_command(tmp_path, windows=False)[-2] == "--rust", "Rust in the folder stays"
    assert update.installer_command(tmp_path, windows=False, dev=True)[-2] == "--dev"
    assert update.installer_command(tmp_path, windows=True, dev=True)[-2] == "-Dev"
    install_ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    assert "[switch]$Rust" in install_ps1 and "[switch]$Dev" in install_ps1
    assert "[switch]$NoLaunch" in install_ps1 and "[AllowEmptyString()][string]$Parts" in install_ps1
    assert "--rust" in (PROJECT / "install.sh").read_text(encoding="utf-8")

    folder = Path("C:/Nitro's folder")
    command = update.finisher_command(
        4321, windows, ["C:/py/pythonw.exe", "-m", "nitro", "dashboard"], folder
    )
    script = command[-1]
    assert command[:5] == ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]
    assert script.index("Wait-Process -Id 4321") < script.index("install.ps1") < script.index("Start-Process")
    assert f"-WorkingDirectory '{str(folder).replace(chr(39), chr(39) * 2)}'" in script
    assert "'-m', 'nitro', 'dashboard'" in script and "Read-Host" in script
    assert "install.ps1" not in update.finisher_command(1, None, None, folder)[-1]

    assert update.reopen_command(["dashboard", "--runs", "runs"], "/v/bin/python") == [
        "/v/bin/python",
        "-m",
        "nitro",
        "dashboard",
        "--runs",
        "runs",
    ]
    assert update.reopen_command(["-q", "tests"]) is None, "not a dashboard: nothing to start again"
    assert update.relaunch_command(99, ["a", "b"])[1:] == [
        "-m",
        "nitro.update",
        "--wait-pid",
        "99",
        "--",
        "a",
        "b",
    ]


def test_an_update_installs_the_parts_chosen_before_without_asking(tmp_path: Path) -> None:
    """With the installer's record of the parts, the update passes them, so the installer's menu never waits
    for Enter; RLBot's installer is not opened again (an installed RLBot v5 stays anyway)."""
    steps = update._install_steps(PROJECT)
    assert update.PARTS == steps.PARTS and update.CHOICES == steps.CHOICES_RECORD
    record = tmp_path / update.CHOICES
    record.parent.mkdir(parents=True)
    chosen = dict.fromkeys(update.PARTS, False) | {"gpu": True, "rlbot": True, "start_menu": True}
    record.write_text(json.dumps(chosen), encoding="utf-8")
    assert update.installer_command(tmp_path, windows=True)[-3:] == ["-Parts", "gpu,start_menu", "-NoLaunch"]
    linux = update.installer_command(tmp_path, windows=False)
    assert linux[-2:] == ["--parts=gpu,start_menu", "--no-launch"]
    parsed = steps.parse(["--uv", "uv", linux[-2], linux[-1]])
    assert steps.switched(parsed) == chosen | {"rlbot": False}
    assert update.installer_command(tmp_path, windows=True, dev=True)[-2] == "gpu,start_menu,dev"
    record.write_text(json.dumps(dict.fromkeys(update.PARTS, False)), encoding="utf-8")
    assert update.installer_command(tmp_path, windows=True)[-2] == "none"
    record.write_text("not json", encoding="utf-8")
    assert "-Parts" not in update.installer_command(tmp_path, windows=True)


def test_the_relauncher_waits_for_the_process_then_starts_the_command(tmp_path: Path) -> None:
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    marker = tmp_path / "started.txt"
    code = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('yes')"
    began = time.monotonic()
    waiter = subprocess.run(
        [*update.relaunch_command(sleeper.pid, [sys.executable, "-c", code])],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(PROJECT / "python")},
        timeout=60,
        check=False,
    )
    sleeper.wait()
    assert waiter.returncode == 0 and time.monotonic() - began >= 1.0, "it waited"
    for _ in range(100):
        if marker.is_file():
            break
        time.sleep(0.05)
    assert marker.read_text() == "yes"


def test_the_installer_keeps_the_copies_an_update_made(tmp_path: Path) -> None:
    steps = update._install_steps(PROJECT)
    temporary = tmp_path / ".tmp"
    for name in (
        "release/core.whl",
        "update/new/README.md",
        "update/replaced/1.0.0/configs/a.toml",
        "loose.txt",
    ):
        (temporary / name).parent.mkdir(parents=True, exist_ok=True)
        (temporary / name).write_text("x", encoding="utf-8")
    steps.clear_temporary(temporary)
    left = sorted(p.relative_to(temporary).as_posix() for p in temporary.rglob("*") if p.is_file())
    assert left == ["update/replaced/1.0.0/configs/a.toml"]
    steps.clear_temporary(tmp_path / "missing")
    assert (tmp_path / "missing").is_dir()


# ---------- the command ----------
def test_nitro_update_has_plain_help_and_runs_without_the_core_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    helped = build_parser()._subparsers._group_actions[0].choices["update"].format_help()  # type: ignore[union-attr]
    assert (
        "--check" in helped and "--yes" in helped and "--installer" not in helped and "--result" not in helped
    )
    assert "only say whether a newer version is out" in helped
    asked: list[dict[str, Any]] = []

    def command(root: Path, runs: Path, **options: Any) -> int:
        asked.append({"root": root, "runs": runs, **options})
        return 0

    monkeypatch.setattr(update, "command", command)
    monkeypatch.setattr("nitro.build_info.check_core", lambda: pytest.fail("an update works on an old core"))
    assert main(["update", "--check", "--runs", str(tmp_path / "runs")]) == 0
    assert main(["update", "--yes", "--installer", "later", "--result", str(tmp_path / "r.json")]) == 0
    assert asked[0]["root"] == PROJECT and asked[0]["runs"] == tmp_path / "runs" and asked[0]["check_only"]
    assert asked[1]["yes"] and asked[1]["installer"] == "later" and asked[1]["result"] == tmp_path / "r.json"


class FakeRegistry:
    """winreg's few calls show_version_in_apps_list makes, over one key."""

    HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_SZ = 1, 2, 4, 1

    def __init__(self, values: dict[str, str] | None) -> None:
        self.values = values

    def OpenKey(self, hive: int, key: str, reserved: int, access: int) -> Any:
        if self.values is None or hive != self.HKEY_CURRENT_USER or key != update.APPS_LIST_KEY:
            raise FileNotFoundError(key)
        assert access & self.KEY_SET_VALUE
        return contextlib.nullcontext("key")

    def QueryValueEx(self, key: str, name: str) -> tuple[str, int]:
        assert self.values is not None
        return self.values[name], self.REG_SZ

    def SetValueEx(self, key: str, name: str, reserved: int, kind: int, value: str) -> None:
        assert self.values is not None
        self.values[name] = value


def test_the_apps_list_shows_the_updated_version(tmp_path: Path) -> None:
    """After an update of a folder NitroLearn-Setup installed, Windows' Apps list shows the new version; an
    entry for another folder, or none (the zip), is left alone."""
    iss = (PROJECT / "installer" / "NitroLearn.iss").read_text(encoding="utf-8")
    app_id = re.search(r"^AppId=\{(\{[0-9A-F-]+\})$", iss, re.MULTILINE)
    assert app_id is not None and update.APPS_LIST_KEY.endswith(f"\\{app_id[1]}_is1")
    entry = {"InstallLocation": str(tmp_path) + os.sep, "DisplayVersion": "1.0.0"}
    assert update.show_version_in_apps_list(tmp_path, "1.1.0", FakeRegistry(entry))
    assert entry["DisplayVersion"] == "1.1.0"
    other = {"InstallLocation": str(tmp_path / "elsewhere"), "DisplayVersion": "1.0.0"}
    assert not update.show_version_in_apps_list(tmp_path, "1.1.0", FakeRegistry(other))
    assert other["DisplayVersion"] == "1.0.0"
    assert not update.show_version_in_apps_list(tmp_path, "1.1.0", FakeRegistry(None))
