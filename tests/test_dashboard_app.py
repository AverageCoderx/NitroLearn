"""The dashboard as an application of its own: the app icon, its AppUserModelID set before the window (on a
Windows faked here), one window per runs folder (two real processes), the shortcut record the uninstaller
uses, and console-free helper processes."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import nitro
from nitro.dashboard import app, winapp

ROOT = Path(__file__).parents[1]
BRAND = ROOT / "python" / "nitro" / "brand"
SIZES = {16, 24, 32, 48, 64, 128, 256}
PYTHON_DIR = str(Path(nitro.__file__).resolve().parents[1])

# Holds the dashboard of the runs folder in argv[1], says so with its own process id (on Windows a venv's
# python.exe starts the real Python as a child, so the started process's id is not the holder's), says when
# it is asked to come forward, and ends at once, without letting go, when a line comes in.
HOLDER = """
import os, sys, threading, time
from pathlib import Path
from nitro.dashboard import winapp
instance = winapp.claim(Path(sys.argv[1]))
print("held" if instance is not None else "busy", os.getpid(), flush=True)

def end_when_told():
    sys.stdin.readline()
    os._exit(0)  # as a crash: the dashboard never lets go of the lock itself

threading.Thread(target=end_when_told, daemon=True).start()
for _ in range(600):
    if instance is not None and instance.show_requested():
        print("asked", flush=True)
        break
    time.sleep(0.05)
threading.Event().wait()
"""


# ---- the icon


def _ico_entries(data: bytes) -> dict[int, bytes]:
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind) == (0, 1)
    images = {}
    for i in range(count):
        width, height, _, _, planes, bits, size, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        assert width == height and (planes, bits) == (1, 32)
        images[width or 256] = data[offset : offset + size]
    return images


def test_icon_holds_every_size() -> None:
    images = _ico_entries((BRAND / "icon.ico").read_bytes())
    assert set(images) == SIZES
    for side, blob in images.items():
        if blob.startswith(b"\x89PNG\r\n\x1a\n"):
            assert struct.unpack(">II", blob[16:24]) == (side, side)
        else:
            header, width, height, _, bits = struct.unpack_from("<IiiHH", blob, 0)
            assert (header, width, height, bits) == (40, side, 2 * side, 32)
            mask_row = (side + 31) // 32 * 4
            assert len(blob) == 40 + side * side * 4 + mask_row * side
    png = (BRAND / "icon.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and struct.unpack(">II", png[16:24]) == (256, 256)


def test_icon_packing() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import make_icon
    finally:
        sys.path.remove(str(ROOT / "tools"))
    red, clear = bytes((255, 0, 0, 255)), bytes(4)
    images = {16: red * 255 + clear, 256: red * (256 * 256)}
    packed = _ico_entries(make_icon.ico(images))
    assert set(packed) == {16, 256} and packed[256].startswith(b"\x89PNG")
    small = packed[16]
    first_row = small[40 : 40 + 64]  # the bottom row comes first: its last pixel is the clear one
    assert first_row[:4] == bytes((0, 0, 255, 255)) and first_row[-4:] == clear
    mask = small[40 + 16 * 16 * 4 :]
    assert mask[:4] == bytes((0, 1, 0, 0))  # only the clear pixel (x = 15) is masked


# ---- the window: a fake pywebview, and a faked Windows


class FakeEvent:
    def __init__(self) -> None:
        self.hooks: list[Callable[[], Any]] = []

    def __iadd__(self, hook: Callable[[], Any]) -> FakeEvent:
        self.hooks.append(hook)
        return self

    def fire(self) -> None:
        for hook in self.hooks:
            hook()


class FakeWindow:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.events = types.SimpleNamespace(
            **{name: FakeEvent() for name in ("before_show", "shown", "closed", "minimized", "restored")}
        )
        handle = types.SimpleNamespace(ToInt64=lambda: 4242)
        self.native = types.SimpleNamespace(Handle=handle)

    def show(self) -> None:
        self.calls.append("show")

    def restore(self) -> None:
        self.calls.append("restore")


def fake_webview(calls: list[str], with_icon: bool = True) -> Any:
    module = types.ModuleType("webview")
    windows: list[FakeWindow] = []

    def create_window(title: str, **_: Any) -> FakeWindow:
        calls.append(f"create_window {title}")
        windows.append(FakeWindow(calls))
        return windows[-1]

    def start_with_icon(debug: bool = False, icon: str | None = None) -> None:
        calls.append(f"start icon={Path(icon).name if icon else None}")
        windows[-1].events.before_show.fire()
        windows[-1].events.closed.fire()

    def start_without_icon(debug: bool = False) -> None:
        calls.append("start")

    module.create_window = create_window  # type: ignore[attr-defined]
    module.start = start_with_icon if with_icon else start_without_icon  # type: ignore[attr-defined]
    module.windows = windows  # type: ignore[attr-defined]
    return module


class FakeUser32:
    def __init__(self, calls: list[str], iconic: bool = True) -> None:
        self.calls = calls
        self.iconic = iconic

    def SetCurrentProcessExplicitAppUserModelID(self, app_id: str) -> int:
        self.calls.append(f"app id {app_id}")
        return 0

    def AllowSetForegroundWindow(self, pid: int) -> int:
        self.calls.append(f"allow {pid}")
        return 1

    def IsWindow(self, window: int) -> int:
        return 1

    def IsIconic(self, window: int) -> int:
        return int(self.iconic)

    def ShowWindow(self, window: int, how: int) -> int:
        self.calls.append(f"restore {window} {how}")
        return 1

    def SetForegroundWindow(self, window: int) -> int:
        self.calls.append(f"front {window}")
        return 1


def fake_windows(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    library = FakeUser32(calls)
    monkeypatch.setattr(winapp, "WINDOWS", True)
    monkeypatch.setattr(winapp, "_dll", lambda name: library)


def test_app_id_is_set_before_the_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    fake_windows(monkeypatch, calls)
    monkeypatch.setitem(sys.modules, "webview", fake_webview(calls))
    runs = tmp_path / "runs"
    assert app.run(ROOT, runs) == 0
    assert calls[:3] == [f"app id {winapp.APP_ID}", f"create_window {app.TITLE}", "start icon=icon.ico"]
    assert winapp.holder(runs) == {"pid": os.getpid(), "window": 4242}  # noted from the form
    assert winapp.claim(runs) is not None  # the closed window let go


def test_dashboard_opens_elsewhere_with_or_without_the_icon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(winapp, "WINDOWS", False)
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "webview", fake_webview(calls))
    assert app.run(ROOT, tmp_path / "a") == 0
    assert calls == [f"create_window {app.TITLE}", "start icon=icon.png"]
    calls.clear()
    monkeypatch.setitem(sys.modules, "webview", fake_webview(calls, with_icon=False))  # an older pywebview
    assert app.run(ROOT, tmp_path / "b") == 0
    assert calls == [f"create_window {app.TITLE}", "start"]


# ---- one window at a time


class Holder:
    """A second process holding the dashboard of a runs folder (HOLDER), and its own process id."""

    def __init__(self, runs: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", HOLDER, str(runs)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": PYTHON_DIR},
        )
        said = self.readline().split()
        assert said[:1] == ["held"]
        self.pid = int(said[1])

    def readline(self) -> str:
        assert self.process.stdout is not None
        return self.process.stdout.readline().strip()

    def end(self) -> None:
        """Ends it the way a crash would (no release), and waits until it has ended."""
        if self.process.poll() is None:
            assert self.process.stdin is not None
            try:
                self.process.stdin.write("end\n")
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


def test_second_start_asks_the_first_forward(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = Holder(runs)
    try:
        assert winapp.claim(runs) is None
        assert winapp.holder(runs)["pid"] == first.pid
        calls: list[str] = []
        monkeypatch.setitem(sys.modules, "webview", fake_webview(calls))
        assert app.run(ROOT, runs) == 0  # the second start: no window of its own
        assert calls == []
        assert first.readline() == "asked"
        assert winapp.claim(tmp_path / "other runs") is not None  # another runs folder is another dashboard
    finally:
        first.end()


def test_a_dead_dashboards_lock_is_taken_over(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = Holder(runs)
    first.end()  # no chance to clean up
    assert winapp.holder(runs)["pid"] == first.pid  # its record is still there
    instance = winapp.claim(runs)
    assert instance is not None and winapp.holder(runs)["pid"] == os.getpid()
    instance.release()
    # A lock file left by an older start, with a process id that is long gone, blocks nothing either.
    (runs / winapp.FOLDER / winapp.LOCK).write_text('{"pid": 999999, "window": 0}', encoding="utf-8")
    again = winapp.claim(runs)
    assert again is not None
    again.release()


def test_second_start_on_windows_brings_the_window_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    first = Holder(runs)
    try:
        lock = runs / winapp.FOLDER / winapp.LOCK
        lock.write_text(json.dumps({"pid": first.pid, "window": 77}), encoding="utf-8")
        calls: list[str] = []
        fake_windows(monkeypatch, calls)
        winapp.summon(runs)
        assert calls == [f"allow {first.pid}", "restore 77 9", "front 77"]
        # The open dashboard takes the request (and removes its file) as soon as it sees it: wait for that
        # rather than look for the file, which may already be gone on a busy computer.
        assert first.readline() == "asked"
    finally:
        first.end()


def test_open_window_comes_forward_when_asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(winapp, "WINDOWS", False)
    monkeypatch.setattr(winapp, "POLL_SECONDS", 0.02)
    runs = tmp_path / "runs"
    instance = winapp.claim(runs)
    assert instance is not None
    calls: list[str] = []
    window = FakeWindow(calls)
    winapp.dress(window, instance)
    try:
        window.events.shown.fire()
        window.events.minimized.fire()
        winapp.summon(runs)
        deadline = time.monotonic() + 5
        while calls != ["restore", "show"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == ["restore", "show"]
    finally:
        window.events.closed.fire()
        instance.release()


# ---- shortcuts and helper processes


def test_shortcut_record_and_removal(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    start, desktop = tmp_path / "Start" / "NitroLearn.lnk", tmp_path / "Desktop" / "NitroLearn.lnk"
    for link in (start, desktop):
        link.parent.mkdir()
        link.write_bytes(b"lnk")
    assert winapp.main(["shortcuts", "--root", str(root), "--record", str(start)]) == 0
    assert winapp.main(["shortcuts", "--root", str(root), "--record", str(start), str(desktop)]) == 0
    record = root / winapp.SHORTCUTS
    assert json.loads(record.read_text(encoding="utf-8")) == {"shortcuts": [str(start), str(desktop)]}
    desktop.unlink()  # already deleted by the user: dropped from the record, not reported as removed
    assert winapp.remove_shortcuts(root) == ([str(start)], [])
    assert not start.exists() and not record.exists()
    assert winapp.main(["shortcuts", "--root", str(root), "--remove"]) == 0  # nothing recorded: nothing to do


def test_shortcut_removal_deletes_only_shortcuts(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    keep = tmp_path / "notes.txt"
    keep.write_text("mine", encoding="utf-8")
    stuck = tmp_path / "folder.lnk"
    stuck.mkdir()  # cannot be deleted as a file
    winapp.record_shortcuts(root, [keep, stuck])
    removed, kept = winapp.remove_shortcuts(root)
    assert removed == [] and kept == [str(stuck)]
    assert keep.read_text(encoding="utf-8") == "mine"
    assert json.loads((root / winapp.SHORTCUTS).read_text(encoding="utf-8")) == {"shortcuts": [str(stuck)]}


def test_scripts_agree_with_the_app() -> None:
    script = (ROOT / "tools" / "app_shortcuts.ps1").read_text(encoding="utf-8")
    assert script.isascii(), "Windows PowerShell 5.1 reads a file without a byte-order mark as ANSI"
    assert f"$AppId = '{winapp.APP_ID}'" in script
    assert "python\\nitro\\brand\\icon.ico" in script
    # Windowless: pythonw.exe itself is the target, never cmd.exe with a console window behind the app.
    assert "::Make($Link, $Windowless, '-m nitro.dashboard.winapp open', $Root, $icon" in script
    assert "cmd.exe" not in script
    # Add-Type's compiler cannot write into a TEMP whose path has letters outside the ANSI code page: it
    # compiles in a plain-letter TEMP, and TEMP is put back.
    add_type = script.index("Add-Type -TypeDefinition")
    assert script.rindex("$env:TEMP = Get-PlainTemp", 0, add_type) > script.rindex("$saved = @(", 0, add_type)
    assert "$env:TEMP, $env:TMP = $saved" in script[add_type:]
    assert "$Windowless = Join-Path $Root '.venv\\Scripts\\pythonw.exe'" in script
    launcher = (ROOT / "Shortcuts" / "shortcuts.ps1").read_text(encoding="utf-8")
    assert "@('-m', 'nitro.dashboard.winapp', 'open')" in launcher, "the Shortcuts folder: the same"
    uninstall = (ROOT / "Uninstall NitroLearn.cmd").read_text(encoding="utf-8")
    assert 'tools\\app_shortcuts.ps1" -Remove' in uninstall


def test_the_windowless_start_keeps_the_tools_inside_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pythonw -m nitro.dashboard.winapp open` (the shortcuts, the setup's Finish page): the dashboard of the
    folder's runs, with the folder's environment as nitro.cmd sets it, and an update's restart starts it the
    same way."""
    from nitro import folder_env

    monkeypatch.setattr(os, "environ", {"PATH": "elsewhere"})
    monkeypatch.setattr(sys, "argv", ["winapp.py", "open"])
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "NitroLearn"
    root.mkdir()
    started: list[list[str]] = []
    assert winapp.open_dashboard(root, lambda argv: started.append(list(argv)) or 0) == 0
    assert started == [["dashboard"]] and Path.cwd() == root and sys.argv[1:] == ["dashboard"]
    windows = sys.platform == "win32"
    for name, value in folder_env.variables(root, windows).items():
        assert os.environ[name] == value, name
    first = [str(root / ".tools" / "cargo" / "bin"), str(root / ".tools" / "uv")]
    assert os.environ["PATH"].split(os.pathsep) == [*first, "elsewhere"]
    assert (root / ".tmp").is_dir() and (root / ".cache" / "pip").is_dir()
    assert winapp.ROOT == ROOT and winapp.main.__module__ == "nitro.dashboard.winapp"


def test_helpers_start_with_the_console_python(tmp_path: Path) -> None:
    windowless = tmp_path / "pythonw.exe"
    windowless.write_bytes(b"")
    assert winapp.console_python(str(windowless)) == str(windowless)  # no twin: kept
    (tmp_path / "python.exe").write_bytes(b"")
    assert winapp.console_python(str(windowless)) == str(tmp_path / "python.exe")
    assert winapp.console_python(sys.executable) == sys.executable
