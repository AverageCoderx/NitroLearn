"""The dashboard as an application of its own: its icon, its own group on the Windows taskbar, one window at
a time per runs folder, console-free helper processes, and the record of the shortcuts made outside
NitroLearn's folder (tools/app_shortcuts.ps1).

The Windows calls are made through `_dll`, and every one of them is optional: when one fails (another
system, an older Windows or pywebview) the dashboard opens as before.

    python -m nitro.dashboard.winapp shortcuts --root <NitroLearn folder> --record <file.lnk> ...
    python -m nitro.dashboard.winapp shortcuts --root <NitroLearn folder> --remove
    pythonw -m nitro.dashboard.winapp open

`open` is how the shortcuts and the setup's Finish page start the dashboard: with pythonw.exe, so no console
window opens at all, in NitroLearn's folder with its tools, caches and temporary files kept inside it (as
nitro.cmd and tools/env.ps1 set them).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import inspect
import json
import os
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from nitro import brand, folder_env

APP_ID = "NitroLearn.Dashboard"  # also on the shortcuts (tools/app_shortcuts.ps1): keep them equal
WINDOWS = sys.platform == "win32"
FOLDER = ".dashboard"  # in the runs folder, next to the dashboard's other files
LOCK = "window.lock"  # held by the open dashboard; holds its process id and window handle
SHOW_REQUEST = "show.request"  # a second start asks the open dashboard to come to the front
POLL_SECONDS = 1.0
SHORTCUTS = Path(".tools") / "shortcuts.json"  # in NitroLearn's folder
ROOT = Path(__file__).resolve().parents[3]  # NitroLearn's folder (python/nitro/dashboard/winapp.py)
_LOCK_AT = 1 << 20  # the byte locked on Windows, past the record so others can still read it
_SW_RESTORE = 9


def icon_file() -> Path:
    """The app icon for this system: the .ico on Windows (every size inside), the PNG elsewhere."""
    return brand.HERE / ("icon.ico" if WINDOWS else "icon.png")


def _dll(name: str) -> Any:
    """A Windows system library (`user32`, `shell32`); AttributeError or OSError on another system."""
    return ctypes.WinDLL(name)  # type: ignore[attr-defined, unused-ignore]


def set_app_id(app_id: str = APP_ID) -> bool:
    """Gives this process its own AppUserModelID, so Windows groups its window as NitroLearn (its own
    taskbar button, icon and pin) rather than as Python. Must come before the window is made. True when
    set."""
    if not WINDOWS:
        return False
    try:
        return int(_dll("shell32").SetCurrentProcessExplicitAppUserModelID(app_id)) == 0
    except Exception:
        return False


def console_python(executable: str = sys.executable) -> str:
    """The Python to start helper processes with. pythonw.exe (the windowless dashboard) has no console, so
    every console program its children start would open a console window of its own; python.exe started
    with CREATE_NO_WINDOW instead gets a hidden console that its own children share."""
    path = Path(executable)
    if path.name.lower() == "pythonw.exe":
        twin = path.with_name("python.exe")
        if twin.is_file():
            return str(twin)
    return executable


class Instance:
    """This process's hold on the dashboard of one runs folder, kept while its window is open. The hold is
    a lock the system drops when the process ends, however it ends, so a crashed dashboard never blocks the
    next start; the file's text says which process and window hold it."""

    def __init__(self, folder: Path, fd: int | None) -> None:
        self.folder = folder
        self._fd = fd

    @property
    def held(self) -> bool:
        return self._fd is not None

    def note(self, window: int) -> None:
        """Writes this process's id and its window's handle into the lock file, for a second start."""
        if self._fd is None:
            return
        text = json.dumps({"pid": os.getpid(), "window": window}).encode()
        with contextlib.suppress(OSError):
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.ftruncate(self._fd, 0)
            os.write(self._fd, text)

    def show_requested(self) -> bool:
        """Whether a second start asked for the window since the last call (the request is taken)."""
        try:
            (self.folder / SHOW_REQUEST).unlink()
        except OSError:
            return False
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        if sys.platform == "win32":
            import msvcrt

            with contextlib.suppress(OSError):
                os.lseek(fd, _LOCK_AT, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


def _try_lock(fd: int) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, _LOCK_AT, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def claim(runs: Path) -> Instance | None:
    """Takes the dashboard of `runs` for this process: None when another open dashboard has it. Where the
    lock cannot be made at all (a read-only folder) the dashboard opens anyway, without the guard."""
    folder = runs / FOLDER
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd = os.open(folder / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return Instance(folder, None)
    if not _try_lock(fd):
        os.close(fd)
        return None
    instance = Instance(folder, fd)
    with contextlib.suppress(OSError):
        (folder / SHOW_REQUEST).unlink()  # one left from before is no request to this window
    instance.note(0)
    return instance


def holder(runs: Path) -> dict[str, int]:
    """What the lock file says of the open dashboard: its process id and window handle (0 when unknown)."""
    try:
        data = json.loads((runs / FOLDER / LOCK).read_text(encoding="utf-8"))
        return {key: int(data.get(key) or 0) for key in ("pid", "window")}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"pid": 0, "window": 0}


def summon(runs: Path) -> None:
    """From a second start: asks the open dashboard of `runs` to come to the front. On Windows this process
    (the one the user just started) may bring a window forward, so it does that itself and lets the open
    dashboard do the same; everywhere, the open dashboard sees the request within a second."""
    first = holder(runs)
    with contextlib.suppress(OSError):
        (runs / FOLDER / SHOW_REQUEST).write_text(str(os.getpid()), encoding="utf-8")
    if WINDOWS:
        with contextlib.suppress(Exception):
            user32 = _dll("user32")
            if first["pid"]:
                user32.AllowSetForegroundWindow(first["pid"])
            _forward(user32, first["window"])


def _forward(user32: Any, window: int) -> None:
    """Restores the window `window` if minimized and puts it in front."""
    if window and user32.IsWindow(window):
        if user32.IsIconic(window):
            user32.ShowWindow(window, _SW_RESTORE)
        user32.SetForegroundWindow(window)


def dress(window: Any, instance: Instance) -> None:
    """Hooks the pywebview window up: on Windows its form gets the app icon and its handle is noted for a
    second start; once shown, it comes to the front whenever a second start asks."""
    events = getattr(window, "events", None)
    if events is None:
        return
    minimized = threading.Event()
    closed = threading.Event()

    def before_show() -> None:
        native = getattr(window, "native", None)
        if not WINDOWS or native is None:
            return
        with contextlib.suppress(Exception):
            from System.Drawing import Icon  # type: ignore[import-not-found, unused-ignore]

            native.Icon = Icon(str(icon_file()))
        with contextlib.suppress(Exception):
            instance.note(int(native.Handle.ToInt64()))

    def forward() -> None:
        handle = holder(instance.folder.parent)["window"] if WINDOWS else 0
        if handle:
            with contextlib.suppress(Exception):
                _forward(_dll("user32"), handle)
                return
        with contextlib.suppress(Exception):
            if minimized.is_set():
                window.restore()
            window.show()

    def watch() -> None:
        while not closed.wait(POLL_SECONDS):
            if instance.show_requested():
                forward()

    hooks: dict[str, Callable[[], None]] = {
        "before_show": before_show,
        "minimized": minimized.set,
        "restored": minimized.clear,
        "maximized": minimized.clear,
        "closed": closed.set,
        "shown": lambda: threading.Thread(target=watch, name="dashboard-forward", daemon=True).start(),
    }
    for name, hook in hooks.items():
        event = getattr(events, name, None)
        if event is not None:
            with contextlib.suppress(Exception):
                event += hook


def start(webview: Any, debug: bool) -> None:
    """Runs pywebview's window loop with the app icon, where this pywebview takes one."""
    options: dict[str, Any] = {"debug": debug}
    try:
        takes_icon = "icon" in inspect.signature(webview.start).parameters
    except (TypeError, ValueError):
        takes_icon = False
    if takes_icon and icon_file().is_file():
        options["icon"] = str(icon_file())
    webview.start(**options)


def record_shortcuts(root: Path, paths: Sequence[Path]) -> list[str]:
    """Adds `paths` (shortcut files about to be made outside `root`) to the record in `root`; the record."""
    file = root / SHORTCUTS
    known = _recorded(file)
    for path in paths:
        text = str(Path(path).absolute())
        if text not in known:
            known.append(text)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"shortcuts": known}, indent=2) + "\n", encoding="utf-8")
    return known


def remove_shortcuts(root: Path) -> tuple[list[str], list[str]]:
    """Deletes the shortcut files recorded in `root` and their record: (removed, could not be removed). Only
    .lnk files are deleted; one already gone is dropped from the record without being reported as removed.
    What could not be removed stays recorded."""
    file = root / SHORTCUTS
    removed: list[str] = []
    kept: list[str] = []
    for text in _recorded(file):
        path = Path(text)
        if path.suffix.lower() != ".lnk":
            continue  # not a shortcut: never deleted, and dropped from the record
        try:
            if path.exists():
                path.unlink()
                removed.append(text)
        except OSError:
            kept.append(text)
    with contextlib.suppress(OSError):
        if kept:
            file.write_text(json.dumps({"shortcuts": kept}, indent=2) + "\n", encoding="utf-8")
        else:
            file.unlink(missing_ok=True)
    return removed, kept


def _recorded(file: Path) -> list[str]:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        return [str(item) for item in data.get("shortcuts", []) if isinstance(item, str)]
    except (OSError, ValueError, AttributeError):
        return []


def open_dashboard(root: Path = ROOT, start: Callable[[Sequence[str]], int] | None = None) -> int:
    """The dashboard of `root`'s runs folder, started as `nitro dashboard` is from nitro.cmd: in `root`, with
    its folder environment. Restarting after an update starts it the same way (sys.argv says how)."""
    folder_env.apply(root)
    os.chdir(root)
    sys.argv[1:] = ["dashboard"]
    if start is not None:
        return start(["dashboard"])
    from nitro import cli

    return cli.main(["dashboard"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nitro.dashboard.winapp")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("open", help="open the dashboard without a console window (pythonw)")
    shortcuts = commands.add_parser("shortcuts", help="the record of the shortcuts made outside the folder")
    shortcuts.add_argument("--root", type=Path, required=True, help="NitroLearn's folder")
    what = shortcuts.add_mutually_exclusive_group(required=True)
    what.add_argument("--record", type=Path, nargs="+", metavar="LNK", help="shortcut files about to be made")
    what.add_argument("--remove", action="store_true", help="delete the recorded shortcuts")
    args = parser.parse_args(argv)
    if args.command == "open":
        return open_dashboard()
    if args.record:
        record_shortcuts(args.root, args.record)
        return 0
    removed, kept = remove_shortcuts(args.root)
    for text in removed:
        print(f"Removed {text}")
    for text in kept:
        print(f"Could not remove {text}: delete it by hand.", file=sys.stderr)
    return 1 if kept else 0


if __name__ == "__main__":
    sys.exit(main())
