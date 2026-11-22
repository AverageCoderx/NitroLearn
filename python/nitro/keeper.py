"""Keeps a run training through crashes and computer restarts (`run.auto_restart`).

With it on, `nitro train` / `nitro resume` becomes a small keeper process: it runs the trainer as a child
(`nitro resume <run>`) and starts it again from the newest save when it ends unexpectedly, at most
`run.auto_restart_limit` times in a row without a new save in between. A stop the user asked for (the
dashboard's Stop, Ctrl+C) or a run that reached its --steps ends the keeper. While a keeper is armed a
start-up entry (Windows Startup folder, Linux XDG autostart) runs `nitro carry-on`, which starts the keeper
of every armed run of the folder again after the computer restarts. What happened is kept in the run's
keep_going.json, for the console and the dashboard's banner.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

from nitro.runs import RunFolder

STATE = "keep_going.json"  # in the run folder: armed or not, crashes in a row, what happened
KEEPER = "keeper.json"  # in the run folder while a keeper runs: its process id
USER_STOP = "user_stop"  # written by a kept trainer when the user asks it to stop
KEPT_ENV = "NITRO_KEPT"  # set for the trainer a keeper starts
WHY_ENV = "NITRO_KEPT_WHY"  # "reboot": a keeper started by `nitro carry-on`
DELAY_ENV = "NITRO_RESTART_DELAY"  # seconds before starting again (tests shorten it)
FILES = (STATE, KEEPER, USER_STOP)  # this computer's only: never packed (nitro pack)
RESTART_CODE = 75  # a kept trainer saved after an Apply: start it again now (not a crash)
DELAY = 30.0
_EVENTS = 20


def kept() -> bool:
    """Whether this process is a trainer started by a keeper."""
    return os.environ.get(KEPT_ENV) == "1"


def note_user_stop(folder: RunFolder) -> None:
    """A kept trainer was asked to stop: its keeper must not start it again."""
    if kept():
        with contextlib.suppress(OSError):
            (folder.root / USER_STOP).write_text("stop\n", encoding="utf-8")


def read_state(folder: RunFolder) -> dict[str, Any]:
    try:
        state = json.loads((folder.root / STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _write_state(folder: RunFolder, state: dict[str, Any]) -> None:
    partial = folder.root / (STATE + ".partial")
    partial.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(partial, folder.root / STATE)


def _event(state: dict[str, Any], kind: str, text: str) -> None:
    state["events"] = [*state.get("events", []), {"at": time.time(), "kind": kind, "text": text}][-_EVENTS:]


def alive(folder: RunFolder) -> bool:
    """Whether a keeper of this run is running (training, or waiting to start it again). A process of another
    program that took its number after a computer restart is not taken for it."""
    from nitro.dashboard.bots import _is_python, pid_alive

    try:
        pid = int(json.loads((folder.root / KEEPER).read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, AttributeError):
        return False
    try:
        return pid > 0 and pid_alive(pid) and _is_python(pid)
    except OSError:  # Windows refuses to look up a process number no process has
        return False


def _claim(folder: RunFolder) -> bool:
    marker = folder.root / KEEPER
    for _ in range(2):
        try:
            handle = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if alive(folder):
                return False
            marker.unlink(missing_ok=True)  # left by a keeper that ended (a power cut)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
        return True
    return False


def _newest_save(folder: RunFolder) -> int:
    complete = folder.complete_checkpoints()
    return int(complete[-1].name) if complete else -1


def _last_words(folder: RunFolder, since: float) -> str:
    """The last line the trainer wrote about why it stopped: the dashboard's console file when it was written
    during this start, else train.log."""
    from nitro.dashboard.jobs import CONSOLE

    for file in (folder.root / CONSOLE, folder.log):
        try:
            if file.stat().st_mtime < since:
                continue
            with file.open("rb") as handle:
                handle.seek(max(0, file.stat().st_size - 8192))
                text = handle.read().decode("utf-8", "replace")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        except OSError:
            continue
        if lines:
            return lines[-1][:300]
    return ""


def _delay() -> float:
    try:
        return max(0.0, float(os.environ.get(DELAY_ENV, DELAY)))
    except ValueError:
        return DELAY


def _child_command(folder: RunFolder, steps: int | None, then_train: bool) -> list[str]:
    command = [sys.executable, "-m", "nitro", "resume", folder.root.name, "--runs", str(folder.root.parent)]
    if steps is not None:
        command += ["--steps", str(steps)]
    if then_train:
        command.append("--then-train")
    return command


def keep(folder: RunFolder, steps: int | None, then_train: bool, limit: int) -> int:
    """Runs the trainer of `folder` until it ends well or is stopped, starting it again after each crash."""
    from nitro.dashboard.bots import STOP_REQUEST
    from nitro.dashboard.jobs import AFTER_PID

    if not _claim(folder):
        print(
            f"'{folder.root.name}' is already kept training by another process (run.auto_restart). Stop it "
            "first (the dashboard's Stop, or Ctrl+C in its window).",
            file=sys.stderr,
        )
        return 1
    (folder.root / STOP_REQUEST).unlink(missing_ok=True)  # older than this keeper; one from now on stops it
    asked = {"stop": False}

    def on_ctrl_c(signum: int, frame: FrameType | None) -> None:
        asked["stop"] = True  # the trainer has the same Ctrl+C and saves; it is not started again

    previous = signal.signal(signal.SIGINT, on_ctrl_c)
    state = read_state(folder)
    reboot = os.environ.pop(WHY_ENV, "") == "reboot"
    saved = _newest_save(folder)
    crashes = 0  # a computer restart without a new save since the keeper last looked counts as a crash
    if reboot and state.get("armed") and saved <= int(state.get("last_save", -1)):
        crashes = int(state.get("crashes", 0)) + 1
    state.update(armed=True, session=time.time(), crashes=crashes, last_save=saved, limit=limit)
    if reboot:
        _event(
            state, "reboot", "started again by itself after the computer restarted (or its window was closed)"
        )
    code, after = 1, dict[str, str]()
    try:
        if crashes > limit:
            _event(
                state,
                "gave_up",
                "stopped starting again by itself: the computer restarted (or its window was closed) "
                f"{crashes} times in a row while it trained, without a new save (run.auto_restart_limit = "
                f"{limit}). "
                "Fix that, then start it",
            )
            print(f"Not starting again by itself: {state['events'][-1]['text']}.", file=sys.stderr)
            return 1
        if reboot:
            print(
                "Training starts again by itself: the computer restarted while it trained (run.auto_restart)."
            )
        _write_state(folder, state)
        entry = add_startup_entry(folder.root.parent)
        if entry:
            print(f"Auto restart is on: {entry} starts it again after a computer restart.", flush=True)
        while True:
            (folder.root / USER_STOP).unlink(missing_ok=True)
            began = time.time()
            env = {**os.environ, KEPT_ENV: "1", **after}
            after = {}
            child = subprocess.Popen(_child_command(folder, steps, then_train), env=env)
            while True:
                try:
                    code = child.wait()
                    break
                except KeyboardInterrupt:  # (a handler set by Python itself before ours)
                    asked["stop"] = True
            if code == RESTART_CODE and not asked["stop"]:
                after = {AFTER_PID: str(child.pid)}  # the next start is an Apply's, as the trainer's own
                _event(state, "apply", "started again with the changes applied in the dashboard")
                _write_state(folder, state)
                continue
            if code == 0 or asked["stop"] or (folder.root / USER_STOP).exists():
                code = 0 if code == RESTART_CODE else code
                break
            saved = _newest_save(folder)
            progressed = saved > int(state.get("last_save", -1))
            state["crashes"] = 0 if progressed else int(state.get("crashes", 0)) + 1
            state["last_save"] = saved
            why = f"training stopped unexpectedly (exit code {code})"
            if words := _last_words(folder, began):
                why += f": {words}"
            if state["crashes"] > limit:
                _event(
                    state,
                    "gave_up",
                    f"stopped starting again by itself: {state['crashes']} crashes in a row without a new "
                    f"save (run.auto_restart_limit = {limit}). The last: {why}. Fix that, then start it",
                )
                print(f"\nNot starting again by itself: {state['events'][-1]['text']}.", file=sys.stderr)
                break
            delay = _delay()
            again = time.time() + delay
            _event(
                state,
                "crash",
                f"{why}; started again by itself at {time.strftime('%H:%M', time.localtime(again))}"
                + (
                    f" (restart {state['crashes']} of {limit} without a new save)"
                    if state["crashes"]
                    else " (it had saved since its last start)"
                ),
            )
            _write_state(folder, state)
            print(f"\nAuto restart: {why}. Starting again in {delay:.0f} s from the newest save.", flush=True)
            while time.time() < again and not asked["stop"]:
                if (folder.root / STOP_REQUEST).exists():
                    (folder.root / STOP_REQUEST).unlink(missing_ok=True)
                    asked["stop"] = True
                time.sleep(min(0.5, max(0.0, again - time.time())))
            if asked["stop"]:
                print("Stop asked for: not starting again.", flush=True)
                code = 0
                break
    finally:
        state["armed"] = False
        _write_state(folder, state)
        (folder.root / USER_STOP).unlink(missing_ok=True)
        (folder.root / KEEPER).unlink(missing_ok=True)
        if not armed_runs(folder.root.parent):
            remove_startup_entry(folder.root.parent)
        signal.signal(signal.SIGINT, previous)
    return code


def armed_runs(runs: Path) -> list[RunFolder]:
    """The runs of `runs` whose keeper was running (or running when the computer went off)."""
    if not runs.is_dir():
        return []
    found = []
    for path in sorted(runs.iterdir()):
        if (path / STATE).is_file() and read_state(RunFolder(path)).get("armed"):
            found.append(RunFolder(path))
    return found


def banner(folder: RunFolder) -> dict[str, Any] | None:
    """What the dashboard says about this run's restarts: the newest crash, computer restart or giving up
    since the keeper last started; None when there is nothing to say."""
    state = read_state(folder)
    events = [e for e in state.get("events", []) if float(e.get("at", 0)) >= float(state.get("session", 0))]
    shown = [e for e in events if e.get("kind") in ("crash", "reboot", "gave_up")]
    if not shown:
        return None
    last = shown[-1]
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(last["at"])))
    text = str(last["text"])
    return {"text": f"{text[:1].upper()}{text[1:]} ({when}).", "bad": last["kind"] == "gave_up"}


def _still_on(folder: RunFolder) -> bool:
    """Whether the run's settings still ask for run.auto_restart (True when they do not read: the trainer
    then says what is wrong)."""
    from nitro.config import ConfigError, load_config

    try:
        return load_config(folder.settings).run.auto_restart
    except (ConfigError, OSError):
        return True


def carry_on(runs: Path) -> int:
    """After a computer restart: starts the keeper of every armed run of `runs` that is not running."""
    from nitro.dashboard import bots
    from nitro.dashboard.jobs import spawn

    started = 0
    for folder in armed_runs(runs):
        if alive(folder) or bots.running(folder) is not None:
            continue
        if not _still_on(folder):  # turned off since its keeper was ended without a word (window closed)
            _write_state(folder, {**read_state(folder), "armed": False})
            continue
        spawn(
            ["resume", folder.root.name, "--runs", str(runs.resolve())],
            folder.root,
            project_root(),
            env={WHY_ENV: "reboot"},
        )
        print(f"Started '{folder.root.name}' again (it was training when the computer went off).")
        started += 1
    if not started and not armed_runs(runs):
        remove_startup_entry(runs)
        print("No bot needs starting again; the start-up entry is removed.")
    return 0


# ---------- the start-up entry ----------


def project_root() -> Path:
    """NitroLearn's folder: the one holding its Python environment (.venv), else the working folder."""
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix).parent
    return Path.cwd()


def _entry_name(runs: Path) -> str:
    from nitro.dashboard.winapp import CARRY_ON_ENTRY

    tag = hashlib.sha256(str(runs.resolve()).encode()).hexdigest()[:8]
    return f"{CARRY_ON_ENTRY} {tag}"


def startup_folder() -> Path | None:
    """Where this system starts programs at log-in for this user; None where not supported."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "Microsoft/Windows/Start Menu/Programs/Startup" if appdata else None
    if sys.platform.startswith("linux"):
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / "autostart"
    return None


def startup_entry(runs: Path) -> Path | None:
    folder = startup_folder()
    if folder is None:
        return None
    return folder / (_entry_name(runs) + (".lnk" if sys.platform == "win32" else ".desktop"))


def add_startup_entry(runs: Path) -> Path | None:
    """Makes the start-up entry for `runs` (recorded first in .tools/shortcuts.json, so uninstalling removes
    it); None when it exists already or cannot be made (said in the console)."""
    from nitro.dashboard import winapp

    entry = startup_entry(runs)
    if entry is None or entry.exists():
        return None
    root = project_root()
    winapp.record_shortcuts(root, [entry])
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            _windows_shortcut(entry, root, runs)
        else:
            command = f'"{sys.executable}" -m nitro carry-on --runs "{runs.resolve()}"'
            entry.write_text(
                "[Desktop Entry]\nType=Application\nName=NitroLearn: carry on training\n"
                f"Exec={command}\nPath={root}\nTerminal=false\nX-GNOME-Autostart-enabled=true\n",
                encoding="utf-8",
            )
    except (OSError, subprocess.SubprocessError) as err:
        winapp.forget_shortcut(root, entry)
        print(f"Auto restart: could not add the start-up entry ({err}); a computer restart is not covered.")
        return None
    return entry


def _windows_shortcut(entry: Path, root: Path, runs: Path) -> None:
    nitro_cmd = root / "nitro.cmd"
    target, arguments = (
        (str(nitro_cmd), f'carry-on --runs "{runs.resolve()}"')
        if nitro_cmd.is_file()
        else (sys.executable, f'-m nitro carry-on --runs "{runs.resolve()}"')
    )

    def quoted(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + quoted(str(entry)) + "); "
        f"$s.TargetPath = {quoted(target)}; $s.Arguments = {quoted(arguments)}; "
        f"$s.WorkingDirectory = {quoted(str(root))}; $s.WindowStyle = 7; "
        "$s.Description = 'NitroLearn: carry on training after a restart'; $s.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined, unused-ignore]
    )


def remove_startup_entry(runs: Path) -> None:
    from nitro.dashboard import winapp

    entry = startup_entry(runs)
    if entry is None:
        return
    with contextlib.suppress(OSError):
        entry.unlink(missing_ok=True)
    if not entry.exists():
        winapp.forget_shortcut(project_root(), entry)
