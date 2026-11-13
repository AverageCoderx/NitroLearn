"""Update NitroLearn, on the App settings page: checking for a newer release only when asked, the update as
its own process (`nitro update`, its output in a console), and closing the dashboard to start it again with
the new version. The update itself is `nitro.update`.

On Windows the dashboard holds the compiled core while it is open, so there the update copies the code and
leaves the installer for when the window has closed: a window of its own runs it, then opens the dashboard
again."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nitro import update
from nitro.dashboard import winapp
from nitro.dashboard.jobs import CONSOLE, Jobs, read_console

UPDATE = ".update"  # the jobs list's name for an update
RESULT = "result.json"
CLOSE_SECONDS = 0.6  # the window closes this long after Restart, so the page gets its answer first
_WHAT = {".build": "the build of NitroLearn", ".convert": "the replay conversion"}


def core_stale(project: Path) -> bool:
    """Whether the compiled core was built from other code than the folder's (`nitro.build_info`, in a
    process of its own: this one's answer is kept from its start)."""
    flags = subprocess.CREATE_NO_WINDOW if update.WINDOWS else 0  # type: ignore[attr-defined, unused-ignore]
    try:
        done = subprocess.run(
            [winapp.console_python(), "-m", "nitro.build_info"],
            cwd=project,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode != 0


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """`nitro.dashboard.api._safe`, looked up when called: that module loads this one."""

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from nitro.dashboard.api import _safe as safe

        return safe(method)(*args, **kwargs)

    return wrapped


class Updating:
    """Mixed into `Api`: every public method is callable from the page."""

    _project: Path
    _runs: Path
    _jobs: Jobs
    _window: Any

    def _update_folder(self) -> Path:
        return self._runs / ".dashboard" / "update"

    @_safe
    def update_check(self) -> dict[str, Any]:
        """Asks GitHub whether a newer NitroLearn is out (only when the page's button is pressed)."""
        try:
            found = update.check(self._project)
        except update.UpdateError as err:
            return {"error": str(err)}
        return {**found.view(), "finish_on_close": update.WINDOWS, "busy": self._update_busy()}

    def _update_busy(self) -> str:
        """Why an update cannot start now: a bot trains, or a command started here still runs."""
        if problem := update.busy(self._runs):
            return problem
        running = [job for job in self._jobs.running() if job.bot != UPDATE]
        if running:
            job = running[0]
            what = _WHAT.get(job.bot, f"{job.bot}'s {job.kind}")
            return f"Wait until {what} is done (or stop it) before updating."
        return ""

    @_safe
    def update_start(self) -> dict[str, Any]:
        """Starts `nitro update` for the release the check found; refused while anything trains or runs."""
        job = self._jobs.get(UPDATE, "update")
        if job is not None and not job.done:
            return {"error": "The update is already running."}
        if problem := self._update_busy():
            return {"error": problem}
        folder = self._update_folder()
        folder.mkdir(parents=True, exist_ok=True)
        for name in (CONSOLE, RESULT):
            (folder / name).unlink(missing_ok=True)
        installer = "later" if update.WINDOWS else "now"
        args = ["update", "--yes", "--runs", str(self._runs), "--installer", installer]
        self._jobs.start(UPDATE, "update", [*args, "--result", str(folder / RESULT)], folder)
        return {"started": True}

    @_safe
    def update_state(self, after: int = 0) -> dict[str, Any]:
        """How the update goes: none, running, done or failed, what it printed after byte `after`, and when
        it has ended what it did (old and new version, the user's files it replaced, where their copies
        are, whether the installer is still to run)."""
        job = self._jobs.get(UPDATE, "update")
        text, end = read_console(self._update_folder(), int(after))
        if job is None:
            # The window closed before the installer ran (Windows): the update is still to be finished.
            result = self._update_result()
            if result.get("installer") == "later" and core_stale(self._project):
                return {"state": "done", "text": text, "next": end, "result": result, "unfinished": True}
            return {"state": "none", "text": text, "next": end}
        if not job.done:
            return {"state": "running", "text": text, "next": end}
        result = self._update_result()
        if job.code != 0 and not result.get("error"):
            lines = [line for line in text.splitlines() if line.strip()]
            result["error"] = lines[-1] if lines else f"It stopped with code {job.code}."
        failed = bool(result.get("error")) or result.get("installer") == "failed"
        return {"state": "failed" if failed else "done", "text": text, "next": end, "result": result}

    def _update_result(self) -> dict[str, Any]:
        """What the last update wrote of itself (`nitro update --result`), or {}."""
        try:
            result = json.loads((self._update_folder() / RESULT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return result if isinstance(result, dict) else {}

    @_safe
    def update_restart(self) -> dict[str, Any]:
        """Closes the dashboard and starts it again the way it was started. On Windows a window of its own
        first runs the installer (when the update left it for now). When this process is not a dashboard
        started by `nitro dashboard`, it only says to close and open NitroLearn again."""
        manual = {"manual": "Close NitroLearn and open it again to use the new version."}
        reopen = update.reopen_command(sys.argv[1:])
        if self._window is None:
            return manual
        pid, cwd = os.getpid(), Path.cwd()
        if update.WINDOWS:
            state = self.update_state(0)
            pending = (state.get("result") or {}).get("installer") == "later"
            steps = (
                update.installer_command(self._project, dev=update.developer_install()) if pending else None
            )
            update.start_detached(
                update.finisher_command(pid, steps, reopen, cwd), self._project, console=True
            )
        elif reopen is None:
            return manual
        else:
            update.start_detached(update.relaunch_command(pid, reopen), cwd, console=False)
        threading.Timer(CLOSE_SECONDS, self._window.destroy).start()
        return {"closing": True}
