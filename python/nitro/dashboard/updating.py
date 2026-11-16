"""Update NitroLearn: the check when the dashboard opens (at most once every EVERY seconds, its answer kept in
runs/.dashboard/update_check.json, silent when GitHub cannot be reached; App settings can turn it off), the
check on App settings' button, the update as its own process (`nitro update`, its output in a console), and
closing the dashboard to start it again with the new version. The update itself is `nitro.update`.

On Windows the dashboard holds the compiled core while it is open, so there the update copies the code and
leaves the installer for when the window has closed: a window of its own runs it, then opens the dashboard
again."""

from __future__ import annotations

import contextlib
import functools
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from nitro import update
from nitro.dashboard import edits, winapp
from nitro.dashboard.jobs import CONSOLE, Jobs, read_console

UPDATE = ".update"  # the jobs list's name for an update
RESULT = "result.json"
CLOSE_SECONDS = 0.6  # the window closes this long after Restart, so the page gets its answer first
NOTICE = "update_check.json"  # in runs/.dashboard: when GitHub was asked, its answer, Not now's version
EVERY = 20 * 3600  # the check when the dashboard opens asks GitHub at most this often (seconds)
CHECK_PREF = "check_updates"  # App settings' switch for that check (on unless set to false)
_ASKING = threading.Lock()  # one check at a time, however many asks come
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


def ask_github() -> update.Release:
    """The newest release, as the check when the dashboard opens asks for it (tests answer for it here)."""
    return update.latest()


def kept_release(data: object) -> update.Release | None:
    """A release as the notice file keeps it, or None when the file holds none (or not one that is whole)."""
    if not isinstance(data, dict):
        return None
    values = [data.get(f.name) for f in fields(update.Release)]
    texts = [v for v in values if isinstance(v, str)]
    if len(texts) != len(values) or not texts[0]:
        return None
    version, tag, date, notes, page = texts
    page = page if page.startswith("https://github.com/") else ""
    return update.Release(version, tag, date, update.first_lines(notes), page)  # plain, whoever kept it


class Updating:
    """Mixed into `Api`: every public method is callable from the page."""

    _project: Path
    _runs: Path
    _jobs: Jobs
    _window: Any
    _prefs: Callable[[], dict[str, Any]]

    def _update_folder(self) -> Path:
        return self._runs / ".dashboard" / "update"

    @_safe
    def update_check(self) -> dict[str, Any]:
        """Asks GitHub whether a newer NitroLearn is out (App settings' button); the answer is kept for the
        Garage's notice too."""
        try:
            found = update.check(self._project)
        except update.UpdateError as err:
            return {"error": str(err)}
        self._keep_notice({**self._kept_notice(), "checked": time.time(), "release": asdict(found.release)})
        return {**found.view(), "finish_on_close": update.WINDOWS, "busy": self._update_busy()}

    # ---------- the check when the dashboard opens ----------
    def _notice_file(self) -> Path:
        return self._runs / ".dashboard" / NOTICE

    def _kept_notice(self) -> dict[str, Any]:
        try:
            data = json.loads(self._notice_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _keep_notice(self, data: dict[str, Any]) -> None:
        try:
            self._notice_file().parent.mkdir(parents=True, exist_ok=True)
            edits.save(self._notice_file(), json.dumps(data, indent=1))
        except OSError:
            pass  # asked again next time

    @_safe
    def update_notice(self) -> dict[str, Any]:
        """Whether a newer NitroLearn is out, for the Garage's notice and App settings' dot. The page asks
        when it opens and every hour; GitHub is asked only when the last ask is EVERY seconds old (or its time
        is in the future), and a failed ask says nothing: the answer kept from before stands. "state" is
        "off" (App settings turned the check off), "unknown" (no answer yet), or the release's "newer",
        "same" or "older"; "not_now" is whether Not now was pressed for this release."""
        if self._prefs().get(CHECK_PREF, True) is False:
            return {"state": "off"}
        kept = self._kept_notice()
        checked = kept.get("checked")
        now = time.time()
        due = not isinstance(checked, (int, float)) or not 0 <= now - checked < EVERY
        if due and _ASKING.acquire(blocking=False):
            try:
                kept = {**self._kept_notice(), "checked": now}
                # Offline, or GitHub said no: silent, asked again after EVERY seconds.
                with contextlib.suppress(update.UpdateError, http.client.HTTPException, OSError, ValueError):
                    kept["release"] = asdict(ask_github())
                self._keep_notice(kept)
            finally:
                _ASKING.release()
        release = kept_release(kept.get("release"))
        try:
            installed = update.installed_version(self._project)
        except update.UpdateError:
            return {"state": "unknown"}
        if release is None:
            return {"state": "unknown", "installed": installed}
        return {
            "state": update.compare(release.version, installed),
            "installed": installed,
            "release": asdict(release),
            "git": update.is_git(self._project),
            "not_now": kept.get("not_now") == release.version,
        }

    @_safe
    def update_not_now(self, version: str) -> dict[str, Any]:
        """Not now on the Garage's notice: no notice for `version` again, only for a later one."""
        self._keep_notice({**self._kept_notice(), "not_now": str(version)})
        return {"not_now": str(version)}

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
