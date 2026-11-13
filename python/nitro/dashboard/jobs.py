"""The `nitro` commands the dashboard starts for a bot (train, resume, export, watch ...). Each runs as its
own process, writing to a console file in the bot's folder, so closing the window never stops training (and
never breaks a pipe a running command writes to); the dashboard reads the file back."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from nitro.dashboard import winapp

CONSOLE = "console.log"  # in the bot's folder: the newest dashboard-started command's output
# Each command started here gets a token in this variable; a trainer writes it into running.json, so its
# start is known though its pid is not the started one (Windows' venv python.exe starts the real Python as
# a child).
JOB_TOKEN = "NITRO_JOB"
AFTER_PID = "NITRO_AFTER_PID"  # a trainer started by another waits for that process to end first


@dataclass
class Job:
    bot: str
    kind: str  # train, check, export, watch, pack, build, convert
    command: list[str]
    console: Path
    process: subprocess.Popen[bytes]
    started: float = field(default_factory=time.time)
    seen_running: bool = False  # a trainer whose running.json was seen: past its start (only ending now)
    token: str = ""  # the JOB_TOKEN it was started with
    stopped: bool = False  # ended from the window (Stop on a replay conversion): not a failure, any code

    @property
    def done(self) -> bool:
        return self.process.poll() is not None

    @property
    def code(self) -> int | None:
        return self.process.poll()


class Jobs:
    """The commands started from this window, newest per bot and kind."""

    def __init__(self, project: Path) -> None:
        self.project = project
        self._jobs: dict[tuple[str, str], Job] = {}
        self._lock = threading.Lock()

    def start(self, bot: str, kind: str, args: list[str], folder: Path) -> Job:
        """Runs `nitro <args>` for `bot`, its output appended to `folder`/console.log."""
        token = uuid.uuid4().hex
        process = spawn(args, folder, self.project, token)
        job = Job(bot, kind, list(process.args), folder / CONSOLE, process, token=token)  # type: ignore[arg-type]
        with self._lock:
            self._jobs[(bot, kind)] = job
        return job

    def get(self, bot: str, kind: str) -> Job | None:
        with self._lock:
            return self._jobs.get((bot, kind))

    def active(self, bot: str) -> list[Job]:
        with self._lock:
            return [job for (name, _), job in self._jobs.items() if name == bot and not job.done]

    def running(self) -> list[Job]:
        """Every command started here that has not ended."""
        with self._lock:
            return [job for job in self._jobs.values() if not job.done]


def spawn(
    args: list[str], folder: Path, cwd: Path, token: str = "", env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Starts `nitro <args>` on its own, its output appended to `folder`/console.log: from the dashboard, and
    by a trainer starting itself again after an Apply (nitro.cli._run)."""
    folder.mkdir(parents=True, exist_ok=True)
    command = [winapp.console_python(), "-u", "-m", "nitro", *args]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        JOB_TOKEN: token,
        **(env or {}),
    }
    flags = 0
    if sys.platform == "win32":  # no console window of its own; it lives on when this window closes
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined, unused-ignore]
    with (folder / CONSOLE).open("ab") as out:
        out.write(f"\n$ nitro {' '.join(args)}   ({time.strftime('%Y-%m-%d %H:%M:%S')})\n".encode())
        out.flush()
        return subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=flags,
        )


def read_console(folder: Path, after: int, limit: int = 200_000) -> tuple[str, int]:
    """What the bot's console file holds after byte `after` (at most `limit` bytes, the newest), and where
    it now ends. A file made anew (shorter than `after`) is read from its start. Windows' line ends (a
    program's console output written there has them) come back as plain new lines."""
    file = folder / CONSOLE
    try:
        size = file.stat().st_size
    except OSError:
        return "", 0
    if after > size:
        after = 0
    start = max(after, size - limit)
    with file.open("rb") as handle:
        handle.seek(start)
        data = handle.read(size - start)
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n"), size
