"""Watching a bot play inside the dashboard's window: the match viewer's files handed to the page through
the window's bridge, the bot's recorded games, and one live game at a time, played by a process of its own
whose lines are read here. Nothing listens on a port, and this process never loads PyTorch: every game is
played by `nitro watch` in its own process.

The window takes its page as text of at most 2 MB, so the viewer (about 1.1 MB with three.js) is not in the
page: the page asks for it the first time a game is watched. Every big text crosses the bridge in pieces of
`PIECE` characters, a few calls instead of one large answer."""

from __future__ import annotations

import atexit
import contextlib
import functools
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from nitro import viewer
from nitro.config.loader import ConfigError
from nitro.dashboard import bots, winapp
from nitro.dashboard.jobs import CONSOLE, Jobs, read_console
from nitro.runs import RunFolder

if TYPE_CHECKING:
    from nitro.viewer import feed

PIECE = 256 * 1024  # characters of one answer
# A bot's recorded games: the newest (`watch.nlrec`, beside the page `nitro watch` writes) and the ones
# before it, kept in its `recordings` folder up to this many and this much.
LAST = "last"
KEPT = "recordings"
KEEP_RECORDINGS = 5
KEEP_BYTES = 48 * 1024 * 1024
SUFFIX = ".nlrec"  # nitro.viewer.recording.SUFFIX (that module loads numpy, which this process may not need)
SECONDS = (5, 600)  # the shortest and the longest game recorded from here
# A page that asks late gets at most this many chunks in one answer (about 20 seconds of play).
POLL_CHUNKS = 300
_ASSETS: dict[str, Callable[[], str]] = {
    "three": viewer.three,
    "script": viewer.maker,
    "style": viewer.css,
    "markup": viewer.markup,
}
_IDENT = re.compile(r"^[0-9A-Za-z][0-9A-Za-z-]{0,63}$")
_STEPS = re.compile(r" at ([0-9][0-9,]*) agent steps$")
_arenas: dict[str, str] = {}  # meshes folder -> its arena as the viewer reads it


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """`nitro.dashboard.api._safe` (a problem comes back as {"error": ...}), found when called: that module
    imports this one."""

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from nitro.dashboard import api

        return api._safe(method)(*args, **kwargs)

    return wrapped


def pieces(text: str, piece: int) -> dict[str, Any]:
    """Piece number `piece` of `text` and how many pieces there are."""
    count = -(-len(text) // PIECE)
    at = max(0, int(piece or 0))
    return {"text": text[at * PIECE : (at + 1) * PIECE], "pieces": count}


# How long a live game's process gets to end once its input closes, and then once it is killed.
END_SECONDS = 10.0
KILL_SECONDS = 5.0


@dataclass
class Live:
    """The live game: its process (`nitro watch NAME --feed`), which plays until its input closes, and what
    it has written so far. Its notices are copied into the bot's console file a line at a time, so no file in
    the bot's folder stays open while it plays (on Windows a folder with an open file cannot be removed)."""

    bot: str
    folder: Path
    process: subprocess.Popen[str]
    reader: feed.Reader
    console_from: int  # where this game's notices begin in the bot's console file
    checked: float = field(default_factory=time.monotonic)  # when the bot's folder was last seen
    notices: threading.Thread | None = None  # copies the process's notices into the console file

    def __post_init__(self) -> None:
        if self.notices is None and self.process.stderr is not None:
            self.notices = threading.Thread(
                target=_copy_notices,
                args=(self.process.stderr, self.folder / CONSOLE),
                name="nitro-live-notices",
                daemon=True,
            )
            self.notices.start()

    def noticed(self, timeout: float) -> None:
        """Waits up to `timeout` seconds for the process's last notices to reach the console file."""
        if self.notices is not None:
            self.notices.join(timeout)

    def end(self) -> None:
        """Ends the game and waits for it: its input closes, which it takes as the sign to stop; a process
        that has not gone `END_SECONDS` later is killed. Then its output is read to the end and every pipe
        to it is closed."""
        with contextlib.suppress(OSError, ValueError):
            if self.process.stdin is not None:
                self.process.stdin.close()
        try:
            self.process.wait(timeout=END_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self.process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=KILL_SECONDS)
        # The process is gone, so its pipes are at their end and the threads reading them finish.
        self.reader.wait(KILL_SECONDS)
        self.noticed(KILL_SECONDS)
        for pipe in (self.process.stdout, self.process.stderr):
            with contextlib.suppress(OSError, ValueError):
                if pipe is not None:
                    pipe.close()


def _copy_notices(lines: IO[str], console: Path) -> None:
    """Appends each line of `lines` to the file `console`, opening it for that line only; a line that cannot
    be written (the bot's folder was removed) is dropped."""
    try:
        for line in lines:
            with contextlib.suppress(OSError), console.open("a", encoding="utf-8", newline="") as file:
                file.write(line)
    except (OSError, ValueError):
        pass  # the pipe was closed


class Watching:
    """What the window asks for to show a bot playing; `nitro.dashboard.api.Api` inherits it."""

    if TYPE_CHECKING:
        _project: Path
        _runs: Path
        _jobs: Jobs
        _window: Any
        set_pref: Callable[..., Any]

        def _folder(self, name: str) -> RunFolder: ...

        def _prefs(self) -> dict[str, Any]: ...

    _live: Live | None = None
    _live_lock = threading.Lock()
    _live_hooked = False

    # ---------- the viewer's files ----------
    @_safe
    def viewer_assets(self, part: str, piece: int = 0) -> dict[str, Any]:
        """One piece of a part of the match viewer: "three" (three.js), "script" (the viewer's own, as
        `nitro.viewer.maker` gives it), "style" (the page holds the fonts already) or "markup"."""
        if part not in _ASSETS:
            return {"error": f"the viewer has no part called {part}"}
        return pieces(_ASSETS[part](), piece)

    @_safe
    def viewer_arena(self, name: str = "", piece: int = 0) -> dict[str, Any]:
        """One piece of the arena's collision meshes as the viewer reads them (`nitro.viewer.arena_blob`):
        the ones the bot `name` plays in, or the project's. No pieces when there are none; the viewer then
        draws a generated arena."""
        folders = [self._project / "collision_meshes"]
        if name:
            # A bot that is gone or whose settings have a problem: the project's meshes will do.
            with contextlib.suppress(ConfigError, OSError, ValueError, KeyError):
                folders.insert(0, self._folder(name).load_settings().paths.collision_meshes)
        for folder in folders:
            key = str(folder)
            blob = _arenas.get(key) or viewer.arena_blob(folder)
            if blob:
                _arenas[key] = blob
                return pieces(blob, piece)
        return pieces("", piece)

    @_safe
    def viewer_prefs(self) -> dict[str, Any]:
        """The viewer's remembered choices (camera, speed, picture quality, what is shown)."""
        kept = self._prefs().get("viewer")
        return {"prefs": kept if isinstance(kept, dict) else {}}

    @_safe
    def set_viewer_prefs(self, prefs: Any) -> dict[str, Any]:
        """Keeps the viewer's choices with the dashboard's own."""
        if not isinstance(prefs, dict) or len(json.dumps(prefs, allow_nan=False)) > 10_000:
            return {"error": "these are not the viewer's choices"}
        self.set_pref("viewer", prefs)
        return {"saved": True}

    # ---------- recorded games ----------
    @_safe
    def saves(self, name: str) -> dict[str, Any]:
        """The bot's saves a game can be played with, newest first: each with its agent steps and when it was
        made."""
        found = []
        for save in reversed(self._folder(name).complete_checkpoints()):
            with contextlib.suppress(OSError):
                found.append({"steps": int(save.name), "time": (save / "state.pt").stat().st_mtime})
        return {"saves": found}

    @_safe
    def watch_record(self, name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Starts recording a game of the bot (`nitro watch`, in its own process; `job(name, "watch")` says
        how it goes). `options`: `seconds` (the game's length), `size` (players a side, one the bot can play;
        its own when left out) and `at` (the agent steps of the save that plays; the newest when left out)."""
        folder = self._folder(name)
        options = options or {}
        running = self._jobs.get(name, "watch")
        if running is not None and not running.done:
            return {"error": f"a game of {name} is being recorded already"}
        seconds = int(options.get("seconds") or 60)
        if not SECONDS[0] <= seconds <= SECONDS[1]:
            return {"error": f"a recorded game is {SECONDS[0]} to {SECONDS[1]} seconds long"}
        args = ["watch", name, "--runs", str(self._runs), "--no-open", "--seconds", str(seconds)]
        args += ["--seed", str(_seed()), *self._game_options(folder, options)]
        at = options.get("at")
        if at is not None:
            if int(at) not in {int(save.name) for save in folder.complete_checkpoints()}:
                return {"error": f"{name} has no save at {int(at):,} agent steps (any more)"}
            args += ["--at", str(int(at))]
        self._keep_last(folder)
        self._jobs.start(name, "watch", args, folder.root)
        return {"started": True}

    @_safe
    def recordings(self, name: str) -> dict[str, Any]:
        """The bot's recorded games that can be watched now, newest first: each with its `id`, title, the
        save that played (`steps`), players a side (`size`), length (`seconds`) and when it was recorded
        (`time`). `page` is the newest one's page (`watch.html`), which can be passed on, when it is there."""
        folder = self._folder(name).root
        last = folder / f"watch{SUFFIX}"
        files = [(LAST, last)] if last.is_file() else []
        same = _stamp(last) if files else None
        files += [(file.stem, file) for file in _kept(folder) if _stamp(file) != same]
        found = []
        for ident, file in files:
            with contextlib.suppress(OSError, ValueError, EOFError):
                found.append({"id": ident, **_note(str(file), *_stamp(file))})
        page = folder / "watch.html"
        return {"recordings": found, "page": str(page) if page.is_file() else ""}

    @_safe
    def recording(self, name: str, ident: str = LAST, piece: int = 0) -> dict[str, Any]:
        """One piece of a recorded game's page data, as JSON text (the viewer's `NV.start` takes it once it
        is put together and read)."""
        folder = self._folder(name).root
        if not _IDENT.match(str(ident)):
            return {"error": "there is no such recording"}
        file = folder / f"watch{SUFFIX}" if ident == LAST else folder / KEPT / f"{ident}{SUFFIX}"
        if not file.is_file():
            return {"error": "that recording is not there any more"}
        return pieces(_text(str(file), *_stamp(file)), piece)

    @_safe
    def open_watch_page(self, name: str) -> dict[str, Any]:
        """Opens the newest recording's page (`watch.html`) in the browser."""
        page = self._folder(name).root / "watch.html"
        if not page.is_file():
            return {"error": "there is no recorded game's page yet"}
        webbrowser.open(page.resolve().as_uri())
        return {"opened": str(page)}

    @_safe
    def show_watch_file(self, name: str) -> dict[str, Any]:
        """Shows the newest recording's page (`watch.html`) in its folder, to pass it on."""
        page = self._folder(name).root / "watch.html"
        if not page.is_file():
            return {"error": "there is no recorded game's page yet"}
        _reveal(page)
        return {"shown": str(page)}

    def _game_options(self, folder: RunFolder, options: dict[str, Any]) -> list[str]:
        """`nitro watch`'s options for the game size asked for; a size the bot cannot play is refused here,
        as the command would."""
        size = int(options.get("size") or 0)
        if not size:
            return []
        config = folder.load_settings()
        if size not in bots.watch_sizes(config):
            own = config.env.team_size
            raise ValueError(
                f"{folder.root.name} reads {own}v{own} games only, so it cannot play {size}v{size} (a padded "
                "observation lets a bot play smaller games; an attention network, every size)"
            )
        return ["--team-size", str(size)]

    def _keep_last(self, folder: RunFolder) -> None:
        """Before a new game is recorded over it, the newest recording goes to the kept ones, and the oldest
        of those go when they are too many or too much."""
        last = folder.root / f"watch{SUFFIX}"
        if not last.is_file():
            return
        kept = _kept(folder.root)
        try:
            if _stamp(last) not in {_stamp(file) for file in kept}:
                note = _note(str(last), *_stamp(last))
                made = time.strftime("%Y%m%d-%H%M%S", time.localtime(last.stat().st_mtime))
                steps = f"{note['steps']:012d}-" if note.get("steps") is not None else ""
                (folder.root / KEPT).mkdir(exist_ok=True)
                shutil.copy2(last, folder.root / KEPT / f"{steps}{made}{SUFFIX}")
                kept = _kept(folder.root)
        except (OSError, ValueError, EOFError):
            return  # a recording that cannot be read or copied is not worth refusing the new game for
        total = 0
        for at, file in enumerate(kept):  # the newest recording itself counts as one of those kept
            with contextlib.suppress(OSError):
                total += file.stat().st_size
                if at >= KEEP_RECORDINGS - 1 or total > KEEP_BYTES:
                    file.unlink()

    # ---------- the live game ----------
    @_safe
    def watch_live_start(self, name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Starts the live game of the bot (`nitro watch NAME --feed`, in its own process, on the processor):
        its newest save plays on and on, and with `follow` (on unless switched off) each new save takes
        over. `size`: players a side, as for a recorded game. One live game at a time: one that plays is
        ended first. `watch_live_state` says when it plays, and why not if it does not."""
        folder = self._folder(name)
        options = options or {}
        if not folder.complete_checkpoints():
            return {"error": f"{name} has no save yet to play with"}
        args = ["watch", name, "--runs", str(self._runs), "--feed", "--seed", str(_seed())]
        args += self._game_options(folder, options)
        if options.get("follow", True):
            args.append("--follow")
        from nitro.viewer import feed  # here, not at the top: it loads numpy, which only a live game needs

        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined, unused-ignore]
        with self._live_lock:
            before, self._live = self._live, None
            if before is not None:
                before.end()
            with (folder.root / CONSOLE).open("ab") as notices:
                said = f"\n$ nitro {' '.join(args)}   ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                notices.write(said.encode())
                begin = notices.tell()
            # The game's lines come back through one pipe and its notices through another, copied into the
            # bot's console. It plays until the pipe to its input closes, which also happens when this
            # process ends.
            process = subprocess.Popen(
                [winapp.console_python(), "-u", "-m", "nitro", *args],
                cwd=self._project,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                creationflags=flags,
            )
            assert process.stdout is not None
            self._live = Live(name, folder.root, process, feed.Reader(process.stdout), begin)
            if not self._live_hooked:
                self._live_hooked = True
                atexit.register(self._end_live)
                if self._window is not None:
                    self._window.events.closed += self._window_closed
        return {"started": True}

    @_safe
    def watch_live_state(self) -> dict[str, Any]:
        """How the live game is doing: "none"; "starting" (its process is loading the bot); "playing", with
        the page data to start the viewer with (`data`); or "ended" when its process went by itself, with
        its last lines (`why`)."""
        live = self._live
        if live is None:
            return {"state": "none"}
        header = live.reader.header
        if live.reader.closed:
            live.noticed(2.0)  # its last notices, which say why it stopped
            text, _ = read_console(live.folder, live.console_from)
            lines = [line for line in text.split("\n$ nitro ", 1)[0].splitlines() if line.strip()]
            why = "\n".join(lines[-6:]) or "it stopped; the Console tab may say why"
            return {"state": "ended", "bot": live.bot, "why": why}
        if header is None:
            return {"state": "starting", "bot": live.bot}
        return {"state": "playing", "bot": live.bot, "data": header}

    def viewer_poll(self, cursor: int = 0) -> dict[str, Any]:
        """What the live game has played since `cursor` (`nitro.viewer.feed.Chunks.after`); a closed answer
        when no game plays."""
        try:
            at = int(cursor or 0)
        except (TypeError, ValueError):
            at = 0
        live = self._live
        if live is None:
            return {"chunks": [], "next": at, "restart": False, "closed": True}
        now = time.monotonic()
        if now - live.checked > 2:  # a bot that was deleted plays no more
            live.checked = now
            if not live.folder.is_dir():
                self._end_live()
        answer = live.reader.after(at)
        extra = len(answer["chunks"]) - POLL_CHUNKS
        if extra > 0:
            answer = {**answer, "chunks": answer["chunks"][:POLL_CHUNKS], "next": answer["next"] - extra}
            answer["closed"] = False
        return answer

    @_safe
    def watch_live_stop(self, name: str = "") -> dict[str, Any]:
        """Ends the live game (the one of bot `name` only, when a name is given)."""
        return {"stopped": self._end_live(name)}

    def _end_live(self, name: str = "") -> bool:
        """Ends the live game (the bot `name`'s only, when given) and waits for its process to end."""
        with self._live_lock:
            live = self._live
            if live is None or (name and live.bot != name):
                return False
            self._live = None
            live.end()
        return True

    def _window_closed(self) -> None:
        """The dashboard's window has closed: the live game ends with it."""
        self._end_live()


def _seed() -> int:
    """A different game every time one is asked for."""
    return time.time_ns() // 1_000_000 % 1_000_000


def _kept(folder: Path) -> list[Path]:
    """The recordings kept in a bot's folder, newest first."""
    try:
        found = [file for file in (folder / KEPT).iterdir() if file.suffix == SUFFIX and file.is_file()]
    except OSError:
        return []
    return sorted(found, key=lambda file: file.stat().st_mtime, reverse=True)


def _stamp(file: Path) -> tuple[int, int]:
    """What tells one writing of a file from the next: when it was written and its size."""
    stat = file.stat()
    return stat.st_mtime_ns, stat.st_size


@functools.lru_cache(maxsize=2)
def _text(file: str, written: int, size: int) -> str:
    """A recording file's content: its page data as JSON text."""
    with gzip.open(file, "rt", encoding="utf-8") as text:
        return text.read()


@functools.lru_cache(maxsize=4 * KEEP_RECORDINGS)
def _note(file: str, written: int, size: int) -> dict[str, Any]:
    """What a list of recordings says of one."""
    with gzip.open(file, "rt", encoding="utf-8") as text:
        data = json.load(text)
    if not isinstance(data, dict) or "frames" not in data:
        raise ValueError(f"{file} holds no recording")
    title = str(data.get("title", ""))
    steps = _STEPS.search(title)
    rate = float(data.get("decisionsPerSecond") or 0)
    return {
        "title": title,
        "steps": int(steps.group(1).replace(",", "")) if steps else None,
        "size": int(data.get("cars") or 0) // 2,
        "seconds": round(len(data.get("decisions") or []) / rate) if rate else 0,
        "time": written / 1e9,
        "bytes": size,
    }


def _reveal(file: Path) -> None:
    """Opens the file's folder with the file picked out, where the system can do that."""
    if sys.platform == "win32":
        subprocess.Popen(["explorer", f"/select,{file}"])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(file)])
    else:
        subprocess.Popen(["xdg-open", str(file.parent)])
