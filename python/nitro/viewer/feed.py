"""Live games for the match viewer, with nothing listening on any port.

A game is played in real time and written one line of JSON at a time: first the page's data (a recording
with no frames yet, marked live), then a chunk per decision with the new frames and the bot's figures. The
lines go into a `Chunks` store, and the page asks through its window's bridge for what is new since it last
asked; it keeps its own history to scroll back in. `nitro watch --live` plays in its own process and fills
the store directly; a process that only writes the lines (`nitro watch --feed`) is read with `Reader`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import IO, Any, Protocol

import numpy as np

from nitro.viewer import recording

# Chunks the reader keeps for a page that asks late (about two minutes of a 15-decisions-a-second game).
KEEP_CHUNKS = 2000


class Playing(Protocol):
    """A game being played one decision at a time (`nitro.watch.Game`)."""

    start: list[float]
    width: int

    def decide(self) -> tuple[list[Any], list[float]]: ...


def header(data: dict[str, Any]) -> str:
    """The first line: the page's data, with no frames and marked live."""
    first = {**data, "frames": "", "packing": "", "decisions": [], "live": True}
    return json.dumps(first, separators=(",", ":"), allow_nan=False)


def chunk(frames: np.ndarray, decisions: list[list[Any]], cars: int, title: str = "") -> str:
    """One line of new frames ([frames, width], real numbers) and the decisions made with them."""
    line: dict[str, Any] = {"f": recording.raw(recording.quantize(frames, cars)), "d": decisions}
    if title:
        line["title"] = title
    return json.dumps(line, separators=(",", ":"), allow_nan=False)


def play(
    game: Playing,
    write: Callable[[str], None],
    *,
    cars: int,
    seconds_per_decision: float,
    stop: threading.Event,
    newer: Callable[[], tuple[Playing, str] | None] | None = None,
    check_seconds: float = 20.0,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Plays `game` in real time until `stop`, writing a chunk per decision. With `newer`, asked every
    `check_seconds`, another game (a newer save, with its title) takes over, starting with its first frame."""

    def rows(flat: list[float]) -> np.ndarray:
        return np.asarray(flat, dtype=np.float32).reshape(-1, game.width)

    written = 0  # frames so far
    base = 0  # where the current game's own frame numbers start
    first = rows(game.start)
    write(chunk(first, [], cars))
    written += len(first)
    began = clock()
    played = 0.0
    next_check = began + check_seconds
    while not stop.is_set():
        decision, flat = game.decide()
        decision[0] += base
        new = rows(flat)
        write(chunk(new, [decision], cars))
        written += len(new)
        played += seconds_per_decision
        behind = clock() - (began + played)
        if behind < 0:
            stop.wait(-behind)
        elif behind > 1:
            began += behind  # this computer fell behind: carry on from now rather than rush to catch up
        if newer is not None and clock() >= next_check:
            next_check = clock() + check_seconds
            found = newer()
            if found is not None:
                game, title = found
                base = written
                first = rows(game.start)
                write(chunk(first, [], cars, title))
                written += len(first)


class Chunks:
    """What a live game has written so far: its header (the page's data) and the newest chunks.

    `after(cursor)` gives what came since `cursor` (the number of chunks the page has had): the chunks, the
    next cursor, and whether the game has ended. A page that asks for chunks already dropped gets a
    `restart` flag and the chunks still kept, so it starts its history afresh."""

    def __init__(self, keep: int = KEEP_CHUNKS) -> None:
        self.keep = keep
        self.header: dict[str, Any] | None = None
        self.closed = False
        self._chunks: list[dict[str, Any]] = []
        self._first = 0  # the number of the oldest chunk kept
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def add(self, line: str) -> None:
        """The game's next line: the header first, then chunks."""
        value = json.loads(line)
        with self._lock:
            if self.header is None:
                self.header = value
                self._ready.set()
                return
            self._chunks.append(value)
            extra = len(self._chunks) - self.keep
            if extra > 0:
                del self._chunks[:extra]
                self._first += extra

    def close(self) -> None:
        """The game has ended: nothing more will come."""
        self.closed = True
        self._ready.set()

    def wait_header(self, timeout: float) -> dict[str, Any] | None:
        """The page's data, once it is written (None if the game ended first or took too long)."""
        self._ready.wait(timeout)
        return self.header

    def after(self, cursor: int) -> dict[str, Any]:
        with self._lock:
            restart = cursor < self._first
            start = max(cursor, self._first) - self._first
            return {
                "chunks": self._chunks[start:],
                "next": self._first + len(self._chunks),
                "restart": restart,
                "closed": self.closed,
            }


class Reader(Chunks):
    """A store filled on a thread from the lines of a process that plays."""

    def __init__(self, lines: IO[str], keep: int = KEEP_CHUNKS) -> None:
        super().__init__(keep)
        self._thread = threading.Thread(target=self._read, args=(lines,), daemon=True)
        self._thread.start()

    def wait(self, timeout: float) -> bool:
        """Waits up to `timeout` seconds for the lines to end (the process closed its output); whether they
        have."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _read(self, lines: IO[str]) -> None:
        try:
            for line in _lines(lines):
                self.add(line)
        except (OSError, ValueError):
            pass  # the process ended in the middle of a line
        finally:
            self.close()


def _lines(stream: IO[str]) -> Iterator[str]:
    for line in stream:
        if line.strip():
            yield line
