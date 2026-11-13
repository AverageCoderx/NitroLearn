"""What the dashboard knows about the bots in a runs folder: their settings, saves, newest numbers, charts and
whether they are training now (from anywhere: the dashboard, a shortcut or a terminal)."""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import pickle
import re
import sys
import threading
import time
import zipfile
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nitro.config.loader import ConfigError
from nitro.config.schema import NitroConfig, StartFromConfig
from nitro.dashboard import livery
from nitro.runs import RunFolder, source_paths

RUNNING = "running.json"  # written by `nitro train` / `nitro resume` while they run (nitro.cli._run)
STOP_REQUEST = "stop.request"  # asks a running bot to save and stop, as Ctrl+C once does
APPLY_RESTART = "apply_restart.json"  # how a trainer's start-again after an Apply went


def bot_folders(runs: Path) -> list[RunFolder]:
    """Every bot (run folder) in `runs`, newest first."""
    if not runs.is_dir():
        return []
    folders = [RunFolder(p) for p in runs.iterdir() if p.is_dir() and (p / "run.json").is_file()]
    return sorted(folders, key=_last_change, reverse=True)


def _modified(file: Path) -> float:
    try:
        return file.stat().st_mtime
    except OSError:
        return 0.0


def _last_change(folder: RunFolder) -> float:
    times = [p.stat().st_mtime for p in (folder.metrics, folder.settings, folder.info) if p.is_file()]
    return max(times, default=0.0)


def running(folder: RunFolder) -> dict[str, Any] | None:
    """What `running.json` says when its process is still alive, else None (a crash can leave the file; a
    process of another program that took its number after a restart is not taken for it)."""
    try:
        info: dict[str, Any] = json.loads((folder.root / RUNNING).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = int(info.get("pid", 0))
    return info if pid_alive(pid) and _is_python(pid) else None


def claim(folder: RunFolder, info: dict[str, Any]) -> bool:
    """Writes `running.json` for this process unless another trainer of the bot has: created exclusively, so
    of two starting at once (a double click) only one goes on. A file left by a trainer that ended (or one
    being written right now, a few seconds old) is told apart by whether its process still runs."""
    marker = folder.root / RUNNING
    for _ in range(2):
        try:
            handle = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                text = marker.read_text(encoding="utf-8")
                young = time.time() - marker.stat().st_mtime < 10
            except OSError:
                continue  # gone meanwhile: try again
            try:
                json.loads(text)
            except ValueError:  # being written by a trainer starting this moment (or left half-written)
                if young:
                    return False
            else:
                if running(folder) is not None:
                    return False
            marker.unlink(missing_ok=True)  # left behind by a trainer that ended
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(info))
        return True
    return False


def _is_python(pid: int) -> bool:
    """Whether process `pid` is a Python program (a trainer is); True when that cannot be read."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined, unused-ignore]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return True
        try:
            size = wintypes.DWORD(1024)
            name = ctypes.create_unicode_buffer(1024)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
                return True
            return "python" in name.value.lower()
        finally:
            kernel32.CloseHandle(handle)
    try:
        return "python" in Path(f"/proc/{pid}/exe").resolve().name.lower()
    except OSError:
        return True


def pid_alive(pid: int) -> bool:
    """Whether process `pid` runs. Never `os.kill(pid, 0)` on Windows: there it ends the process."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined, unused-ignore]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = wintypes.DWORD()
        try:
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # an ended process its parent has not collected yet (a zombie) does not run
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True  # no /proc (macOS): taken as running
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


# The columns the dashboard reads whole besides the newest row: the steps and speed (training time), the
# home page's reward, the trend's, the reward shares' (raw_, paid_, own_abs_, fired_) and the copy steps'
# ([replay_mix]). Every other column is kept for the charts only: thinned, and whole over the newest rows.
_KEPT = {"agent_steps", "steps_per_second", "reward_per_step", "touches_per_minute"}
_THIN_ROWS = 2048  # at most this many rows of a long run, evenly spread
_TAIL_ROWS = 6000  # and at least this many of the newest, every one


def _kept(column: str) -> bool:
    return column in _KEPT or column.startswith(("raw_", "paid_", "own_abs_", "fired_", "replay_mix_"))


@dataclass
class Metrics:
    """A bot's metrics.csv as the dashboard needs it: the newest row whole, the columns its reward shares and
    training time read, as numbers (nan where empty), and every other column for the charts: every
    `stride`-th row (at most _THIN_ROWS, the stride doubling as the run grows) and every row from `tail_first`
    on. Read again only from where it stopped as the file grows (a day's training writes hundreds of MB:
    reading it all every few seconds took seconds and GB); read anew when it was written anew (a resume trims
    it)."""

    header: list[str] = field(default_factory=list)
    columns: dict[str, array[float]] = field(default_factory=dict)
    last: list[str] = field(default_factory=list)
    count: int = 0
    offset: int = 0  # bytes read: up to the end of the last whole line
    identity: tuple[int, int, float, int] = (0, 0, 0.0, 0)  # the file read: device, inode, mtime, size
    thin_rows: list[int] = field(default_factory=list)
    thin: dict[str, array[float]] = field(default_factory=dict)
    stride: int = 1
    tail_first: int = 0
    tail: dict[str, array[float]] = field(default_factory=dict)

    def value(self, column: str, row: int) -> float | None:
        """The column's number in `row`, None where empty, or where only a thinned copy is kept and `row`
        is not in it (`rows_from` lists the rows every column has)."""
        values = self.columns.get(column)
        if values is not None:
            value = values[row]
        elif column in self.tail and row >= self.tail_first:
            value = self.tail[column][row - self.tail_first]
        elif column in self.thin:
            at = bisect.bisect_left(self.thin_rows, row)
            if at == len(self.thin_rows) or self.thin_rows[at] != row:
                return None
            value = self.thin[column][at]
        else:
            return None
        return value if math.isfinite(value) else None

    def rows_from(self, first: int) -> list[int]:
        """The rows from `first` on that every column has: the thinned ones, then every newest one."""
        older = self.thin_rows[bisect.bisect_left(self.thin_rows, first) :]
        newest = range(max(first, self.tail_first), self.count)
        return [r for r in older if r < self.tail_first] + list(newest)

    def _keep(self, row: int, values: list[float]) -> None:
        """Keeps row `row`'s (the next one's) `values` of the chart columns, in `thin`'s order."""
        if row % self.stride == 0:
            self.thin_rows.append(row)
            for kept, value in zip(self.thin.values(), values, strict=True):
                kept.append(value)
            if len(self.thin_rows) > _THIN_ROWS:
                self.thin_rows = self.thin_rows[::2]
                self.thin = {column: kept[::2] for column, kept in self.thin.items()}
                self.stride *= 2
        for kept, value in zip(self.tail.values(), values, strict=True):
            kept.append(value)
        if row + 1 - self.tail_first >= 2 * _TAIL_ROWS:
            for kept in self.tail.values():
                del kept[:_TAIL_ROWS]
            self.tail_first += _TAIL_ROWS

    def _skip(self, row: int) -> None:
        """Row `row` (the next one) is not kept for the charts: the newest rows kept start after it."""
        if row >= self.tail_first:
            for kept in self.tail.values():
                del kept[:]
        self.tail_first = row + 1


_CACHE: dict[Path, Metrics] = {}
# The window's calls run at once, each in a thread of its own: two reading on from the same place would both
# add the new rows. One reads a file at a time.
_READING: dict[Path, threading.Lock] = {}
_LOCKS = threading.Lock()


def metrics(folder: RunFolder) -> Metrics:
    with _LOCKS:
        lock = _READING.setdefault(folder.metrics, threading.Lock())
    with lock:
        return _read(folder)


def _read(folder: RunFolder) -> Metrics:
    file = folder.metrics
    try:
        stat = file.stat()
    except OSError:
        return Metrics()
    found = _CACHE.get(file)
    identity = (stat.st_dev, stat.st_ino, stat.st_mtime, stat.st_size)
    if found and found.identity == identity:
        return found
    if found is None or found.identity[:2] != identity[:2] or stat.st_size < found.offset:
        found = Metrics()  # a new or rewritten file
    with file.open("rb") as handle:
        handle.seek(found.offset)
        data = handle.read()
    whole = data.rfind(b"\n") + 1
    lines = data[:whole].decode("utf-8", errors="replace").splitlines()
    if not found.header and lines:
        found.header = next(csv.reader(lines[:1]), [])
        found.columns = {column: array("d") for column in found.header if _kept(column)}
        charted = [c for c in found.header if c not in found.columns and c != "iteration"]
        found.thin = {column: array("f") for column in charted}
        found.tail = {column: array("f") for column in charted}
        lines = lines[1:]
    if found.header:
        width = len(found.header)
        kept = [(found.columns[c], i) for i, c in enumerate(found.header) if c in found.columns]
        others = [found.header.index(c) for c in found.thin]
        # A long file read at once: rows neither thinned in nor among the newest are not read for the charts.
        newest = found.count + len(lines) - 2 * _TAIL_ROWS
        for row in csv.reader(lines):
            if len(row) != width:
                continue
            for values, i in kept:
                values.append(_float(row[i]))
            if found.count < newest and found.count % found.stride:
                found._skip(found.count)
            else:
                try:
                    found._keep(found.count, [float(row[i]) for i in others])
                except ValueError:  # an empty cell
                    found._keep(found.count, [_float(row[i]) for i in others])
            found.last = row
            found.count += 1
    found.offset += whole
    found.identity = identity
    _CACHE[file] = found
    return found


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def newest(folder: RunFolder) -> dict[str, float]:
    """The newest metrics row, numbers only. Read from the file's two ends (`ends`), not through `metrics`:
    the Garage asks it of every bot, and a long file read whole took seconds a bot."""
    found = ends(folder)
    if not found.last:
        return {}
    return {k: v for k, v in zip(found.header, map(number, found.last), strict=True) if v is not None}


_CHUNK = 8 * 1024  # bytes read at a time from either end of a metrics file


@dataclass(frozen=True)
class Ends:
    """A metrics.csv's header and newest row as `metrics` reads them (whole lines only; a row of another
    width than the header's is passed over), read from the file's two ends only: `start` is where its rows
    begin, `end` where its last whole line ends and `last_at` where the newest row begins."""

    header: list[str] = field(default_factory=list)
    last: list[str] = field(default_factory=list)
    identity: tuple[int, int, float, int] = (0, 0, 0.0, 0)  # the file read: device, inode, mtime, size
    start: int = 0
    end: int = 0
    last_at: int = 0


_ENDS: dict[Path, Ends] = {}


def ends(folder: RunFolder) -> Ends:
    """The bot's metrics file's `Ends`, read again only when the file changed."""
    file = folder.metrics
    try:
        stat = file.stat()
    except OSError:
        return Ends()
    identity = (stat.st_dev, stat.st_ino, stat.st_mtime, stat.st_size)
    found = _ENDS.get(file)
    if found is not None and found.identity == identity:
        return found
    try:
        with file.open("rb") as handle:
            found = _read_ends(handle, stat.st_size, identity)
    except OSError:
        return Ends()
    _ENDS[file] = found
    return found


def _read_ends(handle: Any, size: int, identity: tuple[int, int, float, int]) -> Ends:
    head, start = b"", -1
    while start < 0 and len(head) < size:  # the header: the first whole line
        head += _read_at(handle, len(head), min(_CHUNK, size - len(head)))
        start = head.find(b"\n") + 1 or -1
    if start < 0:
        return Ends(identity=identity)
    header = _row(head[: start - 1])
    end, tail, at = start, b"", size
    while at > start:  # the newest whole row of the header's width, from the end back
        step = min(max(_CHUNK, len(tail)), at - start)  # doubling while no line ends in what was read
        at -= step
        tail = _read_at(handle, at, step) + tail
        if end == start:
            whole = tail.rfind(b"\n")
            if whole < 0:
                continue
            end, tail = at + whole + 1, tail[: whole + 1]
        lines = tail.split(b"\n")[:-1]  # `tail` is these lines, each ended by a newline
        first = 0 if at == start else 1  # a line cut at the chunk's start is read with the next chunk
        line_at = at + len(tail)
        for line in reversed(lines[first:]):
            line_at -= len(line) + 1
            row = _row(line)
            if len(row) == len(header):
                return Ends(header, row, identity, start, end, line_at)
        tail = lines[0] + b"\n" if first else b""
    return Ends(header, [], identity, start, end, end)


def spread_rows(folder: RunFolder, points: int) -> list[list[str]]:
    """About `points` rows of the bot's metrics file spread evenly through it, oldest first, the first and
    the newest always in (only rows of the header's width): the lines at `points` places spread evenly
    through its bytes, read with a few small reads each, so a long file is never read whole."""
    found = ends(folder)
    if not found.last:
        return []
    width, picked, rows = len(found.header), set(), []
    span = found.last_at - found.start
    places = [found.start + span * i // max(1, points - 1) for i in range(points - 1)]
    try:
        with folder.metrics.open("rb") as handle:
            reach = (found.end - found.last_at) * 5 // 4 + 64  # most lines are about as long as the newest
            for place in places:
                begin, line = _line_at(handle, place, found.start, found.end, reach)
                row = _row(line)
                if begin < found.last_at and begin not in picked and len(row) == width:
                    picked.add(begin)
                    rows.append(row)
    except OSError:
        return [found.last]
    return [*rows, found.last]


def _line_at(handle: Any, offset: int, start: int, end: int, reach: int) -> tuple[int, bytes]:
    """Where the line holding byte `offset` begins, and the line, in an open file whose lines from `start`
    to `end` are whole: read `reach` bytes either side of it, twice as far while that holds no whole line."""
    while True:
        low, high = max(start, offset - reach), min(end, offset + reach + 1)
        data = _read_at(handle, low, high - low)
        here = offset - low
        begin = data.rfind(b"\n", 0, here) + 1
        stop = data.find(b"\n", here)
        if (begin or low == start) and stop >= 0:
            return low + begin, data[begin:stop]
        reach *= 2


def _read_at(handle: Any, at: int, size: int) -> bytes:
    """`size` bytes from `at`; OSError when the file is shorter now (it was rewritten meanwhile)."""
    handle.seek(at)
    data: bytes = handle.read(size)
    if len(data) != size:
        raise OSError("the metrics file changed while it was read")
    return data


def _row(line: bytes) -> list[str]:
    """One line of a metrics file as its cells, as `metrics` reads it."""
    text = line.decode("utf-8", errors="replace").rstrip("\r")
    return next(csv.reader([text]), [])


def whole_rows(folder: RunFolder) -> list[list[str]]:
    """Every row of the bot's metrics file of the header's width, read whole (for a short file only)."""
    found = ends(folder)
    if not found.last:
        return []
    try:
        with folder.metrics.open("rb") as handle:
            data = _read_at(handle, found.start, found.end - found.start)
    except OSError:
        return []
    rows = (_row(line) for line in data.split(b"\n")[:-1])
    return [row for row in rows if len(row) == len(found.header)]


def number(text: str) -> float | None:
    """A metrics.csv cell as a finite number, None for an empty or broken one."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


class _Any:
    """Whatever else a save's pickle names: built, never run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


class _Dict(dict[Any, Any]):
    """An OrderedDict (a state dict): one that also takes the attributes PyTorch stores on it."""


class _Shape:
    def __init__(self, size: Any) -> None:
        self.size = tuple(int(n) for n in size)


def _tensor(storage: Any, offset: Any, size: Any, *rest: Any) -> _Shape:
    return _Shape(size)


_PLAIN = {"set": set, "frozenset": frozenset, "tuple": tuple, "list": list, "dict": dict, "int": int}


class _ShapesOnly(pickle.Unpickler):
    """Reads a PyTorch save's pickle for the shapes of its tensors only: no class of the save is loaded or
    run, and no tensor data is read, so the dashboard needs no PyTorch."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *rest: data
        if module == "collections" and name == "OrderedDict":
            return _Dict
        if module == "builtins" and name in _PLAIN:
            return _PLAIN[name]
        return _Any

    def persistent_load(self, pid: Any) -> Any:
        return None


_FACTS: dict[str, tuple[float, dict[str, Any]]] = {}  # save file -> (its mtime, save_facts)


def save_facts(file: Path) -> dict[str, Any]:
    """What the dashboard reads from a save (a checkpoint's state.pt) without PyTorch: its networks'
    parameters, its agent steps and where its run started from. Read once per version of the file; {} for
    one it cannot read."""
    try:
        mtime = file.stat().st_mtime
    except OSError:
        return {}
    known = _FACTS.get(str(file))
    if known is None or known[0] != mtime:
        known = (mtime, _facts(file))
        _FACTS[str(file)] = known
    return known[1]


def _facts(file: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(file) as archive:
            name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
            state = _ShapesOnly(archive.open(name)).load()
    except Exception:
        return {}
    if not isinstance(state, dict):
        return {}
    steps = state.get("agent_steps")
    return {
        "parameters": _count(state),
        "agent_steps": steps if isinstance(steps, int) else None,
        "started_from": state.get("started_from") if isinstance(state.get("started_from"), str) else "",
        "from_replays": state.get("started_from_replays") is True,
        "settings": state.get("settings") if isinstance(state.get("settings"), dict) else {},  # fixed ones
    }


def network_size(folder: RunFolder) -> dict[str, int] | None:
    """The parameters of the networks in the bot's newest save: in all, and in the shared trunk (with the
    attention encoder), the policy and the value network. None without a save, or one it cannot read."""
    saves = folder.complete_checkpoints()
    if not saves:
        return None
    found: dict[str, int] | None = save_facts(saves[-1] / "state.pt").get("parameters")
    return found


def _count(state: dict[str, Any]) -> dict[str, int] | None:
    try:
        model = state["learner"]["model"]
        parts = {"total": 0, "shared": 0, "policy": 0, "value": 0}
        for key, shape in model.items():
            count = math.prod(shape.size) if isinstance(shape, _Shape) else 0
            parts["total"] += count
            part = str(key).split(".")[0]
            part = "value" if part == "value_heads" else part  # the value heads
            parts[part if part in parts else "shared"] += count
        return parts if parts["total"] else None
    except Exception:
        return None


# A transfer's label in a save (nitro.transfer.Source.label): "first at 1,000 agent steps".
_LABEL = re.compile(r"^(?P<name>.+?) (?:at |\()(?P<steps>[\d,]+) agent steps\)?$")


def _short(steps: int) -> str:
    """Agent steps as the page's numbers show them: 250.0M, 1.25B."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if steps >= size:
            return f"{steps / size:.{2 if unit == 'B' else 1}f}{unit}"
    return str(steps)


def find_source(name: str, at: int, runs: Path) -> dict[str, Any]:
    """The bot start_from.run names, as nitro.transfer finds it but without reading a network: its name for
    the page, the steps of the save it gives, and why it cannot be started from (or "")."""
    for path in source_paths(name, runs):
        if (path / "state.pt").is_file():
            steps = int(path.name) if path.name.isdigit() else None
            return {"source": path.parent.parent.name, "steps": steps, "problem": ""}
        folder = RunFolder(path)
        if folder.info.is_file():
            saves = [int(p.name) for p in folder.complete_checkpoints()]
            problem = ""
            if not saves:
                problem = f"{path.name} has no save yet, so there are no trained networks to start from"
            elif at and at not in saves:
                kept = ", ".join(f"{s:,}" for s in reversed(saves))
                problem = f"{path.name} has no save at {at:,} steps any more (it keeps {kept}): pick another"
            steps = at or (saves[-1] if saves else None)
            return {"source": path.name, "steps": steps, "problem": problem}
    problem = f"there is no bot called {name} (looked in {runs})"
    return {"source": Path(name).name or name, "steps": at or None, "problem": problem}


def source_setting(name: str, at: int, runs: Path, path: str) -> Any:
    """A fixed setting (`path`) of the save start_from.run and at name, as nitro.transfer finds it, read
    without PyTorch; None for no save, or a save without it."""
    for where in source_paths(name, runs):
        if (where / "state.pt").is_file():
            return save_facts(where / "state.pt").get("settings", {}).get(path)
        folder = RunFolder(where)
        if folder.info.is_file():
            saves = [p for p in folder.complete_checkpoints() if not at or int(p.name) == at]
            return save_facts(saves[-1] / "state.pt").get("settings", {}).get(path) if saves else None
    return None


def start_state(
    folder: RunFolder, config: NitroConfig | None, saves: list[Path], agent_steps: int
) -> dict[str, Any]:
    """Where the bot starts from (before its first save) or started from (after), in plain words for its page:
    `kind` ("from", "scratch", "replays"), `done` (it has its own networks), the source and its save's steps,
    a short chip, a line, the start button's words and any problem."""
    out: dict[str, Any] = {"kind": "scratch", "done": bool(saves), "source": "", "steps": None}
    out |= {"problem": "", "run": ""}
    start = config.start_from if config is not None else None
    if saves:
        facts = save_facts(saves[-1] / "state.pt")
        label = facts.get("started_from", "")
        if facts.get("from_replays"):
            out["kind"] = "replays"
        elif label:
            match = _LABEL.match(label)
            out["kind"], out["source"] = "from", match.group("name") if match else label
            out["steps"] = int(match.group("steps").replace(",", "")) if match else None
        elif not facts and start is not None and start.source:  # a save it cannot read: the settings' word
            out["kind"], out["source"] = "from", Path(start.source).name
    elif config is not None and config.pretrain.enabled:
        out["kind"] = "replays"
    elif start is not None and start.source:
        out |= find_source(start.source, start.at, folder.root.parent)
        out["kind"] = "from"
    if start is not None:
        out["run"] = start.run
    return out | _start_words(out, start, agent_steps)


def _start_words(out: dict[str, Any], start: StartFromConfig | None, agent_steps: int) -> dict[str, str]:
    source, steps, done = out["source"], out["steps"], out["done"]
    save = f"'s save at {steps:,} steps" if steps else ""
    if out["kind"] == "from":
        chip = ("Started from " if done else "Starts from ") + source
        if done or start is None:
            line = f"Started from {source}{save}."
        else:
            parts = start.parts
            took = (
                f"everything {source} learned (its policy, value network and scaling)"
                if parts == "all"
                else f"{source}'s policy whole and every part of its value network that fits, and its scaling"
                if parts == "matching"
                else f"only {source}'s policy, with a fresh value network"
            )
            line = f"Transfer learning is on: the first start takes {took}, from {source}{save}."
    elif out["kind"] == "replays":
        chip = "Started from your replays" if done else "Starts from your replays"
        line = (
            "Started from a copy of the players in your replays ([pretrain])."
            if done
            else "It copies the players in your replays first ([pretrain]), so transfer learning is not used."
        )
    else:
        chip = "Started from scratch" if done else "Starts from scratch"
        line = "Started from scratch." if done else "Transfer learning is off: it starts from scratch."
        if not done and start is not None and start.run and not start.enabled:
            line += f" Switch it on to start from {Path(start.run).name} instead."
    if done:
        line += (
            " It has its own trained networks now (its saves), so where it started from can no longer change;"
            " New bot from this one makes a bot that starts from somewhere else."
        )
    if done and agent_steps:
        word = "Carry on training"
    elif done:
        word = "Start training from the copy"
    elif out["kind"] == "from":
        short = f" (save at {_short(steps)} steps)" if steps else ""
        word = f"Start from {source}{short}"
    elif out["kind"] == "replays":
        word = "Copy replays"
    else:
        word = "Start training from scratch"
    return {"chip": chip, "line": line, "word": word}


def copy_view(folder: RunFolder) -> dict[str, Any] | None:
    """The bot's copy of replays for its page: the progress the copy writes as it goes, or for a copy
    finished before it wrote one, what its pretrain.json kept; None when it has none."""
    from nitro import pretrain

    work = folder.root / pretrain.COPY
    data = pretrain.read_live(work)
    if data is not None:
        return data
    try:
        details = json.loads((work / "pretrain.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    large = details.get("large") or {}
    progress = {
        "history": details.get("epochs", []),
        "before": details.get("before"),
        "continued_replays": (details.get("continued_from") or {}).get("replays", 0),
    }
    return {
        "finished": True,
        "doing": "done",
        "learned": details.get("replays", 0),
        "limit": details.get("replays", 0),
        "held_back": details.get("held_out_replays", 0),
        "rounds": pretrain.chart_rounds(progress) if large else [],
        "skill": details.get("skill"),
        "skill_every": 0,
        "by_size": large.get("by_size", {}),
    }


def start_choices(runs: Path, own: str) -> list[dict[str, Any]]:
    """What a bot can start from: every other bot with a save (its saves, newest first)."""
    return [
        {"value": f.root.name, "label": f.root.name, "saves": [int(p.name) for p in reversed(saves)]}
        for f in sorted(bot_folders(runs), key=lambda f: f.root.name.lower())
        if f.root.name != own and (saves := f.complete_checkpoints())
    ]


# A bot's folder -> its settings file's and run.json's versions, its settings and why they cannot be read.
_SETTINGS: dict[Path, tuple[Any, NitroConfig | None, str]] = {}


def listed_settings(folder: RunFolder) -> tuple[NitroConfig | None, str]:
    """The bot's settings as the Garage lists it, or None and why they cannot be read; read again only when
    its settings file or run.json changed (reading them took most of a refresh of many bots)."""
    try:
        files = (folder.settings.stat(), folder.info.stat())
        version = tuple((s.st_ino, s.st_mtime_ns, s.st_size) for s in files)
    except OSError:
        version = None
    known = _SETTINGS.get(folder.root)
    if version is not None and known is not None and known[0] == version:
        return known[1], known[2]
    config, problem = None, ""
    try:
        config = folder.load_settings()
    except ConfigError as err:
        problem = str(err)
    except (OSError, ValueError, KeyError) as err:
        problem = f"cannot read its settings: {err}"
    if version is not None:
        _SETTINGS[folder.root] = (version, config, problem)
    return config, problem


def summary(folder: RunFolder) -> dict[str, Any]:
    """One bot as the dashboard lists it."""
    saves = folder.complete_checkpoints()
    row = newest(folder)
    state = running(folder)
    config, problem = listed_settings(folder)
    so_far = folder.copy_so_far() if saves else None
    own = [] if so_far is not None else saves  # a save of an unfinished copy is not its own networks
    copies = config is not None and config.pretrain.enabled
    copying = ((folder.root / "pretrained").is_dir() or copies) and not own
    agent_steps = int(row.get("agent_steps", 0)) or (int(saves[-1].name) if saves else 0)
    start = start_state(folder, config, own, agent_steps)
    status = "training" if state else ("problem" if problem else "stopped")
    if state and copying:
        status = "copying"
    out: dict[str, Any] = {
        "name": folder.root.name,
        "status": status,
        "problem": problem,
        "saves": len(saves),
        "newest_save": int(saves[-1].name) if saves else None,
        "agent_steps": agent_steps,
        "steps_per_second": row.get("steps_per_second"),
        "reward_per_step": row.get("reward_per_step"),
        "updated": _last_change(folder),
        "started_from": start["source"] if start["kind"] == "from" else "",
        "start": start,
        "start_word": start["word"],
        "copied_replays": (folder.root / "pretrained").is_dir(),
        # Its first start copies the replays, saves and stops; the next one trains.
        "copy_first": copies and not own,
        "copy_so_far": so_far,  # its save holds the copy part-way: {"learned", "limit"} replays
        "warmup": warmup(config, start, agent_steps) if own else None,
        "parameters": network_size(folder) if saves else None,
        "livery": livery.read(folder.root),  # its race number and colours: {number, main, contrast, ink}
    }
    if state:
        out["running_since"] = state.get("started", 0)
        out["running_seconds"] = max(0.0, time.time() - float(state.get("started", time.time())))
        # Whether a round of training has ended since this start: until then its numbers are older ones.
        out["counted"] = _modified(folder.metrics) >= float(state.get("started", 0))
    if config is not None:
        out.update(
            {
                "team_size": config.env.team_size,
                "tick_skip": config.env.tick_skip,
                "observation": config.env.observation,
                "encoder": config.model.encoder,
                "gco": config.model.value_heads_apart,  # the GCO design: a value side with heads of its own
                "obs_team_size": config.env.obs_team_size,
                "sizes": watch_sizes(config),
            }
        )
    return out


def warmup(config: NitroConfig | None, start: dict[str, Any], agent_steps: int) -> dict[str, Any] | None:
    """The value warm-up of a bot started from replays, for its Overview: its setting, its steps and
    how many are trained (the policy is held still until then, nitro.training); None when it has none."""
    if config is None or start["kind"] != "replays":
        return None
    steps = config.pretrain.value_warmup_steps
    if not steps:
        return None
    return {"path": "pretrain.value_warmup_steps", "steps": steps, "done": min(agent_steps, steps)}


# The most players a side the simulator plays (nitro_core::snapshot::MAX_CARS, 8 cars).
MOST_A_SIDE = 4


def watch_sizes(config: NitroConfig) -> list[int]:
    """The team sizes a bot can play: its own; with a padded observation every size up to its room;
    with an attention encoder every size the simulator plays, since it reads any number of cars (how well
    beyond the sizes it trained at is not measured). Other sizes would change what a network reads."""
    if config.model.uses_attention:
        return list(range(1, MOST_A_SIDE + 1))
    if config.env.obs_team_size:
        return list(range(1, config.env.obs_team_size + 1))
    return [config.env.team_size]


def wandb_url(folder: RunFolder) -> str:
    """The run's W&B page, when it has one (`nitro.tracking` keeps it in wandb_run.json)."""
    try:
        note = json.loads((folder.root / "wandb_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    url = note.get("url", "")
    return url if isinstance(url, str) and url.startswith("https://") else ""
