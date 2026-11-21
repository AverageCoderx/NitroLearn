"""`nitro get-replays`: ready-made high-level replays for replay copying, without a ballchasing.com key.

The replays are 1v1 and 2v2 games from Rolv Arild's High-Level Rocket League Replay Dataset (curated
Grand Champion and above ranked games and scrims, filtered for bots, cheaters, handicap challenges and
mutators; kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset). A copy is published as the
assets of one GitHub release: a manifest.json and chunks, each a zip of a few hundred .replay files, the
best-scored games first. Only the chunks needed for the number of replays asked for are downloaded, each
resumable (HTTP Range) and checked against its SHA-256; replays already downloaded or converted are skipped.
They land in replays/dataset/<size>, the most skilled game newest (so a copy limited to N replays takes the
best N), and are converted into the store (with the timing fit) the way `nitro convert-replays` does.
Plain HTTPS through Python's own client, which on Windows trusts the system certificate store (so it works
behind antivirus HTTPS scanning); nothing listens on a port.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from nitro import replays

REPO = "AverageCoderx/NitroLearn-replays"
TAG = "dataset-v1"
MANIFEST_URL = f"https://github.com/{REPO}/releases/download/{TAG}/manifest.json"
FORMAT = 1
SIZES = ("1v1", "2v2", "3v3")
CREDIT = (
    "Replays: Rolv Arild's High-Level Rocket League Replay Dataset "
    "(kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset), games from ballchasing.com."
)
CONVERTED_BYTES = 10 * 2**20  # a converted replay in the store, about (8-10 MB a 5-minute game)
CHUNK_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}\.zip$")
REPLAY_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")
# Unpacked replays are dated from here back, a second per rank, the same on every run: the game folders'
# "newest first" then means best first.
NEWEST = 1_600_000_000
ATTEMPTS = 6  # tries in a row that bring no byte more, before a chunk is given up (what came is kept)
READ = 1 << 20

Opener = Callable[[urllib.request.Request, float], IO[bytes]]
Say = Callable[[str], None]


class DatasetError(RuntimeError):
    """The replays could not be downloaded; the message says why in plain words."""


class NotPublished(DatasetError):
    """The replay list is not online yet (the server has no such file)."""


NOT_YET = (
    "Replay downloads aren't available yet: the replays have not been put online. They are coming in a "
    "later update. Until then, replays you have yourself can be converted and copied as before."
)


def folder() -> Path:
    return replays.OUT / "dataset"


def size_folder(size: str) -> Path:
    """Where the downloaded replays of one game size are kept (a replay folder for `[pretrain]`)."""
    return folder() / size


def progress_file() -> Path:
    return folder() / "downloading.json"


@dataclass(frozen=True)
class Chunk:
    name: str
    size: str  # "1v1"
    bytes: int  # the zip
    unpacked: int  # its replays
    sha256: str
    replays: tuple[tuple[str, float], ...]  # (id, skill score), best first


@dataclass(frozen=True)
class Manifest:
    credit: str
    chunks: tuple[Chunk, ...]

    def of_size(self, size: str) -> list[Chunk]:
        return [c for c in self.chunks if c.size == size]

    def available(self) -> dict[str, dict[str, int]]:
        """Per game size: replays and gigabytes (of zips) published."""
        out: dict[str, dict[str, int]] = {}
        for c in self.chunks:
            row = out.setdefault(c.size, {"replays": 0, "bytes": 0, "chunks": 0})
            row["replays"] += len(c.replays)
            row["bytes"] += c.bytes
            row["chunks"] += 1
        return out


def read_manifest(data: bytes) -> Manifest:
    """The manifest, every field checked (its names become file names here)."""
    try:
        raw = json.loads(data.decode("utf-8"))
        if raw.get("format") != FORMAT:
            raise DatasetError(
                f"the replay list is format {raw.get('format')}, this NitroLearn reads format {FORMAT}: "
                "update NitroLearn"
            )
        chunks = []
        for c in raw["chunks"]:
            name, size, sha = str(c["name"]), str(c["size"]), str(c["sha256"]).lower()
            ids = tuple((str(i), float(s)) for i, s in c["replays"])
            if not CHUNK_NAME.match(name) or size not in SIZES or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError(f"bad chunk entry {name!r}")
            if not ids or not all(REPLAY_ID.match(i) for i, _ in ids):
                raise ValueError(f"bad replay id in {name}")
            chunks.append(Chunk(name, size, int(c["bytes"]), int(c["unpacked"]), sha, ids))
        return Manifest(str(raw.get("credit", CREDIT)), tuple(chunks))
    except DatasetError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        raise DatasetError(f"the replay list could not be read ({type(err).__name__}: {err})") from None


def _request(url: str, start: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": "NitroLearn-get-replays"}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def _open(request: urllib.request.Request, seconds: float) -> IO[bytes]:
    answer: IO[bytes] = urllib.request.urlopen(request, timeout=seconds)
    return answer


NETWORK_HINT = (
    "Check the internet connection and try again; what was downloaded is kept. Antivirus web shields "
    "(HTTPS scanning) and company proxies can hold back downloads: allow NitroLearn's folder in the "
    "antivirus."
)


def fetch_manifest(url: str = MANIFEST_URL, opener: Opener = _open, say: Say = print) -> Manifest:
    """The published list, kept in replays/dataset/manifest.json; the kept copy when offline."""
    kept = folder() / "manifest.json"
    try:
        with opener(_request(url), 60) as answer:
            data = answer.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        if kept.is_file():
            say(f"Could not fetch the replay list ({_reason(err)}); using the copy downloaded before.")
            return read_manifest(kept.read_bytes())
        if url == MANIFEST_URL and isinstance(err, urllib.error.HTTPError) and err.code in (404, 410):
            raise NotPublished(NOT_YET) from None
        raise DatasetError(f"could not fetch the replay list ({_reason(err)}). {NETWORK_HINT}") from None
    manifest = read_manifest(data)
    kept.parent.mkdir(parents=True, exist_ok=True)
    partial = kept.with_name(kept.name + ".partial")
    partial.write_bytes(data)
    partial.replace(kept)
    return manifest


def _reason(err: BaseException) -> str:
    if isinstance(err, urllib.error.HTTPError):
        return f"the server answered {err.code}"
    if isinstance(err, urllib.error.URLError):
        return str(err.reason)[:160]
    return f"{type(err).__name__}: {str(err)[:160]}"


def download(
    url: str,
    target: Path,
    size: int,
    sha256: str,
    opener: Opener = _open,
    seen: Callable[[int], None] | None = None,
    attempts: int = ATTEMPTS,
    wait: Callable[[float], None] = time.sleep,
) -> None:
    """`url` into `target` (`size` bytes), carrying on from a `.part` file left by an earlier try or a cut
    connection; checked against `sha256` before it is kept. `seen` hears the bytes held so far."""
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    problem: BaseException | None = None
    failures = 0  # in a row without a byte more: a slow, often cut connection still gets there
    while failures < attempts:
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        before = have
        try:
            with opener(_request(url, have), 60) as answer:
                status = getattr(answer, "status", 200)
                if have and status != 206:  # the server sent the whole file again
                    have = before = 0
                with part.open("ab" if have else "wb") as out:
                    while block := answer.read(READ):
                        out.write(block)
                        have += len(block)
                        if seen:
                            seen(have)
            if have >= size:
                break
            problem = OSError(f"the connection ended at {have:,} of {size:,} bytes")
        except urllib.error.HTTPError as err:
            if err.code in (403, 404, 410):
                raise DatasetError(f"{target.name} is not on the server ({_reason(err)})") from None
            problem = err
        except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
            problem = err
        have = part.stat().st_size if part.is_file() else 0
        failures = 0 if have > before else failures + 1
        if failures < attempts:
            wait(min(60.0, 2.0 * 2**failures))
    have = part.stat().st_size if part.is_file() else 0
    if have != size:
        why = _reason(problem) if problem else "it stopped early"
        raise DatasetError(f"could not download {target.name} ({why}). {NETWORK_HINT}")
    digest = hashlib.sha256()
    with part.open("rb") as file:
        while block := file.read(READ):
            digest.update(block)
    if digest.hexdigest() != sha256:
        part.unlink()
        raise DatasetError(
            f"{target.name} arrived damaged (its SHA-256 differs); run it again to fetch it anew"
        )
    part.replace(target)


def unpack(zip_path: Path, chunk: Chunk, into: Path, newest: float) -> int:
    """The chunk's listed replays from `zip_path` into `into` (nothing else in it is read), the best one
    dated `newest` and each next one a second older. Returns how many were written."""
    wanted = {f"{i}.replay": rank for rank, (i, _) in enumerate(chunk.replays)}
    into.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            rank = wanted.get(info.filename)
            if rank is None or info.is_dir():
                continue
            target = into / info.filename
            if not target.is_file():
                partial = target.with_name(target.name + ".partial")
                with archive.open(info) as source, partial.open("wb") as out:
                    shutil.copyfileobj(source, out, READ)
                partial.replace(target)
                written += 1
            stamp = newest - rank
            os.utime(target, (stamp, stamp))
    return written


def have_replay(replay_id: str, size: str) -> bool:
    """Downloaded before, or converted into either store."""
    if (size_folder(size) / f"{replay_id}.replay").is_file():
        return True
    return any(replays.find_stored(replay_id, store=replays.store_folder(fit)) for fit in (True, False))


def plan(
    manifest: Manifest, size: str, count: int, have: Callable[[str, str], bool] = have_replay
) -> list[Chunk]:
    """The chunks to fetch for `count` replays of `size` not had yet (0 = all), best first."""
    chosen: list[Chunk] = []
    new = 0
    for chunk in manifest.of_size(size):
        if count and new >= count:
            break
        missing = sum(not have(i, size) for i, _ in chunk.replays)
        if missing:
            chosen.append(chunk)
            new += missing
    return chosen


def space_needed(chunk: Chunk, convert: bool) -> int:
    """Disk a chunk needs: its zip, its replays, and (converting) their conversions and the store's margin."""
    extra = len(chunk.replays) * CONVERTED_BYTES + replays.STORE_MIN_FREE if convert else 0
    return chunk.bytes + chunk.unpacked + extra


@dataclass
class Fetching:
    """A download so far, for the dashboard."""

    size: str
    chunks: int = 0
    chunk: int = 0  # the one being fetched, from 1
    bytes_total: int = 0
    bytes_done: int = 0  # of the chunks fetched now
    replays_new: int = 0
    began: float = field(default_factory=time.time)

    def view(self, state: str, note: str = "") -> dict[str, Any]:
        seconds = max(time.time() - self.began, 1.0)
        rate = self.bytes_done / seconds
        left = (self.bytes_total - self.bytes_done) / rate if rate > 0 else None
        return {
            "state": state,
            "size": self.size,
            "chunks": self.chunks,
            "chunk": self.chunk,
            "bytes_total": self.bytes_total,
            "bytes_done": self.bytes_done,
            "replays_new": self.replays_new,
            "mb_per_second": round(rate / 2**20, 1),
            "seconds_left": round(left) if left is not None else None,
            "note": note,
            "updated": time.time(),
        }


def _write_progress(view: dict[str, Any]) -> None:
    live = progress_file()
    live.parent.mkdir(parents=True, exist_ok=True)
    partial = live.with_name(live.name + ".partial")
    partial.write_text(json.dumps(view), encoding="utf-8")
    partial.replace(live)


def get_replays(
    size: str,
    count: int,
    url: str = MANIFEST_URL,
    opener: Opener = _open,
    say: Say = print,
    convert: bool = True,
    free_bytes: Callable[[Path], int] = lambda p: shutil.disk_usage(p).free,
    wait: Callable[[float], None] = time.sleep,
) -> Fetching:
    """Downloads and unpacks the chunks holding `count` replays of `size` not had yet (0 = every one
    published), best first. Converting is left to the caller (`convert` only sizes the disk check)."""
    if size not in SIZES:
        raise DatasetError(f"--size is one of {', '.join(SIZES)}, not {size!r}")
    manifest = fetch_manifest(url, opener, say)
    published = manifest.available().get(size)
    if not published:
        raise DatasetError(
            f"no {size} replays are published (there are: "
            + (", ".join(f"{k} {v['replays']:,}" for k, v in sorted(manifest.available().items())) or "none")
            + ")"
        )
    chunks = plan(manifest, size, count)
    status = Fetching(size, chunks=len(chunks), bytes_total=sum(c.bytes for c in chunks))
    say(manifest.credit)
    if not chunks:
        say(f"Every {size} replay asked for is here already ({published['replays']:,} published).")
        _write_progress(status.view("done", "nothing new to download"))
        return status
    say(
        f"Downloading {len(chunks)} of {published['chunks']} {size} chunks "
        f"({status.bytes_total / 2**30:.1f} GB) into {Path(*size_folder(size).parts[-3:])}."
    )
    base = url.rsplit("/", 1)[0]
    ranks: dict[str, int] = {}
    seen_before = 0
    for c in manifest.of_size(size):
        ranks[c.name], seen_before = seen_before, seen_before + len(c.replays)
    downloads = folder() / "downloads"
    newest = float(NEWEST)
    done_before = 0
    last = 0.0
    for number, chunk in enumerate(chunks, 1):
        status.chunk = number
        downloads.mkdir(parents=True, exist_ok=True)
        part = downloads / f"{chunk.name}.part"
        held = part.stat().st_size if part.is_file() else 0
        need = space_needed(chunk, convert) - held
        free = free_bytes(downloads)
        if free < need:
            note = (
                f"The disk has {free / 2**30:.1f} GB free; the next chunk needs about {need / 2**30:.1f} GB "
                f"(download, unpack{', convert and the store margin' if convert else ''}). Stopped after "
                f"{status.replays_new:,} new replays: free some space and run it again to carry on."
            )
            say(note)
            _write_progress(status.view("disk_full", note))
            return status

        def seen(have: int, before: int = done_before) -> None:
            nonlocal last
            status.bytes_done = before + have
            if time.time() - last >= 2:
                last = time.time()
                _write_progress(status.view("downloading"))

        zip_path = downloads / chunk.name
        say(f"Chunk {number} of {len(chunks)}: {chunk.name} ({chunk.bytes / 2**20:,.0f} MB)")
        try:
            download(f"{base}/{chunk.name}", zip_path, chunk.bytes, chunk.sha256, opener, seen, wait=wait)
            rank = ranks[chunk.name]
            status.replays_new += unpack(zip_path, chunk, size_folder(size), newest - rank)
        except DatasetError as err:
            _write_progress(status.view("failed", str(err)))
            raise
        except (OSError, zipfile.BadZipFile) as err:
            _write_progress(status.view("failed", str(err)))
            raise DatasetError(f"could not unpack {chunk.name} ({type(err).__name__}: {err})") from None
        finally:
            with contextlib.suppress(OSError):
                if zip_path.is_file():
                    zip_path.unlink()
        done_before += chunk.bytes
        status.bytes_done = done_before
        _write_progress(status.view("downloading"))
    say(f"Downloaded {status.replays_new:,} new {size} replays.")
    _write_progress(status.view("downloaded"))
    return status
