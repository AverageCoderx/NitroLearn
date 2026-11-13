"""Where recordings of real games are: the five RocketSim ships (about 25 minutes of 1v1, 2v2 and 3v3
play) and the ones made from replays. They feed the `replay` state setter (nitro/config/state_setters.py) and
`nitro pretrain`.

RocketSim's five come from its repository at the commit the pinned simulator was published from (MIT, like
the rest of that repository), each checked against its pinned SHA-256 hash: from NitroLearn's own release
(rocketsim-recordings.zip, the same five files with RocketSim's licence; a folder of the release's files when
there is one, else the release on GitHub), else straight from RocketSim's repository. They are
downloaded once (by the installer when chosen there, `nitro fetch-recordings`, or by themselves the first
time a run or `nitro pretrain --rocketsim` needs them) into `rocketsim_recordings/` in NitroLearn's folder.
Standard library only: the installer loads this file to see whether they are there."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nitro.config.state_setters import Replay

ROOT = Path(__file__).resolve().parents[2]
# The RocketSim commit crates.io's rocketsim 0.3.3 (the pinned simulator) was published from; its test crate
# holds the recordings.
PINNED_REVISION = "45722876705adf5dc673aa764e80382cfe8c3312"
URL = f"https://raw.githubusercontent.com/ZealanL/RocketSim/{PINNED_REVISION}/rocketsim_test/recordings/"
# name -> (size in bytes, SHA-256)
RECORDINGS = {
    "daizen_2v2_300s.rlpr.zst": (
        16209491,
        "124f1490afb69caea539328efdc87eb98a1dee4c359c6ae969520ec3590ac9e5",
    ),
    "nexto_1v1_300s.rlpr.zst": (12423841, "4b67f4426786a9f1261120cfadf2bd00253f65fe7f4796355a671004f280d2fe"),
    "partycannon_3v3_300s.rlpr.zst": (
        29169828,
        "d31f18f13cd4e0567b39ee2329faedea5b3ce3a35e00d056f265a24e9a86c88e",
    ),
    "veai_3v3_320wall.rlpr.zst": (
        30457239,
        "55567925e0299be0fe089f8b035846e088d456873a7b13dd8cf5574539f11309",
    ),
    "wisp_3v3_300s.rlpr.zst": (30797809, "509b0bf8c9d50ab5db0bd62c476c06f936d798a91432707f2e969fa957bc9ad9"),
}
FOLDER = ROOT / "rocketsim_recordings"
# NitroLearn's release keeps its own copy of the five (tools/bundles.py makes it).
RELEASE_ASSET = "rocketsim-recordings.zip"
RELEASES = "https://github.com/AverageCoderx/NitroLearn/releases/download/v{version}/"
RELEASE_FILES_ENV = "NITRO_RELEASE_FILES"  # a folder of the release's files (the installer sets it)
# A download that sends nothing for this long has stalled (as `nitro fetch-meshes`).
TIMEOUT_SECONDS = 120
ATTEMPTS = 4


class RecordingFetchError(RuntimeError):
    """RocketSim's recordings could not be downloaded, or did not match their pinned hashes."""


def total_mb(names: list[str] | None = None) -> int:
    return round(sum(RECORDINGS[name][0] for name in names or RECORDINGS) / 1e6)


def rocketsim_checkout(revision: str = PINNED_REVISION) -> Path | None:
    """cargo's checkout of RocketSim at that commit (a developer's build may have one), or None."""
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    candidates = sorted((cargo_home / "git" / "checkouts").glob(f"rocketsim-*/{revision[:7]}"))
    return candidates[0] if candidates else None


def _missing(folder: Path) -> list[str]:
    """The recordings not (completely) in `folder`. A file only gets its name once its hash was checked."""
    return [name for name, (size, _) in RECORDINGS.items() if not _has_size(folder / name, size)]


def downloaded(folder: Path | None = None) -> bool:
    """Whether all five are in `folder` (NitroLearn's rocketsim_recordings/)."""
    return not _missing(FOLDER if folder is None else folder)


def remove_downloaded(folder: Path | None = None) -> list[str]:
    """Deletes the five from `folder` (nothing else in it), and the folder when that empties it; returns the
    names deleted."""
    folder = FOLDER if folder is None else folder
    removed = []
    for name in RECORDINGS:
        try:
            (folder / name).unlink()
        except FileNotFoundError:
            continue
        removed.append(name)
    with contextlib.suppress(OSError):
        folder.rmdir()
    return removed


def _has_size(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def bundled_recordings(folder: Path | None = None) -> list[Path]:
    """RocketSim's five real-game recordings (`.rlpr.zst`) on this computer: the downloaded copies, else a
    cargo checkout's; an empty list when neither has all five. Never downloads."""
    folder = FOLDER if folder is None else folder
    if not _missing(folder):
        return [folder / name for name in RECORDINGS]
    checkout = rocketsim_checkout()
    if checkout is not None:
        found = checkout / "rocketsim_test" / "recordings"
        if not _missing(found):
            return [found / name for name in RECORDINGS]
    return []


def _download(url: str, target: Path) -> str:
    """Streams `url` into `target` and returns its SHA-256; a stall of TIMEOUT_SECONDS fails."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as answer, target.open("wb") as out:
        while chunk := answer.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _version(root: Path = ROOT) -> str | None:
    try:
        text = (root / "Cargo.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    found = re.search(
        r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', text, re.MULTILINE | re.DOTALL
    )
    return found[1] if found else None


def _take(archive: Path, folder: Path, missing: list[str]) -> list[str]:
    """The `missing` recordings out of the release's archive into `folder`, each kept only when it matches its
    pinned size and hash; the names placed."""
    placed = []
    with zipfile.ZipFile(archive) as zipped:
        inside = set(zipped.namelist())
        for name in missing:
            if name not in inside:
                continue
            size, wanted = RECORDINGS[name]
            part = folder / f"{name}.{os.getpid()}.part"
            digest = hashlib.sha256()
            try:
                with zipped.open(name) as source, part.open("wb") as out:
                    while chunk := source.read(1 << 20):
                        digest.update(chunk)
                        out.write(chunk)
                if digest.hexdigest() == wanted and part.stat().st_size == size:
                    os.replace(part, folder / name)
                    placed.append(name)
            finally:
                part.unlink(missing_ok=True)
    return placed


def _from_release(folder: Path, missing: list[str]) -> list[str]:
    """The `missing` recordings from NitroLearn's release copy: a folder of the release's files first, else
    the release on GitHub. The names placed; [] without a copy (nothing is said: RocketSim's own copy is
    next)."""
    local = os.environ.get(RELEASE_FILES_ENV)
    archive = Path(local) / RELEASE_ASSET if local else ROOT / "release-files" / RELEASE_ASSET
    if archive.is_file():
        try:
            return _take(archive, folder, missing)
        except (OSError, zipfile.BadZipFile):
            return []
    version = _version()
    if version is None:
        return []
    part = folder / f"{RELEASE_ASSET}.{os.getpid()}.part"
    try:
        _download(RELEASES.format(version=version) + RELEASE_ASSET, part)
        return _take(part, folder, missing)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile):
        return []
    finally:
        part.unlink(missing_ok=True)


def fetch_recordings(
    folder: Path | None = None,
    say: Callable[[str], None] = print,
    download: Callable[[str, Path], str] | None = None,
) -> list[Path]:
    """Puts RocketSim's five recordings in `folder` (NitroLearn's rocketsim_recordings/), downloading only the
    missing ones (NitroLearn's release copy first, then RocketSim's repository), each checked against its
    pinned hash, and returns their paths."""
    folder = FOLDER if folder is None else folder
    download = download or _download
    missing = _missing(folder)
    if not missing:
        say(f"RocketSim's five recordings are already in {folder}.")
        return [folder / name for name in RECORDINGS]
    say(
        f"Downloading RocketSim's real-game recordings ({len(missing)} files, {total_mb(missing)} MB, once) "
        f"from NitroLearn's release (else from github.com/ZealanL/RocketSim) into {folder}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    placed = _from_release(folder, missing)
    for name in [name for name in missing if name not in placed]:
        size, wanted = RECORDINGS[name]
        part = folder / f"{name}.{os.getpid()}.part"
        try:
            for attempt in range(ATTEMPTS):
                try:
                    got = download(URL + name, part)
                    break
                except (urllib.error.URLError, OSError) as err:
                    if attempt == ATTEMPTS - 1:
                        raise RecordingFetchError(
                            f"could not download {URL + name} ({err}). Check the internet connection and "
                            "try again with `nitro fetch-recordings` (antivirus web shields and company "
                            "proxies that scan HTTPS can also block it)"
                        ) from None
                    time.sleep(2 * (attempt + 1))
            if got != wanted or part.stat().st_size != size:
                raise RecordingFetchError(
                    f"the downloaded {name} is not the expected file (its SHA-256 hash differs), so it was "
                    "not kept. Try again later with `nitro fetch-recordings`"
                )
            os.replace(part, folder / name)
        finally:
            part.unlink(missing_ok=True)
    say(f"Put RocketSim's five recordings in {folder} (hashes checked).")
    return [folder / name for name in RECORDINGS]


def needed_recordings(say: Callable[[str], None] = print) -> list[Path]:
    """RocketSim's recordings for something that uses them now: those on this computer, else downloaded."""
    return bundled_recordings() or fetch_recordings(say=say)


def replay_files(setter: Replay) -> list[str]:
    """The recording files a replay setter names: its patterns expanded (sorted, without repeats), plus the
    bundled recordings when asked for. Empty when nothing matches."""
    found: list[str] = []
    for pattern in setter.files:
        matches = sorted(glob.glob(str(pattern)))
        found.extend(matches if matches else ([str(pattern)] if Path(pattern).is_file() else []))
    if setter.bundled:
        found.extend(str(path) for path in needed_recordings())
    return list(dict.fromkeys(found))


def missing_replay_files(setter: Replay) -> str | None:
    """Why a replay setter has nothing to draw from, in plain English; None when it has recordings.
    RocketSim's recordings count as there: a run downloads them when it starts, never a settings check."""
    if setter.bundled:
        return None
    found = [path for pattern in setter.files for path in glob.glob(str(pattern))]
    if found or any(Path(pattern).is_file() for pattern in setter.files):
        return None
    if setter.files:
        return "no recording matches " + ", ".join(str(p) for p in setter.files)
    return "name at least one recording file, or set bundled = true"
