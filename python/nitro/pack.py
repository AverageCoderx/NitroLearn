"""`nitro pack` and `nitro unpack`: a bot moved to another computer and back, such as a rented cloud machine
.

A pack is one zip: the run folder with its newest save (or every save), the data files its settings read
(recordings, by their paths relative to the folder the bot's settings came from), the source of its own Rust
rewards and, only when asked for, the arena meshes it trains with (game files: a pack with them is for your
own computers only, never to pass on). It names no folder of
this computer: a full path in the bot's settings, saves or notes is written relative to the bot's folder,
the folder its settings came from or NitroLearn's folder, else by its name alone. Unpacking puts those
data files in the run folder (`settings-files/`) where the settings' relative paths find them, and points
the run at that computer's collision meshes, so `nitro resume` carries on there. Nothing is ever
overwritten: a pack file of that name is refused, and a run of that name is refused (or, with `replace`,
kept aside in `runs/.replaced/`). Every name in a pack is checked before anything is written, so a pack can
only write inside the bot's own folders."""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import time
import tomllib
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from nitro import __version__, rebuild, tracking
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.rewards import Custom, custom_kinds
from nitro.config.schema import NitroConfig
from nitro.runs import RunFolder
from nitro.viewer import LIVE_PAGE

ROOT = Path(__file__).resolve().parents[2]  # NitroLearn's folder
BUILT_IN = ROOT / "crates" / "nitro_core" / "src" / "custom"  # rewards compiled into every NitroLearn
FORMAT = 1
SUFFIX = ".nitropack.zip"
MANIFEST = "pack.json"
SETTINGS_FILES = "settings-files"  # in an unpacked run folder: the files its settings read
REPLACED = ".replaced"  # in the runs folder: runs an unpack with `replace` set aside
RUST = Path("my-settings") / "rust"  # where an unpack puts the source of the bot's own Rust rewards
WANDB = "wandb_run.json"  # nitro.tracking.STATE_FILE: which W&B run the bot reports to
# Not packed: what belongs to this computer only (a live window's page holds the arena's meshes, which
# travel in a pack only when asked for; kept recordings of games are for watching here; "cards" holds a run
# on several graphics cards' start file and helpers' logs; W&B's own folder names this computer, its folders
# and its user; wandb_run.json, which carries the W&B run on, travels), and the logs of what ran here: the
# commands' console (their command lines name this computer's folders), the trainer's log and how its last
# start-again after an Apply went.
_LOGS = {"console.log", "train.log", "apply_restart.json"}
_SKIPPED = {"running.json", "stop.request", SETTINGS_FILES, LIVE_PAGE, "recordings", "cards", "wandb", *_LOGS}
_SECTIONS = ("run/", "files/", "rust/", "meshes/")
_FULL_PATH = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")  # a full path on Linux, macOS or Windows
_WINDOWS_PATH = re.compile(r"^([A-Za-z]:[\\/]|\\\\)")  # one on Windows: a drive letter, or \\server\share


class PackError(ValueError):
    """The bot cannot be packed or the pack cannot be unpacked; the message says why."""


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("+")[0].split(".") if part.isdigit())


def _defaults(value: Any) -> Any:
    """The settings' names and default values (paths left out: they differ between computers)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _defaults(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, tuple | list):
        return [_defaults(item) for item in value]
    return "path" if isinstance(value, Path) else repr(value)


def settings_fingerprint() -> str:
    """Changes whenever a setting is added, removed or gets another default: the version number of
    NitroLearn does not (it stays 1.0.0 between releases)."""
    text = json.dumps(_defaults(NitroConfig()), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _core_hash() -> str | None:
    from nitro import _core

    found = getattr(_core, "SOURCE_HASH", None)
    return found if isinstance(found, str) else None


def _paths(value: Any, at: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Path]]:
    """Every path in the settings, with where it is (field names and list positions)."""
    if isinstance(value, Path):
        yield at, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _paths(getattr(value, field.name), (*at, field.name))
    elif isinstance(value, tuple | list):
        for k, item in enumerate(value):
            yield from _paths(item, (*at, k))


def _written(raw: Any, at: tuple[Any, ...]) -> str | None:
    """The text the settings file has at `at`, or None where it leaves the setting at its default (a list
    of paths may be written as one string)."""
    for key in at:
        if isinstance(raw, str) and key == 0:
            continue
        try:
            raw = raw[key]
        except (KeyError, IndexError, TypeError):
            return None
    return raw if isinstance(raw, str) else None


def _absolute(text: str) -> bool:
    return PureWindowsPath(text).is_absolute() or PurePosixPath(text).is_absolute() or ":" in text


def _as_written(path: str, quote: str = "") -> re.Pattern[str]:
    """The full path `path` wherever a text writes it, between `quote`s: on Windows (a drive letter or a
    server's share) with either slash, its backslashes doubled as inside "..." in TOML and JSON, in any
    letter case, as Windows reads a path; elsewhere as it is, or as "..." escapes it."""
    if _WINDOWS_PATH.match(path):
        between = r"(?:\\\\|[\\/])".join(re.escape(part) for part in re.split(r"[\\/]", path))
        return re.compile(re.escape(quote) + between + re.escape(quote), re.IGNORECASE)
    forms = dict.fromkeys((path, json.dumps(path, ensure_ascii=False)[1:-1]))
    return re.compile(re.escape(quote) + "(?:" + "|".join(map(re.escape, forms)) + ")" + re.escape(quote))


def _replaced(pattern: re.Pattern[str], by: str, text: str) -> str:
    """`text` with `by` wherever `pattern` matches it (the same object when it matches nowhere)."""
    changed, count = pattern.subn(lambda _: by, text)
    return changed if count else text


@dataclasses.dataclass(frozen=True)
class _Here:
    """The folders of this computer a bot's files may name, and how a pack names them instead."""

    bases: tuple[PurePath, ...]  # a full path inside one is carried relative to it
    named: tuple[tuple[re.Pattern[str], str], ...]  # a folder as texts write it, and what they say instead

    @classmethod
    def of(cls, run: Path, settings_dir: Path) -> _Here:
        """The bot's folder `run`, the folder its settings came from and NitroLearn's folder, each also as
        its links resolve; in texts also the home folder, as ~."""
        folders = (run, settings_dir, ROOT)
        return cls.made(
            [form for folder in folders for form in (Path(os.path.abspath(folder)), folder.resolve())],
            Path.home(),
        )

    @classmethod
    def made(cls, bases: Iterable[PurePath], home: PurePath) -> _Here:
        """`of` with these folders and this home folder, which may be another system's (Windows paths
        on Linux, in the tests)."""
        kept = tuple(dict.fromkeys(bases))
        named = {str(base): base.name for base in kept}
        if len(home.parts) > 1:
            named[str(home)] = "~"
        longest = sorted(named.items(), key=lambda pair: -len(pair[0]))  # a folder before the one it is in
        return cls(kept, tuple((_as_written(folder), name) for folder, name in longest))

    def text(self, text: str) -> str:
        """`text` with the folders of this computer in it by their names (the same object when it names
        none)."""
        for folder, name in self.named:
            text = _replaced(folder, name, text)
        return text

    def path(self, text: str) -> str:
        """`text` as a pack carries it: a full path relative to the folder of `bases` it is in, else by its
        name alone; any other text as `text` gives it."""
        if not _FULL_PATH.match(text):
            return self.text(text)
        for base in self.bases:
            path = type(base)(text)  # read as the base's system reads it (on Windows: either slash, any case)
            if path.is_relative_to(base):
                return path.relative_to(base).as_posix() or "."
        return re.split(r"[\\/]", text.rstrip("\\/"))[-1]

    def value(self, value: Any) -> Any:
        """`value` (what a save or a JSON file holds) with every text in it as `path` gives it; the same
        object when nothing in it changes."""
        if isinstance(value, str):
            return self.path(value)
        if isinstance(value, dict):
            carried = {key: self.value(item) for key, item in value.items()}
            return value if all(carried[key] is item for key, item in value.items()) else carried
        if isinstance(value, list | tuple):
            items = [self.value(item) for item in value]
            if all(new is old for new, old in zip(items, value, strict=True)):
                return value
            return tuple(items) if isinstance(value, tuple) else items
        return value


def data_files(folder: RunFolder, config: NitroConfig) -> tuple[list[tuple[Path, str]], list[str]]:
    """The files the run's settings read that a pack carries, each with its path relative to the folder the
    settings came from, and notes on the ones it cannot carry."""
    settings_dir = Path(json.loads(folder.info.read_text(encoding="utf-8"))["settings_dir"])
    raw = tomllib.loads(folder.settings.read_text(encoding="utf-8-sig"))
    found: dict[str, Path] = {}
    notes = []
    for at, _ in _paths(config):
        where = ".".join(str(part) for part in at)
        if at[0] == "paths":
            continue  # the meshes: each computer has its own (an unpack points the run at them)
        if at[0] == "pretrain":
            continue  # the replays are copied once, before the first save
        text = _written(raw, at)
        if text is None:
            continue  # a default: nothing the settings file points at
        if _absolute(text):
            raise PackError(
                f"{where} = '{text}' is a full path, which another computer does not have: move the file "
                f"into {settings_dir} (where the bot's settings came from) or a folder below it, write its "
                "path relative to that folder, then pack again"
            )
        if "\\" in text:
            raise PackError(
                f"{where} = '{text}' is written with \\, which only Windows reads as a folder separator: "
                "write it with / (as in recordings/match.rlpr), then pack again"
            )
        # The path as the settings write it, not where links lead: a linked starts folder keeps its name,
        # so the unpacked settings find the file under the same path.
        written = Path(os.path.normpath(settings_dir / text))
        if written.is_dir():
            setting = ".".join(str(part) for part in (at[:-1] if isinstance(at[-1], int) else at))
            files = [f for f in written.rglob("*") if f.is_file()]
            count = f"{len(files):,} file{'s' if len(files) != 1 else ''}"
            notes.append(
                f"{setting} = '{text}' is a folder ({count}, {_size(files)}): packs carry files only; copy "
                f"it to the other computer and point {setting} at it there"
            )
            continue
        if written.is_file():
            matches = [written]
        elif glob.has_magic(text):
            matches = sorted(Path(m) for m in glob.glob(str(written)))
        else:
            matches = []
        if not matches:
            notes.append(f"{where}: no file {text} here, so nothing is packed for it")
        for file in matches:
            if file.is_file():
                found[Path(os.path.relpath(file, settings_dir)).as_posix()] = file
    return [(file, rel) for rel, file in sorted(found.items())], notes


def _size(files: list[Path]) -> str:
    total = sum(f.stat().st_size for f in files)
    return f"{total / 1e9:.1f} GB" if total >= 1e9 else f"{total / 1e6:.1f} MB"


def _replay_folders_gone(run: Path) -> list[str]:
    """Notes on the replay_mix folders the unpacked bot's settings point at that are not here."""
    from nitro.replay_mix import folders

    try:
        config = RunFolder(run).load_settings()
    except (ConfigError, OSError, ValueError):
        return []
    if not config.replay_mix.active:
        return []
    return [
        f"replay_mix will look for its replays in {place}, which is not here: copy them there, or point "
        "replay_mix.replays at where they are"
        for place in folders(config)
        if not place.is_dir()
    ]


def _files_of(folder: Path) -> list[Path]:
    """Every file in `folder`, following linked folders (a checkpoints folder on another drive) but not a
    link to something inside `folder` itself (W&B's latest-run), which would be packed twice."""
    top = folder.resolve()

    def inside(path: Path) -> bool:
        return path.is_symlink() and path.resolve().is_relative_to(top)

    found = []
    for here, folders, names in os.walk(folder, followlinks=True):
        folders[:] = sorted(name for name in folders if not inside(Path(here) / name))
        found += [Path(here) / name for name in sorted(names) if not inside(Path(here) / name)]
    return found


def pack(
    run: Path, out: Path | None = None, all_saves: bool = False, meshes: bool = False
) -> tuple[Path, list[str]]:
    """Packs the run folder `run` into one zip (default: `<name>-<date-time>.nitropack.zip` in the current
    folder; an existing file is never replaced), with the arena meshes only when `meshes` is on; returns
    its path and notes on what it did not carry."""
    from nitro.dashboard.bots import running

    folder = RunFolder(run)
    if not folder.info.is_file():
        raise PackError(f"{run} is not a NitroLearn bot (it has no run.json)")
    if running(folder) is not None:
        raise PackError(f"{run.name} is training: stop it first, so the pack holds its newest save")
    info = json.loads(folder.info.read_text(encoding="utf-8"))
    saves = folder.complete_checkpoints()
    config = folder.load_settings()
    if not saves and config.start_from.source:
        raise PackError(
            f"{run.name} has no save yet and starts from another bot "
            f"({config.start_from.source}): start it here until its first save (a few minutes), then pack it"
        )
    if (not saves and config.pretrain.enabled) or folder.copy_so_far() is not None:
        raise PackError(f"{run.name} has not finished copying its replays: do that here first ([pretrain])")
    files, notes = data_files(folder, config)
    missing = missing_rust_files(config)
    if missing:
        reward, source = missing[0]
        raise PackError(
            f"the reward '{reward}' is built from {source}, which is not there any more: put the file back, "
            "so the pack can carry it to the other computer, then pack again"
        )
    own = own_rust_files(config)
    kept = [saves[-1]] if saves and not all_saves else saves
    if len(saves) > len(kept):
        left = len(saves) - len(kept)
        notes.append(f"{left} older save{'s' if left != 1 else ''} left out (--all-saves packs every save)")
    if offline := len(tracking.offline_runs(run)):
        sessions = (
            "1 W&B session kept offline stays"
            if offline == 1
            else f"{offline} W&B sessions kept offline stay"
        )
        notes.append(
            f"{sessions} on this computer (W&B's own files name its folders and user): send them from here "
            f"with nitro wandb {run.name}"
        )
    arena = config.paths.collision_meshes
    here = _Here.of(run, Path(info["settings_dir"]))
    if meshes and not (arena / "soccar").is_dir():
        notes.append(f"no arena meshes in {here.path(str(arena))}, so none are packed")
    arena_files = _files_of(arena) if meshes and (arena / "soccar").is_dir() else []
    out = out or Path.cwd() / f"{run.name}-{time.strftime('%Y-%m-%d-%H%M')}{SUFFIX}"
    if out.exists():
        raise PackError(f"{out} already exists: it is kept; pack to another file (--to)")
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "nitro_version": __version__,
        "settings_fingerprint": settings_fingerprint(),
        "core": _core_hash(),
        "run": run.name,
        "made": time.strftime("%Y-%m-%d %H:%M"),
        "id": secrets.token_hex(8),  # this pack and no other (two made in one minute)
        "saves": [save.name for save in kept],
        "files": [{"rel": rel, "zip": f"files/{k}/{file.name}"} for k, (file, rel) in enumerate(files)],
        "rust": [f"rust/{path.name}" for path in own],
        "meshes": bool(arena_files),
        "notes": here.value(notes),
    }
    partial = out.with_name(out.name + ".partial")
    spare = out.with_name(out.name + ".save")  # a save written anew without this computer's folders
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in _files_of(run):
                inside = path.relative_to(run)
                name = f"run/{inside.as_posix()}"
                if inside.parts[0] in _SKIPPED or path.name.endswith(".partial"):
                    continue
                if inside == Path(folder.info.name):  # this computer's folders stay here
                    zipped.writestr(name, json.dumps(here.value(_travelling(info))))
                    continue
                if inside.parts[0] == "checkpoints" and inside.parts[1] not in manifest["saves"]:
                    continue  # older saves, and any a stopped save left unfinished
                if path.suffix == ".pt":  # saves barely compress and are the bulk: stored
                    carried = _carried_save(path, here, spare)
                    zipped.write(carried, name, compress_type=zipfile.ZIP_STORED)
                    spare.unlink(missing_ok=True)
                elif path.suffix == ".toml":
                    zipped.writestr(name, _carried_settings(path.read_text(encoding="utf-8-sig"), here))
                elif path.suffix in _TEXTS:
                    zipped.writestr(name, _carried_text(path.read_bytes(), path.suffix, here))
                else:
                    zipped.write(path, name)
            missing = [
                s for s in manifest["saves"] if f"run/checkpoints/{s}/state.pt" not in zipped.namelist()
            ]
            if missing:
                raise PackError(
                    f"the save {missing[0]} could not be read from {folder.checkpoints}: not packed"
                )
            for entry, (file, _) in zip(manifest["files"], files, strict=True):
                zipped.write(file, entry["zip"])
            for path in own:
                zipped.write(path, f"rust/{path.name}")
            for path in arena_files:
                zipped.write(path, f"meshes/{path.relative_to(arena).as_posix()}")
            zipped.writestr(MANIFEST, json.dumps(manifest, indent=1))
        if out.exists():
            raise PackError(f"{out} appeared while packing: it is kept; pack to another file (--to)")
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)
        spare.unlink(missing_ok=True)
    return out, notes


_TEXTS = (".json", ".log", ".csv", ".txt", ".html")  # a bot's files a pack carries as `_carried_text`


def _carried_save(path: Path, here: _Here, spare: Path) -> Path:
    """The save (or other PyTorch file) at `path` as a pack carries it: the file itself when it names no
    folder of this computer (or cannot be read), else `spare` with it written anew without them."""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # not a save PyTorch reads: carried as it is
        return path
    carried = here.value(state)
    if carried is state:
        return path
    torch.save(carried, spare)
    return spare


def _carried_text(data: bytes, suffix: str, here: _Here) -> bytes:
    """A JSON file with `here.value` applied to what it holds, another text with the folders of this
    computer by their names; as it is when it is neither."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if suffix == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            return data
        carried = here.value(value)
        return data if carried is value else json.dumps(carried, indent=1).encode("utf-8")
    carried = here.text(text)
    return data if carried is text else carried.encode("utf-8")


def _carried_settings(text: str, here: _Here) -> str:
    """The bot's settings file with every full path in its tables (the meshes, the replays copied once)
    as `here.path` gives it, wherever the file names it, and any other folder of this computer `here`
    knows by its name."""
    try:
        full = _full_paths(tomllib.loads(text))
    except tomllib.TOMLDecodeError:
        return here.text(text)
    for value in sorted(full, key=len, reverse=True):  # a/b/c before a/b, which a/b/c starts with
        shown = here.path(value)
        for quote, carried in (
            ('"', json.dumps(shown, ensure_ascii=False)),  # "a\\b"
            ("'", f"'{shown}'"),  # 'a\b', as written
            ("", shown),  # in a comment, or with the other slash elsewhere
        ):
            text = _replaced(_as_written(value, quote), carried, text)
    text = here.text(text)
    if left := _full_paths(tomllib.loads(text)):
        raise PackError(
            f"the bot's settings hold the full path '{left[0]}' in a form a pack cannot rewrite: write it "
            'as a plain "..." text, then pack again'
        )
    return text


def _full_paths(value: Any) -> list[str]:
    """Every text in `value` (a settings file's tables) that is a full path."""
    if isinstance(value, str):
        return [value] if _FULL_PATH.match(value) else []
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return [found for item in items for found in _full_paths(item)]


def _travelling(info: dict[str, Any]) -> dict[str, Any]:
    """run.json as a pack carries it: without the folders of this computer (where the settings came from,
    where its meshes are), which an unpack sets for the computer it unpacks on."""
    local = ("settings_dir", "settings_file", "settings_copied", "collision_meshes")
    return {key: value for key, value in info.items() if key not in local}


@dataclasses.dataclass
class Unpacked:
    folder: Path
    notes: list[str]
    rust: list[Path]  # the bot's own Rust rewards added here: build them in before it trains


def _open(archive: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise PackError(
            f"{archive} is not a NitroLearn pack (not a zip file, or not fully downloaded)"
        ) from None


def _read_manifest(zipped: zipfile.ZipFile, archive: Path) -> dict[str, Any]:
    try:
        found: dict[str, Any] = json.loads(zipped.read(MANIFEST))
    except (KeyError, ValueError, zipfile.BadZipFile):
        raise PackError(f"{archive} is not a NitroLearn pack (it has no readable {MANIFEST})") from None
    return found


def manifest(archive: Path) -> dict[str, Any]:
    """What the pack says about itself (its bot's name, saves, files)."""
    with _open(archive) as zipped:
        return _read_manifest(zipped, archive)


def unpacked_here(archive: Path, runs: Path) -> bool:
    """Whether the bot in `runs` was unpacked from this same pack (tools/cloud_setup.sh run again)."""
    try:
        made = manifest(archive)
        info = json.loads((runs / made["run"] / "run.json").read_text(encoding="utf-8"))
    except (PackError, OSError, ValueError, KeyError, TypeError):
        return False
    if made.get("id") is not None:
        return bool(info.get("pack_id") == made["id"])
    return bool(info.get("unpacked") == made.get("made"))


def _plain(rest: str, climb: bool = False) -> PurePosixPath | None:
    """`rest` (a name inside a pack) as a plain relative path, or None when it is not one: empty parts,
    "." or "..", a drive or a backslash anywhere. With `climb`, leading "../" are allowed (data files)."""
    parts = rest.split("/")
    if climb:
        while parts and parts[0] == "..":
            parts = parts[1:]
    if not parts or any(not part or part in (".", "..") or ":" in part or "\\" in part for part in parts):
        return None
    return PurePosixPath(rest)


def _check(archive: Path, zipped: zipfile.ZipFile, found: dict[str, Any]) -> None:
    """Refuses a pack with any name that could write outside the bot's own folders, before anything is
    written."""
    names = set(zipped.namelist())
    for member in names - {MANIFEST}:
        if member in _SECTIONS:
            continue
        if member.endswith("/"):
            member = member[:-1]
        section, _, rest = member.partition("/")
        if f"{section}/" not in _SECTIONS or _plain(rest) is None:
            raise PackError(f"{archive} holds '{member}', which a pack never does: not unpacked")
    for entry in found.get("files", []):
        if _plain(entry["rel"], climb=True) is None or entry["zip"] not in names:
            raise PackError(f"{archive} holds '{entry['rel']}', which a pack never does: not unpacked")
    for member in found.get("rust", []):
        if member not in names or not member.endswith(".rs") or len(PurePosixPath(member).parts) != 2:
            raise PackError(f"{archive} holds '{member}', which a pack never does: not unpacked")


def unpack(
    archive: Path,
    runs: Path,
    name: str = "",
    replace: bool = False,
    project: Path | None = None,
    meshes: Path | None = None,
) -> Unpacked:
    """Unpacks `archive` into the runs folder `runs` as bot `name` (default: its own name). `project` is
    NitroLearn's folder (its my-settings); `meshes` the collision meshes the bot uses here
    (default: the project's)."""
    from nitro.dashboard.bots import running

    project = project or ROOT
    meshes = meshes or project / "collision_meshes"
    with _open(archive) as zipped:
        found = _read_manifest(zipped, archive)
        if found.get("format") != FORMAT:
            raise PackError(
                f"{archive} is a pack of format {found.get('format')}; this NitroLearn reads {FORMAT}"
            )
        if _version(found["nitro_version"]) > _version(__version__):
            raise PackError(
                f"the pack was made by NitroLearn {found['nitro_version']}, newer than this one "
                f"({__version__}): update NitroLearn here first (git pull, then build it)"
            )
        _check(archive, zipped, found)
        notes = _same_code(zipped, found)
        name = name or found["run"]
        if not name or _plain(name) is None or "/" in name or name.startswith("."):
            raise PackError(f"'{name}' cannot name a bot's folder")
        runs = runs.resolve()
        target = runs / name
        if target.exists() and not replace:
            raise PackError(
                f"there is already a bot called '{name}' in {runs}: unpack it under another name (--name) "
                "or with --replace (the bot here is then kept in the runs folder's .replaced)"
            )
        if target.exists() and running(RunFolder(target)) is not None:
            raise PackError(f"{name} is training here: stop it first, then replace it")
        runs.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f".unpack-{name}-", dir=runs))
        try:
            for member in zipped.namelist():
                if member.startswith("run/") and not member.endswith("/"):
                    _write_inside(scratch, member[len("run/") :], zipped.read(member))
            # The settings' relative paths: from a folder deep enough that every "../" stays inside.
            depth = max((PurePosixPath(f["rel"]).parts.count("..") for f in found["files"]), default=0)
            inside = Path(SETTINGS_FILES, *(["up"] * depth))
            for entry in found["files"]:
                path = (scratch / inside / entry["rel"]).resolve()
                if not path.is_relative_to((scratch / SETTINGS_FILES).resolve()):
                    raise PackError(
                        f"{archive} holds '{entry['rel']}', which a pack never does: not unpacked"
                    )
                _write(path, zipped.read(entry["zip"]))
            notes += found.get("notes", [])
            if name != found["run"] and (scratch / WANDB).is_file():
                (scratch / WANDB).unlink()  # a copy beside the original: its own W&B run
                notes.append(f"unpacked as '{name}': it starts a W&B run of its own")
            info = json.loads((scratch / "run.json").read_text(encoding="utf-8"))
            info["settings_dir"] = str(target / inside)
            info["collision_meshes"] = str(meshes)
            info["unpacked"] = found["made"]
            info["pack_id"] = found.get("id")
            (scratch / "run.json").write_text(json.dumps(info), encoding="utf-8")
            # The step most likely to fail (Windows refuses to move a folder a program has a file open in)
            # comes first, so a failure changes nothing that other bots use.
            if target.exists():
                if not replace:
                    raise PackError(f"a bot called '{name}' appeared in {runs} while unpacking: not replaced")
                aside = runs / REPLACED / f"{name}-{time.strftime('%Y-%m-%d-%H%M%S')}"
                aside.parent.mkdir(parents=True, exist_ok=True)
                target.rename(aside)
                notes.append(f"the bot that was here is kept in {aside}")
            scratch.rename(target)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        notes += _install_meshes(zipped, meshes)
        if not (meshes / "soccar").is_dir():
            notes.append(f"no collision meshes in {meshes} yet: get them with nitro fetch-meshes")
        rust, more = _add_rust(zipped, found["rust"], project, target / "rust")
        notes += more
    notes += _replay_folders_gone(target)
    return Unpacked(target, notes, rust)


def _same_code(zipped: zipfile.ZipFile, found: dict[str, Any]) -> list[str]:
    """Refuses a pack whose settings this NitroLearn cannot read (a setting it does not know); a note when
    the pack was made by other code (settings' defaults or the Rust part may differ)."""
    if found.get("settings_fingerprint") in (None, settings_fingerprint()):
        return []
    try:
        text = zipped.read("run/settings.toml").decode("utf-8-sig")
        with tempfile.TemporaryDirectory() as scratch:
            parse_toml(text, NitroConfig, file=Path("settings.toml"), base_dir=Path(scratch))
    except KeyError:
        return []
    except ConfigError as err:
        unknown = [issue for issue in err.issues if "is not a known setting" in issue.message]
        if unknown:
            raise PackError(
                f"the pack was made by a newer NitroLearn: '{unknown[0].path}' is not a setting this one "
                "knows. Update NitroLearn here first (git pull, then build it), so both computers run the "
                "same version"
            ) from None
    return [
        "the pack was made by a different version of NitroLearn: some settings' defaults may differ here. "
        "Best to run the same version on both computers (update the older one)"
    ]


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_inside(root: Path, rest: str, data: bytes) -> None:
    """Writes `data` at `root/rest`, a plain relative name (checked again: nothing lands outside `root`)."""
    path = (root / rest).resolve()
    if _plain(rest) is None or not path.is_relative_to(root.resolve()):
        raise PackError(f"'{rest}' would land outside {root}: not unpacked")
    _write(path, data)


def _install_meshes(zipped: zipfile.ZipFile, meshes: Path) -> list[str]:
    """The pack's arena meshes, into `meshes` when it has none (meshes already there are never touched)."""
    members = [m for m in zipped.namelist() if m.startswith("meshes/") and not m.endswith("/")]
    if not members or (meshes / "soccar").is_dir():
        return []
    for member in members:
        target = (meshes / member[len("meshes/") :]).resolve()
        if not target.exists():
            _write_inside(meshes, member[len("meshes/") :], zipped.read(member))
    return [f"the arena meshes came with the pack: unpacked into {meshes}"]


def _add_rust(
    zipped: zipfile.ZipFile, members: list[str], project: Path, copy: Path
) -> tuple[list[Path], list[str]]:
    """The bot's own Rust rewards (`rust/<file>`): a copy in the run folder (`copy`), and each file name added
    once to the next build unless a different file of that name is already listed or in my-settings/rust."""
    if not members:
        return [], []
    listed = rebuild.listed(project)
    by_name = {path.name: path for path in listed}
    add, notes = [], []
    for member in members:
        data = zipped.read(member)
        name = PurePosixPath(member).name
        kept = copy
        _write_inside(kept, name, data)
        known = by_name.get(name)
        if known is not None:
            if not (known.is_file() and known.read_bytes() == data):
                notes.append(
                    f"{name}: a different file of that name is built in here; the packed one is in {kept}"
                )
            continue
        mine = project / RUST / name
        if mine.exists() and mine.read_bytes() != data:
            notes.append(
                f"{name}: a different file of that name is in {mine.parent}; the packed one is in {kept}"
            )
            continue
        if not mine.exists():
            _write(mine, data)
        add.append(mine.resolve())
        by_name[name] = mine.resolve()
    if add:
        rebuild.write_list(project, [*listed, *add])
    return add, notes


def own_rust_files(config: NitroConfig) -> list[Path]:
    """The source files of the rewards of one's own `config` uses (not the built-in library's)."""
    kinds = custom_kinds()
    files = set()
    for reward in _customs(config.rewards):
        if reward.name in kinds:
            path = Path(kinds[reward.name].file)
            if not path.is_relative_to(BUILT_IN) and path.is_file():
                files.add(path)
    return sorted(files)


def missing_rust_files(config: NitroConfig) -> list[tuple[str, Path]]:
    """The rewards of one's own `config` uses whose source file is gone from where the core was built from
    (each with that path): they still train here, built in, but cannot travel."""
    kinds = custom_kinds()
    missing = {}
    for reward in _customs(config.rewards):
        if reward.name in kinds:
            path = Path(kinds[reward.name].file)
            if not path.is_relative_to(BUILT_IN) and not path.is_file():
                missing[reward.name] = path
    return sorted(missing.items())


def _customs(value: Any) -> list[Custom]:
    """Every reward of one's own in `value`, also those inside all_of and other combinations."""
    if isinstance(value, Custom):
        return [value]
    if isinstance(value, tuple | list):
        return [found for item in value for found in _customs(item)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [found for f in dataclasses.fields(value) for found in _customs(getattr(value, f.name))]
    return []
