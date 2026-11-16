"""After an install: what an older NitroLearn put in its folder that this version no longer uses is removed,
so a newer version never needs an uninstall first. tools/install_steps.py does this at the end of every
install, so it happens on each way to a newer version: `nitro update` (and the dashboard's Update button), the
setup wizard over an older folder, and the installer run again. Standard library only.

What goes, and nothing else:

- packages in .venv the installer put there that nothing this version installs needs. The installer's
  record (.tools/installed-packages.json) names every package it installed and everything those require
  (their Requires-Dist, with markers and extras); a folder an older installer set up, without a record, gets
  one before the install, from what the NitroLearn installed then requires. Only names in the record can go;
  every other package (one the user installed, with uv or pip) stays, with what it needs, as do uv's seed
  packages (pip, setuptools, wheel) and a package something other than uv installed;
- Pythons in .tools/python other than the one .venv runs on (an older pinned Python), known by uv's names
  for them even when half removed;
- the installer's own leftovers: an unfinished .tools/vendor.new, PyTorch's caches once PyTorch changed or
  went (and the remembered PyTorch build once it went), pip's cache, and cargo's build folder target/ when
  the folder's own Rust (.tools) built it and that Rust is no longer the pinned one, or is gone (a target/
  built with any other Rust is the user's and stays);
- uv's cached downloads of the packages that were removed or replaced (`uv cache clean`; all of them when an
  older Python goes, since they were made for it), and the entries uv no longer uses (`uv cache prune`).

What a tidy could not remove (a file in use, a path too long, antivirus) is listed in .tools/tidy-pending.json
and tried again by every later tidy, while it is still unused.

A path is removed only inside one of the folders the installer owns (.tools, .cache, target), reached
without passing a link; a link is removed itself, never followed. The user's folders (bots, runs, settings,
saves, replays...) and .tmp's update backups are never in reach. A failure is said and the rest goes on: a
tidy never fails the install. `plan` only looks, so what would go can be listed without changing anything.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Never removed from .venv: uv's seed packages (`uv venv --seed`) and uv itself.
SEED = ("pip", "setuptools", "wheel", "uv")
# The installer installs every package with uv, which marks it so (the INSTALLER file in its .dist-info).
OURS = "uv"
# The packages the installer put in .venv, with everything they require (normalised names, a JSON list).
RECORD = Path(".tools") / "installed-packages.json"
# Paths a tidy could not remove, relative to NitroLearn's folder (a JSON list): tried again every tidy.
PENDING = Path(".tools") / "tidy-pending.json"
# NitroLearn's own package (its first name too), whose requirements tell an older install's packages.
ANCHORS = ("nitrolearn", "gigalearn-nitro")
# What older installers installed beside NitroLearn (name -> the extras asked for), Linux adding the
# dashboard's Qt window.
OLD_ROOTS: dict[str, tuple[str, ...]] = {"torch": (), "maturin": ()}
OLD_LINUX_ROOTS: dict[str, tuple[str, ...]] = {"pywebview": ("qt",)}
# The only folders in NitroLearn's folder a path removed here may be in.
OWNED = (".tools", ".cache", "target")
# uv's names for the Pythons it installs (and for its links to the newest of a minor version).
PYTHON_NAME = re.compile(r"^(?:cpython|pypy|graalpy|pyodide)-\d+\.\d+", re.IGNORECASE)
# uv's name for one install (implementation-version-system-machine-libc): known even with its Python gone.
PYTHON_INSTALL = re.compile(
    r"^(?:cpython|pypy|graalpy|pyodide)-\d+\.\d+\.\d+[^-]*"
    r"-(?:linux|windows|macos|darwin)-[a-z0-9_]+-[a-z0-9_]+$",
    re.IGNORECASE,
)
TORCH_CACHES = ("torch", "torchinductor", "triton")  # in .cache: TORCH_HOME and its compilers' caches
SHOWN = 4  # packages named in the summary; the rest are counted


def normal(name: str) -> str:
    """A package's name as Python packaging compares names (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------- environment markers ----------

_TOKEN = re.compile(
    r"\s*(?:(?P<text>'[^']*'|\"[^\"]*\")|(?P<op>===|==|!=|<=|>=|~=|<|>|\(|\))|(?P<word>[A-Za-z_][\w.]*))"
)
_VERSIONS = frozenset({"python_version", "python_full_version", "implementation_version"})


def environment() -> dict[str, str]:
    """This Python's values for the names an environment marker can use (PEP 508)."""
    info = sys.implementation.version
    version = f"{info.major}.{info.minor}.{info.micro}"
    if info.releaselevel != "final":
        version += info.releaselevel[0] + str(info.serial)
    return {
        "os_name": os.name,
        "sys_platform": sys.platform,
        "platform_machine": platform.machine(),
        "platform_python_implementation": platform.python_implementation(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_version": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "python_full_version": platform.python_version(),
        "implementation_name": sys.implementation.name,
        "implementation_version": version,
    }


def _tokens(marker: str) -> list[tuple[str, str]]:
    found, at = [], 0
    while at < len(marker.rstrip()):
        match = _TOKEN.match(marker, at)
        if match is None or match.end() == at:
            raise ValueError(marker)
        kind = match.lastgroup or ""
        found.append((kind, match[kind]))
        at = match.end()
    return found


def _version(text: str) -> tuple[int, ...]:
    numbers = re.match(r"^\d+(?:\.\d+)*", text.strip())
    if numbers is None:
        raise ValueError(text)
    return tuple(int(part) for part in numbers[0].split("."))


_ORDER: dict[str, Callable[[tuple[int, ...], tuple[int, ...]], bool]] = {
    "==": lambda one, two: one == two,
    "!=": lambda one, two: one != two,
    "<": lambda one, two: one < two,
    "<=": lambda one, two: one <= two,
    ">": lambda one, two: one > two,
    ">=": lambda one, two: one >= two,
}
_COMPARISONS = frozenset({*_ORDER, "~=", "===", "in", "not in"})


def _compare(left: str, op: str, right: str, names: tuple[str, str]) -> bool:
    if op in ("in", "not in"):
        return (left in right) == (op == "in")
    if "extra" in names:
        left, right = normal(left), normal(right)
    if op == "===":
        return left == right
    if op in ("==", "!=") and right.endswith(".*"):
        prefix = _version(right[:-2])
        return (_version(left)[: len(prefix)] == prefix) == (op == "==")
    if op in ("==", "!=") and not _VERSIONS.intersection(names):
        return (left == right) == (op == "==")
    one, given = _version(left), _version(right)
    width = max(len(one), len(given))
    one, two = one + (0,) * (width - len(one)), given + (0,) * (width - len(given))
    if op == "~=":
        return one >= two and one[: len(given) - 1] == two[: len(given) - 1]
    return _ORDER[op](one, two)


class _Marker:
    """PEP 508's marker grammar: or-ed and-ed comparisons, with parentheses."""

    def __init__(self, tokens: list[tuple[str, str]], env: Mapping[str, str]) -> None:
        self.tokens, self.env, self.at = tokens, env, 0

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.at] if self.at < len(self.tokens) else ("", "")

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if not token[0]:
            raise ValueError("the marker ends early")
        self.at += 1
        return token

    def either(self) -> bool:
        value = self.both()
        while self.peek() == ("word", "or"):
            self.take()
            value = self.both() or value
        return value

    def both(self) -> bool:
        value = self.one()
        while self.peek() == ("word", "and"):
            self.take()
            value = self.one() and value
        return value

    def one(self) -> bool:
        if self.peek() == ("op", "("):
            self.take()
            value = self.either()
            if self.take() != ("op", ")"):
                raise ValueError("a parenthesis is not closed")
            return value
        left, left_name = self.value()
        kind, op = self.take()
        if (kind, op) == ("word", "not") and self.take() == ("word", "in"):
            op = "not in"
        if op not in _COMPARISONS:
            raise ValueError(f"not a comparison: {op}")
        right, right_name = self.value()
        return _compare(left, op, right, (left_name, right_name))

    def value(self) -> tuple[str, str]:
        kind, text = self.take()
        if kind == "text":
            return text[1:-1], ""
        if kind == "word" and text in self.env:
            return self.env[text], text
        raise ValueError(f"unknown marker name {text}")


def marker_holds(marker: str, env: Mapping[str, str]) -> bool:
    """Whether an environment marker holds in `env`. One that cannot be read counts as holding: keeping a
    package is always safe, removing one that is needed is not."""
    try:
        reader = _Marker(_tokens(marker), env)
        value = reader.either()
        if reader.peek()[0]:
            raise ValueError("more after the marker")
        return value
    except (ValueError, IndexError):
        return True


def requirement(line: str) -> tuple[str, frozenset[str], str]:
    """A Requires-Dist line as (name, extras, marker)."""
    head, _, marker = line.partition(";")
    found = re.match(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[([^\]]*)\])?", head)
    if found is None:
        raise ValueError(line)
    extras = frozenset(normal(extra.strip()) for extra in (found[2] or "").split(",") if extra.strip())
    return normal(found[1]), extras, marker.strip()


def applies(marker: str, env: Mapping[str, str], extras: Iterable[str]) -> bool:
    """Whether a requirement with `marker` is needed by a package installed with `extras`."""
    return not marker or any(marker_holds(marker, {**env, "extra": extra}) for extra in ("", *extras))


# ---------- the packages in .venv ----------


@dataclass(frozen=True)
class Dist:
    name: str  # as the package spells it
    version: str
    requires: tuple[str, ...]
    installer: str  # what installed it (its INSTALLER file), lower case; "" when it does not say
    extras: tuple[str, ...] = ()  # the extras it offers (Provides-Extra), normalised


def installed(places: Sequence[Path]) -> dict[str, Dist]:
    """The packages installed in `places` (site-packages), by their normalised names."""
    found: dict[str, Dist] = {}
    for dist in importlib.metadata.distributions(path=[str(place) for place in places]):
        name = dist.metadata["Name"]
        if not name:
            continue
        marked = (dist.read_text("INSTALLER") or "").strip().lower()
        offers = tuple(normal(extra) for extra in dist.metadata.get_all("Provides-Extra") or ())
        found.setdefault(normal(name), Dist(name, dist.version, tuple(dist.requires or ()), marked, offers))
    return found


def needed(roots: Mapping[str, Iterable[str]], dists: Mapping[str, Dist], env: Mapping[str, str]) -> set[str]:
    """Every installed package `roots` (name -> extras) need, they included, following what each requires."""
    wanted: dict[str, set[str]] = {}
    waiting = [(normal(name), frozenset(normal(e) for e in extras)) for name, extras in roots.items()]
    while waiting:
        name, extras = waiting.pop()
        if name in wanted and extras <= wanted[name]:
            continue
        wanted.setdefault(name, set()).update(extras)
        dist = dists.get(name)
        for line in dist.requires if dist is not None else ():
            try:
                child, child_extras, marker = requirement(line)
            except ValueError:
                continue
            if applies(marker, env, wanted[name]):
                waiting.append((child, child_extras))
    return {name for name in wanted if name in dists}


@dataclass(frozen=True)
class Packages:
    remove: tuple[str, ...]  # names as the packages spell them
    by_hand: tuple[str, ...]  # kept: not the installer's (not in its record, or not installed by uv)
    why_none: str = ""  # why nothing is removed, when the packages this version needs are not all there
    record: tuple[str, ...] = ()  # the installer's packages once these are removed (normalised names)


def unneeded(
    roots: Mapping[str, Iterable[str]],
    dists: Mapping[str, Dist],
    env: Mapping[str, str],
    anchor: str,
    ours: Iterable[str],
) -> Packages:
    """The installer's packages in `dists` nothing in `roots` needs. `ours`: the installer's record of the
    packages it put there; what `roots` need now joins it. Only those can go, and of them only ones uv
    installed: every other package is the user's and stays, with what it needs, as do uv's seed packages.
    Nothing goes when `anchor` (NitroLearn itself) is not installed: then what this version needs cannot be
    told."""
    if normal(anchor) not in dists:
        return Packages((), (), f"{anchor} is not installed, so what it needs cannot be told")
    record = {normal(name) for name in ours} | needed(roots, dists, env)
    theirs = sorted(
        name
        for name, dist in dists.items()
        if name not in SEED and (name not in record or dist.installer != OURS)
    )
    keep = needed(dict(roots) | {name: () for name in (*SEED, *theirs)}, dists, env)
    remove = sorted(dists[name].name for name in dists if name in record and name not in keep)
    gone = {normal(name) for name in remove}
    return Packages(
        tuple(remove),
        tuple(dists[name].name for name in theirs),
        record=tuple(sorted(record - gone)),
    )


def first_record(dists: Mapping[str, Dist], env: Mapping[str, str], pinned: Iterable[str] = ()) -> set[str]:
    """The installer's record for a folder an older installer set up (it kept none): NitroLearn as
    installed then, with each extra it offers, what older installers put beside it (OLD_ROOTS) and the
    packages `pinned` (the release copy's list), with everything those require. Empty without NitroLearn."""
    roots: dict[str, Iterable[str]] = {name: dists[name].extras for name in ANCHORS if name in dists}
    if not roots:
        return set()
    roots |= OLD_ROOTS
    if env.get("sys_platform", "").startswith("linux"):
        roots |= OLD_LINUX_ROOTS
    for name in pinned:
        roots.setdefault(normal(name), ())
    return needed(roots, dists, env)


def read_names(path: Path) -> set[str] | None:
    """A JSON list of names (the record, the pending paths), or None when there is none to read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    return {str(item) for item in data if isinstance(item, str) and item}


def write_names(path: Path, names: Iterable[str]) -> None:
    """Writes `names` as read_names reads them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(names)), indent=0) + "\n", encoding="utf-8")


def seed_record(
    root: Path, places: Sequence[Path], env: Mapping[str, str] | None = None, pinned: Iterable[str] = ()
) -> bool:
    """Before an install: the installer's record for a folder that has none yet (first_record), so the
    older version's packages can go afterwards and nothing else. True when it was written."""
    path = root / RECORD
    if not places or path.exists():
        return False
    write_names(path, first_record(installed(places), env or environment(), pinned))
    return True


# ---------- paths ----------


def is_link(path: Path) -> bool:
    """A symbolic link, or on Windows any reparse point (a junction too): never followed here."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    return stat.S_ISLNK(info.st_mode) or bool(reparse)


def _top(path: str, folder: str) -> str | None:
    """The name of the entry of `folder` that `path` is in (or is), lower case; None when outside it."""
    try:
        rel = os.path.relpath(path, folder)
    except ValueError:  # another drive
        return None
    first = Path(rel).parts[0] if rel not in ("", ".") else ""
    return None if not first or first == ".." else first.lower()


def venv_home(prefix: Path) -> Path | None:
    """The base Python's folder a virtual environment names in its pyvenv.cfg (`home`), or None."""
    try:
        text = (prefix / "pyvenv.cfg").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    found = re.search(r"^home\s*=\s*(.+?)\s*$", text, re.MULTILINE)
    return Path(found[1]) if found else None


def running_pythons(folder: Path, running: Iterable[Path]) -> set[str]:
    """The entries of `folder` (.tools/python) the running Python is in, lower case (as named and as it
    resolves); empty when it is not in `folder`."""
    real = os.path.realpath(folder)
    keep: set[str] = set()
    for path in running:
        for top in (_top(str(path), str(folder)), _top(os.path.realpath(path), real)):
            if top:
                keep.add(top)
    return keep


def old_pythons(folder: Path, running: Iterable[Path]) -> list[Path]:
    """The Pythons uv installed in `folder` (.tools/python) other than the one in `running` (.venv's base
    Python, as pyvenv.cfg names it and as it resolves): each install, and each of uv's version links that
    does not lead to the kept one. Only entries named as uv names them; its own files (.lock, .temp...)
    stay. An install is known by its Python in it, or by uv's full name for one (a half-removed one has no
    Python left). Nothing when the running Python is not in `folder`: then which one is in use cannot be
    told."""
    if not folder.is_dir() or is_link(folder):
        return []
    real = os.path.realpath(folder)
    keep = running_pythons(folder, running)
    if not keep:
        return []
    old = []
    for entry in sorted(folder.iterdir()):
        name = entry.name.lower()
        if not PYTHON_NAME.match(name) or name in keep:
            continue
        if is_link(entry):
            if _top(os.path.realpath(entry), real) in keep:
                continue  # uv's link to the kept install
        elif not entry.is_dir() or not (
            PYTHON_INSTALL.match(name) or (entry / "python.exe").is_file() or (entry / "bin").is_dir()
        ):
            continue
        old.append(entry)
    return old


def _rustc_lines(target: Path) -> list[str]:
    """What rustc told cargo when it last built in `target` (its .rustc_info.json), a line each; empty when
    there is no such record."""
    try:
        info = json.loads((target / ".rustc_info.json").read_text(encoding="utf-8"))
        return [line.strip() for out in info["outputs"].values() for line in out["stdout"].splitlines()]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []


def rust_release(target: Path) -> set[str]:
    """The Rust versions cargo last used to build in `target` (its .rustc_info.json); empty when unknown."""
    return {line.removeprefix("release: ") for line in _rustc_lines(target) if line.startswith("release: ")}


def built_by_own_rust(root: Path) -> bool:
    """Whether cargo last built in `root`'s target/ with the folder's own Rust: the Rust it ran says its
    home (sysroot) is in .tools/rustup. A target/ built with any other Rust (the user's) is never touched."""
    rustup = root / ".tools" / "rustup"
    for line in _rustc_lines(root / "target"):
        if not os.path.isabs(line):
            continue
        for path, place in ((line, str(rustup)), (os.path.realpath(line), os.path.realpath(rustup))):
            if _top(path, place) is not None:
                return True
    return False


def holds_something(path: Path) -> bool:
    if is_link(path):
        return True
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def stale_target(root: Path, rust: str | None) -> bool:
    """Whether target/ is the folder's own Rust's build of another Rust version than `rust` (the pinned
    one; None when the folder has no Rust any more)."""
    built = rust_release(root / "target")  # cargo's own record: target/ is a cargo build folder
    return bool(built) and built_by_own_rust(root) and (rust is None or rust not in built)


def stale_paths(root: Path, *, torch_gone: bool, torch_changed: bool, rust: str | None) -> list[Path]:
    """The installer's leftovers in `root` that this install no longer uses (see the module's notes).
    `rust`: the pinned Rust version of the folder's own Rust, whose builds in target/ are kept; None when
    the folder has no Rust any more."""
    tools, cache = root / ".tools", root / ".cache"
    found = [tools / "vendor.new", cache / "pip"]
    if torch_gone:
        found.append(tools / "torch-choice.txt")
    if torch_gone or torch_changed:
        found += [cache / name for name in TORCH_CACHES]
    if stale_target(root, rust):
        found.append(root / "target")
    return [path for path in found if holds_something(path)]


def pending_paths(root: Path, *, pythons: set[str], torch_installed: bool, rust: str | None) -> list[Path]:
    """What earlier tidies could not remove (PENDING) that is there still and still unused: never the
    Python .venv runs on (`pythons`: its entries in .tools/python; none is retried when that is not
    known), a target/ built since with the pinned Rust or the user's own, or the remembered PyTorch build
    while PyTorch is installed."""
    found = []
    for rel in sorted(read_names(root / PENDING) or ()):
        parts = Path(rel).parts
        if Path(rel).is_absolute() or refusal(root, root / rel) or not os.path.lexists(root / rel):
            continue
        if parts[:2] == (".tools", "python") and (
            len(parts) < 3 or not pythons or parts[2].lower() in pythons
        ):
            continue
        if parts[0] == "target" and rust_release(root / "target") and not stale_target(root, rust):
            continue
        if parts == (".tools", "torch-choice.txt") and torch_installed:
            continue
        found.append(root / rel)
    return found


def refusal(root: Path, path: Path) -> str:
    """Why `path` may never be removed here, or "" when it may: it is inside one of the folders the
    installer owns in `root` (OWNED) and no folder on the way to it is a link."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return f"{path} is outside NitroLearn's folder"
    whole = len(rel.parts) == 1 and rel.parts[0] != "target"  # .tools or .cache itself
    if not rel.parts or rel.parts[0] not in OWNED or ".." in rel.parts or whole:
        return f"{rel.as_posix()} is not one of the installer's own"
    on_the_way = root
    for part in rel.parts[:-1]:
        on_the_way = on_the_way / part
        if is_link(on_the_way):
            return f"{on_the_way.relative_to(root).as_posix()} is a link, which is never followed"
    return ""


def remove_inside(root: Path, path: Path) -> str:
    """Removes `path`, a file, a folder or a link (the link itself), when `refusal` allows it. "" when it is
    gone, else why not."""
    if problem := refusal(root, path):
        return problem
    rel = path.relative_to(root)
    try:
        if is_link(path):
            try:
                path.unlink()
            except OSError:
                os.rmdir(path)  # Windows: a link to a folder (or a junction) goes as a folder does
        elif path.is_dir():
            stuck: list[str] = []
            shutil.rmtree(path, onerror=lambda function, name, info: stuck.append(name))
            if stuck:
                return f"{rel.as_posix()}: {len(stuck)} of its files could not be removed (in use?)"
        elif path.exists():
            path.unlink()
    except OSError as err:
        return f"{rel.as_posix()}: {err.strerror or err}"
    return ""


# ---------- the whole tidy ----------


@dataclass(frozen=True)
class Plan:
    packages: Packages
    paths: tuple[Path, ...]
    cached: tuple[str, ...]  # packages whose downloads uv's cache may drop: removed or replaced since
    whole_cache: bool = False  # an older Python goes: every download in uv's cache was for it or is reused


@dataclass
class Tidied:
    packages: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)  # relative to NitroLearn's folder
    problems: list[str] = field(default_factory=list)


def plan(
    root: Path,
    places: Sequence[Path],
    roots: Mapping[str, Iterable[str]],
    before: Mapping[str, str],
    *,
    anchor: str = "nitrolearn",
    prefix: Path | None = None,
    base: Path | None = None,
    rust: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Plan:
    """What a tidy of `root` removes now, changing nothing. `places`: .venv's site-packages; `roots`: the
    packages this version installs (name -> extras); `before`: each package's version before this install
    (name -> version); `prefix`, `base`: .venv and the Python it runs on (sys.prefix, sys.base_prefix);
    `rust`: the pinned Rust version of the folder's own Rust, None when the folder has none. The installer's
    record (RECORD) names the packages that may go; none without one."""
    dists = installed(places)
    ours = read_names(root / RECORD) or set()
    packages = unneeded(roots, dists, env or environment(), anchor, ours)
    after = {name: dist.version for name, dist in dists.items()}
    for gone in packages.remove:
        after.pop(normal(gone), None)
    torch = normal("torch")
    torch_gone = torch in before and torch not in after
    torch_changed = torch in before and torch in after and before[torch] != after[torch]
    paths = stale_paths(root, torch_gone=torch_gone, torch_changed=torch_changed, rust=rust)
    prefix = prefix or Path(sys.prefix)
    running = [Path(base or sys.base_prefix), *filter(None, [venv_home(prefix)])]
    folder = root / ".tools" / "python"
    pythons = old_pythons(folder, running)
    in_use = running_pythons(folder, running) if folder.is_dir() and not is_link(folder) else set()
    pending = pending_paths(root, pythons=in_use, torch_installed=torch in after, rust=rust)
    every = list(dict.fromkeys([*paths, *pythons, *pending]))
    cached = sorted(name for name, version in before.items() if after.get(name) != version)
    return Plan(packages, tuple(every), tuple(cached), whole_cache=bool(pythons))


def carry_out(
    found: Plan,
    root: Path,
    uv: Callable[[list[str]], int],
    places: Sequence[Path],
    python: str = sys.executable,
) -> Tidied:
    """Removes what `found` lists: the packages (`uv pip uninstall`), the paths (remove_inside), then uv's
    cached downloads of what was removed or replaced and its unused entries. `uv` runs uv with the given
    arguments and returns its exit code. Whatever cannot be removed is listed in `problems`; a package of
    those stays in the installer's record and a path goes in PENDING, so every later tidy tries again."""
    done = Tidied()
    names = list(found.packages.remove)
    if names:
        code = uv(["pip", "uninstall", "--python", python, *names])
        left = {normal(name) for name in installed(places)}
        done.packages = [name for name in names if normal(name) not in left]
        stuck = [name for name in names if normal(name) in left]
        if stuck:
            done.problems.append(f"packages still in .venv (uv ended with {code}): {', '.join(stuck)}")
    if not found.packages.why_none:
        left = {normal(name) for name in installed(places)}
        try:
            write_names(
                root / RECORD,
                [name for name in (*found.packages.record, *map(normal, names)) if name in left],
            )
        except OSError as err:
            done.problems.append(
                f"the record of the installer's packages could not be written ({err.strerror or err})"
            )
    pending = []
    for path in found.paths:
        problem = remove_inside(root, path)
        if problem:
            done.problems.append(problem)
            if not refusal(root, path) and os.path.lexists(path):
                pending.append(path.relative_to(root).as_posix())
        else:
            done.paths.append(path.relative_to(root).as_posix())
    with contextlib.suppress(OSError):
        if pending:
            write_names(root / PENDING, pending)
        else:
            (root / PENDING).unlink(missing_ok=True)
    if (root / ".cache" / "uv").is_dir():  # UV_CACHE_DIR (folder_env)
        dropped = [] if found.whole_cache else list(found.cached)
        if (found.whole_cache or dropped) and uv(["cache", "clean", *dropped]) != 0:
            done.problems.append("uv's cache could not drop the old downloads")
        if uv(["cache", "prune"]) != 0:
            done.problems.append("uv's cache could not be pruned")
    return done


def listed(names: Sequence[str], shown: int = SHOWN) -> str:
    if len(names) <= shown + 1:
        return ", ".join(names)
    return f"{', '.join(names[:shown])} and {len(names) - shown} more"


def summary(done: Tidied, freed_gb: float | None) -> str:
    """What a tidy removed, in one line."""
    parts = []
    if done.packages:
        count = len(done.packages)
        parts.append(f"{count} package{'s' if count != 1 else ''} ({listed(done.packages)})")
    parts += done.paths
    freed = f"; about {freed_gb:.1f} GB freed" if freed_gb is not None and freed_gb >= 0.05 else ""
    if not parts:
        return "Nothing left over from an older version to remove" + freed + "."
    return f"Removed what this version no longer uses: {listed(parts, 6)}{freed}."


def describe(found: Plan, root: Path) -> list[str]:
    """What `found` would remove, a line each (for a look before anything changes)."""
    lines = [f"package {name}" for name in found.packages.remove]
    lines += [path.relative_to(root).as_posix() for path in found.paths]
    if found.whole_cache:
        lines.append("uv's cached downloads (an older Python goes)")
    else:
        lines += [f"uv's cached downloads of {name}" for name in found.cached]
    if found.packages.why_none:
        lines.append(f"(no packages: {found.packages.why_none})")
    if found.packages.by_hand:
        lines.append(f"(kept, installed by hand: {', '.join(found.packages.by_hand)})")
    return lines
