"""The dashboard's resources library: files a person adds, recognised by what they
hold, kept as a list (their paths, never copies) with notes, and offered where they fit: whole settings files
as starting points for new bots, reward sets in a bot's Rewards tab, Rust rewards compiled in by Build
(`nitro rebuild`); Rust observation builders and action parsers are recognised and kept for a later step.
Notes can also be written on the built-in rewards, observations and action sets."""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

from nitro.config.loader import ConfigError, load_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import edits, rewards

LIBRARY = "library.json"  # in the runs folder's .dashboard folder
SECTIONS = frozenset(f.name for f in dataclasses.fields(NitroConfig))
KIND_WORDS = {
    "settings": "Settings file",
    "rewards": "Reward set",
    "rust_reward": "Rust reward",
    "rust_observation": "Rust observation builder",
    "rust_actions": "Rust action parser",
}
# What a Rust or C++ file defines, by the kind of thing it implements.
_RUST = (
    ("rust_reward", re.compile(r"\bimpl\b[^{;]*\bReward\b\s+for\b")),
    ("rust_observation", re.compile(r"\bimpl\b[^{;]*\bObsBuilder\b\s+for\b")),
    ("rust_actions", re.compile(r"\bimpl\b[^{;]*\bActionParser\b\s+for\b")),
)
_CPP = re.compile(r":\s*public\s+(?:\w+::)*(Reward|ObsBuilder|ActionParser)\b")


def detect(path: Path) -> dict[str, Any]:
    """What the file at `path` is: its kind (a key of KIND_WORDS), a title, a line about what it holds, and a
    problem in plain words when NitroLearn cannot use it."""
    out: dict[str, Any] = {"kind": "", "title": path.stem.replace("_", " "), "about": "", "problem": ""}
    if not path.is_file():
        return {**out, "problem": "The file is no longer there."}
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return _toml(path, out)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".rs":
        for kind, pattern in _RUST:
            if pattern.search(text):
                return {
                    **out,
                    "kind": kind,
                    "about": (
                        "Your own reward, in Rust: Build on the Resources page compiles it into NitroLearn."
                        if kind == "rust_reward"
                        else f"Your own {KIND_WORDS[kind].lower()[5:]}, in Rust, kept for a later step that "
                        "compiles these too (they must also reach the exported RLBot bot)."
                    ),
                }
        return {
            **out,
            "problem": "A Rust file, but it implements no Reward, ObsBuilder or ActionParser that "
            "NitroLearn could use.",
        }
    if suffix in (".cpp", ".h", ".hpp", ".cc"):
        found = _CPP.search(text)
        what = f"a C++ {found.group(1)}" if found else "C++"
        return {
            **out,
            "problem": f"This is {what}: C++ files cannot be loaded as they are. Its maths can "
            "be written again as a Rust file, which this library takes.",
        }
    if suffix == ".py":
        return {
            **out,
            "problem": "Rewards and observations in Python would slow training a great deal; NitroLearn "
            "takes them as Rust files.",
        }
    return {
        **out,
        "problem": "NitroLearn takes .toml settings files and reward sets, and .rs Rust files.",
    }


def _toml(path: Path, out: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        return {**out, "problem": f"It is not a valid settings file: {err}."}
    keys = set(data)
    unknown = sorted(keys - SECTIONS)
    if not keys:
        return {**out, "problem": "The file is empty (or only comments)."}
    if unknown:
        return {**out, "problem": f"Not a NitroLearn settings file: it has {', '.join(unknown)}."}
    problems = edits.issues(text, path, path.parent)
    problem = "; ".join(issue.describe() for issue in problems[:3])
    if keys == {"rewards"}:
        views, _ = rewards.read(text)
        on = [v["name"] for v in views if v.get("enabled", True) is not False]
        count = f"{len(views)} reward" + ("" if len(views) == 1 else "s")
        about = f"{count}: {', '.join(on[:6])}{' …' if len(on) > 6 else ''}"
        return {**out, "kind": "rewards", "about": about, "problem": problem}
    if problem:
        return {**out, "kind": "settings", "problem": problem}
    return {**out, "kind": "settings", "about": describe_settings(load_toml(path, NitroConfig))}


def _in_moves_group(path: Path) -> bool:
    """A reward set that comes with NitroLearn for a trick move: its header has a `# group: moves` line."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return False
    header = itertools.takewhile(lambda line: line.startswith("#"), lines)
    return any(line[1:].strip() == rewards.MOVES_GROUP for line in header)


def describe_settings(config: NitroConfig) -> str:
    """A settings file in a few words: game size, tick skip, observation, network."""
    size = config.env.team_size
    network = {"mlp": "plain network", "transformer": "transformer network"}.get(
        config.model.encoder, "attention network"
    )
    if config.model.residual:
        network = "residual " + network
    sizes = "x".join(str(n) for n in config.model.policy)
    observation = config.env.observation
    about = f"{size}v{size}, tick skip {config.env.tick_skip}, {observation} observation, {network} {sizes}"
    if config.ppo.signals_on():
        goals_only = [r.name for r in config.rewards if r.enabled] == ["goal"]
        about += ", goal-only reward" if goals_only else ""
        about += ", GCO learning signals"
    if config.ppo.optimizer == "muon":
        about += ", Muon"
    return about


class Library:
    """The list of added files and the notes on resources, kept in the runs folder. The reward sets that
    come with NitroLearn (`built_in`, configs/reward-sets) are listed first."""

    def __init__(self, runs: Path, built_in: Path | None = None) -> None:
        self.file = runs / ".dashboard" / LIBRARY
        self.built_in = built_in

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"files": [], "notes": {}}
        if not isinstance(data, dict):
            return {"files": [], "notes": {}}
        data.setdefault("files", [])
        data.setdefault("notes", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        edits.save(self.file, json.dumps(data, indent=1, ensure_ascii=False))

    def entries(self) -> list[dict[str, Any]]:
        """Every added file with what it is now (a file can change or go after it was added)."""
        data = self._read()
        notes = data["notes"]
        shipped = sorted(self.built_in.glob("*.toml")) if self.built_in and self.built_in.is_dir() else []
        return [
            {
                "id": "set-" + path.stem,
                "path": str(path),
                "added": 0,
                "built_in": True,
                "trick_move": _in_moves_group(path),
                **detect(path),
                "note": notes.get("file:set-" + path.stem, ""),
            }
            for path in shipped
        ] + [
            {**entry, **detect(Path(entry["path"])), "note": notes.get("file:" + entry["id"], "")}
            for entry in data["files"]
        ]

    def add(self, path: Path) -> dict[str, Any]:
        """Adds the file at `path` (once); refuses a file NitroLearn cannot use, saying why."""
        path = path.resolve()
        found = detect(path)
        if not found["kind"]:
            raise ValueError(f"{path.name}: {found['problem']}")
        data = self._read()
        for entry in data["files"]:
            if Path(entry["path"]) == path:
                return {**entry, **found}
        entry = {"id": uuid.uuid4().hex[:12], "path": str(path), "added": time.time()}
        data["files"].append(entry)
        self._write(data)
        return {**entry, **found}

    def remove(self, ident: str) -> None:
        """Takes a file off the list (the file itself stays where it is)."""
        if any(entry["id"] == ident and entry.get("built_in") for entry in self.entries()):
            raise ValueError("that reward set comes with NitroLearn, so it stays on the list")
        data = self._read()
        data["files"] = [entry for entry in data["files"] if entry["id"] != ident]
        data["notes"].pop("file:" + ident, None)
        self._write(data)

    def notes(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self._read()["notes"].items()}

    def set_note(self, key: str, text: str) -> dict[str, str]:
        data = self._read()
        if text.strip():
            data["notes"][key] = text
        else:
            data["notes"].pop(key, None)
        self._write(data)
        return self.notes()

    def path_of(self, ident: str, kind: str) -> Path:
        for entry in self.entries():
            if entry["id"] == ident and entry["kind"] == kind:
                return Path(entry["path"])
        raise ValueError(f"there is no {KIND_WORDS.get(kind, kind).lower()} {ident} in the library")


def settings_sources(project: Path, library: Library) -> list[dict[str, Any]]:
    """Every settings file a new bot can start from: the presets, then the files in my-settings, then the
    settings files added to the library. Each: an id for the window, a title, its game size and a summary."""
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add(ident: str, title: str, path: Path, group: str) -> None:
        if path.resolve() in seen:
            return
        seen.add(path.resolve())
        try:
            config = load_toml(path, NitroConfig)
        except ConfigError:
            return  # a broken file is listed with its problem on the Resources page, not offered here
        found.append(
            {
                "id": ident,
                "title": title,
                "group": group,
                "team_size": config.env.team_size,
                "about": describe_settings(config),
            }
        )

    for path in sorted((project / "configs" / "presets").glob("*.toml")):
        add(path.stem, preset_title(path.stem), path, "preset")
    for path in sorted((project / "my-settings").glob("*.toml")):
        add("file:" + str(path.resolve()), path.stem.replace("_", " "), path, "my-settings")
    for entry in library.entries():
        if entry["kind"] == "settings" and not entry["problem"]:
            add("file:" + str(Path(entry["path"]).resolve()), entry["title"], Path(entry["path"]), "library")
    return found


def preset_title(stem: str) -> str:
    """A preset as the page names it: its card's title (the wizard's) or its file name in words, as
    "1v1 starter preset" or "GCO bot preset"."""
    from nitro.dashboard.garage import PRESET_CARDS

    title = str(PRESET_CARDS.get(stem, {}).get("title") or stem.replace("_", " "))
    return f"{title[:1].upper()}{title[1:]} preset"


def save_reward_set(project: Path, name: str, entries: list[dict[str, Any]], bot: str) -> Path:
    """Writes `entries` (rewards as the editor holds them) as a reward set in my-settings/reward-sets."""
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$", name or ""):
        raise ValueError("a reward set's name uses letters, digits, spaces, '-', '_' and '.'")
    folder = project / "my-settings" / "reward-sets"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / (name.strip().replace(" ", "_") + ".toml")
    if file.exists():
        raise ValueError(f"there is already a reward set called {name}")
    header = f"# Reward set '{name}', saved from {bot} by the dashboard on {time.strftime('%Y-%m-%d')}.\n"
    fresh = [{"from": None, "view": entry["view"]} for entry in entries]
    text = rewards.written(header, fresh)
    edits.save(file, text)
    return file
