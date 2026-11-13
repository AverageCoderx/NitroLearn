"""Every setting described for the dashboard's editor, from the settings' own definitions (their help text,
bounds and choices): whatever a settings file can say, the editor shows, without a list to keep in step."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import tomllib
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from nitro.config.loader import Rule
from nitro.config.schema import NitroConfig
from nitro.dashboard import guide, layout
from nitro.runs import FIXED_SETTINGS, game_size_free

# Lists of their own kind, edited in their own tabs (rewards, starts).
SKIPPED = ("rewards", "state_setters")
_WHY_FIXED = "The bot's saved networks were made with this; start a new bot from this one to change it."
_WORDS = (
    ("lr", "learning rate"), ("kl", "KL"), ("ppo", "PPO"), ("gpu", "GPU"), ("fp16", "FP16"),
    ("kph", "(km/h)"), ("uu", "(units)"),
)  # fmt: skip
# Sections read only when a bot is made: changing them later changes nothing.
AT_START = {
    "start_from": "Read at the bot's first start.",
    "pretrain": "Read only when the bot is first made.",
}
# Settings whose own name says too little to someone new.
_LABELS = {
    "fuzz_uu": "Random shift (units)",  # a kickoff start's
    "start_from.enabled": "Transfer learning",
    "start_from.run": "Start from",
    "start_from.at": "Which save",
    "start_from.parts": "What to take",
    "pretrain.enabled": "Learn from replays first",
    "pretrain.replays": "Replay folders",
    "pretrain.team_sizes": "Game sizes",
    "pretrain.limit": "How many replays",
    "pretrain.passes": "Times each replay is learned from",
    "pretrain.batch_replays": "Replays a round",
    "pretrain.workers": "Replays converted at once",
    "pretrain.prev_action_dropout": "Share of decisions without the previous move",
    "pretrain.keep_converted": "Keep converted replays for later bots",
    "pretrain.skill_every": "Skill check every (rounds)",
    "pretrain.skill_points": "Points against each earlier version",
    "pretrain.save_every": "Save the copy so far every (replays)",
    "pretrain.value_warmup_steps": "Value warm-up steps (policy held still)",
    "replay_mix.enabled": "Keep copying replays while training",
    "replay_mix.share": "Share of learning steps that copy replays",
    "replay_mix.lr": "Copy steps' learning rate",
    "replay_mix.minibatch": "Decisions per copy step",
    "replay_mix.replays": "Converted replay folders",
}


def describe(
    config: NitroConfig, text: str, locked: bool, fixed: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """The sections of `config` with each setting's current value, as written in `text` (the settings
    file); with `locked` (the bot has saves), the settings its networks depend on cannot change (but the game
    size where a resume takes another), nor those in `fixed` (setting path -> why)."""
    lines = _written(text)
    with_saves = frozenset(FIXED_SETTINGS) - set(game_size_free(config))
    locks = with_saves if locked else frozenset()
    sections = []
    for f in dataclasses.fields(NitroConfig):
        if f.name in SKIPPED:
            continue
        value = getattr(config, f.name)
        if dataclasses.is_dataclass(value):
            sections.append(_section(f.name, f, value, lines, locks, fixed or {}, with_saves))
    return sections


def _section(
    path: str,
    f: dataclasses.Field[Any],
    value: Any,
    lines: set[str],
    locks: frozenset[str],
    fixed: dict[str, str],
    with_saves: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    hints = get_type_hints(type(value))
    items: list[dict[str, Any]] = []
    subsections: list[dict[str, Any]] = []
    for sub in dataclasses.fields(value):
        child = getattr(value, sub.name)
        child_path = f"{path}.{sub.name}"
        if dataclasses.is_dataclass(child) and not isinstance(child, type):
            subsections.append(_section(child_path, sub, child, lines, locks, fixed, with_saves))
            continue
        if child_path in layout.NOT_SHOWN:
            continue
        item = _item(child_path, sub, hints[sub.name], child, lines, locks, fixed.get(child_path, ""))
        items.append({**item, **_placed(child_path, with_saves)})
    rule = _rule(f)
    return {
        "path": path,
        "label": label(path.rsplit(".", 1)[-1]),
        "help": rule.help if rule else "",
        "note": AT_START.get(path, ""),
        "items": items,
        "sections": subsections,
    }


def _item(
    path: str,
    f: dataclasses.Field[Any],
    tp: Any,
    value: Any,
    lines: set[str],
    locks: frozenset[str],
    why: str,
) -> dict[str, Any]:
    why = why or (_WHY_FIXED if path in locks else "")
    return {
        **field_item(path, f, tp, value),
        "written": path in lines,
        "fixed": bool(why),
        "why_fixed": why,
    }


def _placed(path: str, with_saves: frozenset[str]) -> dict[str, Any]:
    """What the Settings tab adds to a setting: its group, its level, its plain title and its place in the
    list (`layout`), its written "Nerd explanation" if there is one, and when a change to it counts: "saves"
    (fixed once the bot has a save), "first_start" (read when the bot is made) or "next_start"."""
    place = layout.place(path)
    section = path.split(".", 1)[0]
    return {
        "group": place.group,
        "level": place.level,
        "title": place.title,
        "order": layout.order(path),
        "nerd": nerd(path),
        "applies": "saves" if path in with_saves else "first_start" if section in AT_START else "next_start",
    }


def nerd(path: str) -> str:
    """The written "Nerd explanation" of `path` (`guide.NERD`, keyed as `guide.MORE`), or "" when there is
    none: the page then builds one from the setting's own facts."""
    return guide.NERD.get(path, "")


def field_item(path: str, f: dataclasses.Field[Any], tp: Any, value: Any) -> dict[str, Any]:
    """One setting (a field of type `tp` holding `value`) as the editor shows it: label, help text, kind of
    input, choices and limits."""
    rule = _rule(f)
    kind, options, nullable = _kind(tp)
    default = f.default if f.default is not dataclasses.MISSING else None
    return {
        "path": path,
        "label": _LABELS.get(path) or label(path.rsplit(".", 1)[-1]),
        "help": rule.help if rule else "",
        "plain": layout.PLAIN.get(path, ""),
        "more": guide.more(path),
        "kind": kind,
        "options": options,
        "nullable": nullable,
        "minimum": rule.minimum if rule else None,
        "maximum": rule.maximum if rule else None,
        "value": plain(value),
        "default": plain(default),
    }


def _written(text: str) -> set[str]:
    """The dotted paths of the settings `text` writes itself (the rest take their defaults)."""
    found: set[str] = set()

    def walk(table: dict[str, Any], prefix: str) -> None:
        for key, value in table.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, path)
            else:
                found.add(path)

    with contextlib.suppress(tomllib.TOMLDecodeError):  # a broken file: nothing counts as written
        walk(tomllib.loads(text), "")
    return found


def _rule(f: dataclasses.Field[Any]) -> Rule | None:
    rule = f.metadata.get("rule")
    return rule if isinstance(rule, Rule) else None


def _kind(tp: Any) -> tuple[str, list[Any], bool]:
    """What the editor shows for a setting of type `tp`: its kind, the choices of a list of them, and
    whether it may be left unset."""
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        kind, options, _ = _kind(args[0]) if len(args) == 1 else ("other", [], False)
        return kind, options, True
    if origin is Literal:
        return "choice", list(get_args(tp)), False
    if origin is tuple:
        parts = get_args(tp)
        inner = parts[0] if parts else Any
        if inner in (int, float):
            return "numbers", [], False
        if inner in (str, Path) or get_origin(inner) is Literal:
            return "words", list(get_args(inner)) if get_origin(inner) is Literal else [], False
        return "other", [], False
    if tp is bool:
        return "bool", [], False
    if tp is int:
        return "int", [], False
    if tp is float:
        return "float", [], False
    if tp is Path:
        return "path", [], False
    if tp is str:
        return "text", [], False
    return "other", [], False


def plain(value: Any) -> Any:
    """`value` as JSON can carry it."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return value


def label(key: str) -> str:
    """A setting's name as words: `policy_lr` -> 'Policy learning rate'."""
    text = key.replace("_", " ")
    for short, long in _WORDS:
        text = " ".join(long if word == short else word for word in text.split(" "))
    return text[:1].upper() + text[1:]
