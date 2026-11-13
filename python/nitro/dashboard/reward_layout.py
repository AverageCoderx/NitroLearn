"""How the Rewards tab lays out a bot's rewards: named folders, each folder's order, the order of the rewards
in no folder, and which folders are closed. A view only: kept in the bot's folder, never in its settings file,
so it changes nothing about training and needs no Apply.

A reward is named by its key: its name, with `#2`, `#3`, ... for a second or third reward of that name (the
same keys its notes use)."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nitro.dashboard import edits

FILE = "reward_layout.json"  # in the bot's folder
MAX_FOLDERS = 50
MAX_NAME = 60
_ID = re.compile(r"[A-Za-z0-9_-]{1,24}")


def keys(names: Iterable[str]) -> list[str]:
    """The key of each reward, in order, from their names: `touch_ball`, `all_of`, `all_of#2`."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
    return out


def empty() -> dict[str, Any]:
    return {"folders": [], "loose": []}


def cleaned(raw: Any, known: Iterable[str], strict: bool = False) -> dict[str, Any]:
    """A layout as the page may use it: only the rewards in `known`, each in one place at most, folders with
    a name and an id. With `strict` (a layout sent by the page) a folder without a name or with the name of
    another is refused (ValueError, in plain words); else such a folder is renamed."""
    allowed = set(known)
    placed: set[str] = set()

    def rewards_of(value: Any) -> list[str]:
        out = []
        for key in value if isinstance(value, list) else []:
            if isinstance(key, str) and key in allowed and key not in placed:
                placed.add(key)
                out.append(key)
        return out

    folders: list[dict[str, Any]] = []
    names: set[str] = set()
    ids: set[str] = set()
    listed = raw.get("folders") if isinstance(raw, dict) else None
    for number, item in enumerate(listed if isinstance(listed, list) else [], 1):
        if not isinstance(item, dict):
            continue
        if len(folders) == MAX_FOLDERS:
            if strict:
                raise ValueError(f"a bot can have at most {MAX_FOLDERS} reward folders")
            break
        name = " ".join(str(item.get("name") or "").split())[:MAX_NAME]
        if strict and not name:
            raise ValueError("a folder needs a name")
        if strict and name.casefold() in names:
            raise ValueError(f"there is already a folder called '{name}'")
        name = _free_name(name or f"Folder {number}", names)
        names.add(name.casefold())
        ident = item.get("id")
        if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in ids:
            ident = _free_id(ids)
        ids.add(ident)
        folders.append(
            {
                "id": ident,
                "name": name,
                "collapsed": item.get("collapsed") is True,
                "rewards": rewards_of(item.get("rewards")),
            }
        )
    loose = rewards_of(raw.get("loose") if isinstance(raw, dict) else None)
    return {"folders": folders, "loose": loose}


def _free_name(name: str, taken: set[str]) -> str:
    out, n = name, 2
    while out.casefold() in taken:
        out, n = f"{name[: MAX_NAME - 4]} {n}", n + 1
    return out


def _free_id(taken: set[str]) -> str:
    n = 1
    while f"f{n}" in taken:
        n += 1
    return f"f{n}"


def _stored(folder: Path) -> Any:
    try:
        return json.loads((folder / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read(folder: Path, known: Iterable[str]) -> dict[str, Any]:
    """The layout kept in the bot's `folder`, without the rewards no longer in `known` (the bot's rewards in
    its settings file and in its waiting changes). A reward gone from both is dropped from the file too, so
    one of its name added later starts out of any folder."""
    stored = _stored(folder)
    if stored is None:
        return empty()
    layout = cleaned(stored, known)
    if layout != stored:
        with contextlib.suppress(OSError):  # shown cleaned all the same; the file waits for the next change
            _save(folder, layout)
    return layout


def write(folder: Path, raw: Any, known: Iterable[str]) -> dict[str, Any]:
    """Keeps a layout sent by the page in the bot's `folder` and returns it as kept."""
    layout = cleaned(raw, known, strict=True)
    _save(folder, layout)
    return layout


def _save(folder: Path, layout: dict[str, Any]) -> None:
    edits.save(folder / FILE, json.dumps(layout, indent=1, ensure_ascii=False))
