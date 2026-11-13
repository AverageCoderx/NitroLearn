"""The bot's game starts (`[[state_setters]]`) for the dashboard's Starts tab: what each kind is, the starts a
settings file lists as the editor shows them, and the file with edited ones written back (comments kept)."""

from __future__ import annotations

import dataclasses
import inspect
import tomllib
from typing import Any, get_args, get_type_hints

from nitro.config.state_setters import DEFAULT_STATE_SETTERS, StateSetter
from nitro.dashboard import describe, rewards

KINDS: dict[str, Any] = {cls.name: cls for cls in get_args(StateSetter)}
_SHARED = ("name", "weight")


def catalog() -> list[dict[str, Any]]:
    """Every kind of start: its name, title, what it does and its settings (with their defaults)."""
    out = []
    for name, cls in KINDS.items():
        hints = get_type_hints(cls)
        out.append(
            {
                "name": name,
                "title": "Random" if name == "random" else describe.label(name),
                "about": inspect.cleandoc(cls.__doc__ or ""),
                "params": [
                    describe.field_item(f.name, f, hints[f.name], f.default)
                    for f in dataclasses.fields(cls)
                    if f.name not in _SHARED
                ],
            }
        )
    return out


def read(text: str) -> tuple[list[dict[str, Any]], bool]:
    """The starts `text` (a settings file) lists, as editor views, and whether it lists any: without
    `[[state_setters]]` every game starts with a kickoff, shown here as it would be written."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return [], False
    listed = data.get("state_setters")
    if isinstance(listed, list):
        return [view(item if isinstance(item, dict) else {}) for item in listed], True
    return [view({"name": s.name, "weight": s.weight}) for s in DEFAULT_STATE_SETTERS], False


def view(data: dict[str, Any]) -> dict[str, Any]:
    """One start as the editor shows it: every setting present, defaults filled in."""
    name = str(data.get("name", ""))
    cls = KINDS.get(name)
    params = {}
    if cls is not None:
        for f in dataclasses.fields(cls):
            if f.name not in _SHARED:
                value = data.get(f.name, f.default)
                params[f.name] = [str(v) for v in value] if isinstance(value, list | tuple) else value
    return {"name": name, "known": cls is not None, "weight": data.get("weight", 1.0), "params": params}


def data_of(item: dict[str, Any]) -> dict[str, Any]:
    """An editor view as its `[[state_setters]]` block writes it (settings at their defaults are left out
    when the block is written)."""
    out: dict[str, Any] = {"name": item["name"], "weight": rewards._number(item.get("weight", 1.0))}
    for key, value in (item.get("params") or {}).items():
        if isinstance(value, str) and _listed(item["name"], key):  # a list typed as "a, b"
            value = [part.strip() for part in value.split(",") if part.strip()]
        if value is None or value == "":
            continue
        out[key] = value if isinstance(value, bool | list) else rewards._number(value)
    return out


def _listed(name: str, key: str) -> bool:
    cls = KINDS.get(name)
    fields = {f.name: f for f in dataclasses.fields(cls)} if cls else {}
    return key in fields and isinstance(fields[key].default, tuple)


def written(text: str, entries: list[dict[str, Any]]) -> str:
    """`text` with its starts replaced by `entries` (each {"from": the block it was or None, "view": ...})."""
    return rewards.written(text, entries, key="state_setters", data=data_of, kinds=KINDS)


def shares(views: list[dict[str, Any]]) -> list[float | None]:
    """Each start's share of games (its weight over the sum), None where a weight is not a number."""
    weights = [rewards._number(v.get("weight")) for v in views]
    numbers = [
        w if isinstance(w, int | float) and not isinstance(w, bool) and w >= 0 else None for w in weights
    ]
    total = sum(w for w in numbers if w is not None)
    return [None if w is None or total <= 0 else w / total for w in numbers]


def changes(before: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[str]:
    """The waiting changes to the starts in plain words, one line each."""
    lines = []
    kept = set()
    for entry in entries:
        item, index = entry["view"], entry.get("from")
        if not isinstance(index, int) or not 0 <= index < len(before):
            lines.append(f"{item['name']}: added, weight {rewards._show(item.get('weight'))}")
            continue
        kept.add(index)
        old = before[index]
        name = item["name"]
        if rewards._number(old.get("weight")) != rewards._number(item.get("weight")):
            lines.append(
                f"{name}: weight {rewards._show(old.get('weight'))} -> {rewards._show(item.get('weight'))}"
            )
        for key, value in (item.get("params") or {}).items():
            was = (old.get("params") or {}).get(key)
            if _same(was) != _same(value):
                lines.append(f"{name}: {describe.label(key).lower()} {_shown(was)} -> {_shown(value)}")
    lines.extend(f"{old['name']}: removed" for i, old in enumerate(before) if i not in kept)
    return lines


def _same(value: Any) -> Any:
    if isinstance(value, str) and "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value) if isinstance(value, tuple) else rewards._number(value)


def _shown(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ", ".join(str(v) for v in value) or "(none)"
    return rewards._show(value)
