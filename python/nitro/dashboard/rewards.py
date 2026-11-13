"""A bot's rewards in the dashboard: each `[[rewards]]` block as the editor shows it, changes
written back into the settings file block by block (its comments kept), and each reward's share of all the
reward paid over the last minute of training, to spot a reward the bot farms."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import json
import math
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args, get_type_hints

import tomlkit
from tomlkit.items import AoT, Comment, Table, Whitespace

from nitro.config.rewards import DEFAULT_REWARDS, STATE_REWARDS, Reward, ZeroSum, custom_kinds
from nitro.config.schema import EnvConfig
from nitro.dashboard import bots, describe, reward_guide
from nitro.runs import RunFolder, numbered

# The fields every reward has; the rest are its own parameters.
_SHARED = ("name", "weight", "zero_sum", "enabled", "parts")
# The training's own farming warning (nitro.health): one reward over this share of all reward paid.
DOMINANT_SHARE = 0.9
_DEFAULT_TICK_SKIP: int = EnvConfig().tick_skip
# The built-in rewards by name (a reward of one's own has no fixed name: `custom_kinds`).
KINDS: dict[str, type[Any]] = {
    cls.__dataclass_fields__["name"].default: cls
    for cls in get_args(Reward)
    if cls.__dataclass_fields__["name"].default
}
_RUST_KINDS = {"f32": "float", "f64": "float", "bool": "bool", "String": "text"}
# The header line that puts a reward file or a reward set in the trick-move group.
MOVES_GROUP = "group: moves"


def catalog() -> list[dict[str, Any]]:
    """Every reward a settings file can name, with what it pays for and its own parameters."""
    out = []
    for name, cls in KINDS.items():
        hints = get_type_hints(cls)
        params = [
            describe.field_item(f.name, f, hints[f.name], f.default)
            for f in dataclasses.fields(cls)
            if f.name not in _SHARED
        ]
        out.append(
            {
                "name": name,
                "title": describe.label(name),
                "about": inspect.cleandoc(cls.__doc__ or ""),
                "every_step": name in STATE_REWARDS,
                "group": name == "all_of",
                "params": params,
                "more": reward_guide.MORE.get(name, ""),
                "nerd": _nerd(name),
            }
        )
    for kind in custom_kinds().values():
        out.append(
            {
                "name": kind.name,
                "title": describe.label(kind.name),
                "about": kind.about or "A reward of your own.",
                "every_step": kind.every_step,
                "group": False,
                "own": True,
                "file": kind.file,
                # The library's rewards come with NitroLearn; the rest are files of one's own.
                "source": _source(kind.file),
                "trick_move": _source(kind.file) == "library" and in_moves_group(kind.file),
                "family": Path(kind.file).stem.replace("_", " "),
                "more": reward_guide.MORE.get(kind.name, ""),
                "nerd": _nerd(kind.name),
                "params": [
                    {
                        "path": name,
                        "label": describe.label(name),
                        "help": doc,
                        "kind": _RUST_KINDS.get(rust_type, "int" if rust_type[:1] in "iu" else "other"),
                        "options": [],
                        "nullable": optional,
                        "minimum": 0 if rust_type.startswith("u") else None,
                        "maximum": None,
                        "value": None,
                        "default": json.loads(default) if default else None,
                    }
                    for name, rust_type, optional, doc, default in kind.params
                ],
            }
        )
    return out


def _nerd(name: str) -> str:
    """The written "Nerd explanation" of the reward `name` (`reward_guide.NERD`, keyed as its `MORE`), or ""
    when there is none: the page then builds one from the reward's own settings."""
    return reward_guide.NERD.get(name, "")


def _source(file: str) -> str:
    """Where a compiled-in reward comes from: "library" (custom/library, the rewards that come with
    NitroLearn), "example" (the example every build has) or "own" (a file of one's own)."""
    parts = [part.lower() for part in Path(file.replace("\\", "/")).parts]
    if parts[-2:-1] == ["library"] and parts[-3:-2] == ["custom"]:
        return "library"
    return "example" if parts[-1:] == ["example.rs"] and parts[-2:-1] == ["custom"] else "own"


@functools.cache
def in_moves_group(file: str) -> bool:
    """Whether a reward that comes with NitroLearn is one of the trick-move rewards, listed as their own
    group: its file's `//!` header has a `group: moves` line."""
    try:
        lines = Path(file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    header = itertools.takewhile(lambda line: line.startswith("//!"), lines)
    return any(line[3:].strip() == MOVES_GROUP for line in header)


def help_texts() -> dict[str, str]:
    """The help of the settings every reward shares, for the editor's tooltips."""
    texts = {}
    for prefix, cls in (("", KINDS["air"]), ("zero_sum.", ZeroSum)):
        for f in dataclasses.fields(cls):
            item = describe.field_item(f.name, f, Any, None)
            if item["help"]:
                texts[prefix + f.name] = item["help"]
    return texts


def read(text: str) -> tuple[list[dict[str, Any]], bool]:
    """The rewards `text` (a settings file) writes, as editor views, and whether it writes them: without any
    `[[rewards]]` the run uses the default rewards, shown here as they would be written."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return [], False
    listed = data.get("rewards")
    if isinstance(listed, list):
        return [view(item if isinstance(item, dict) else {}) for item in listed], True
    return [view(as_written(reward)) for reward in DEFAULT_REWARDS], False


def as_written(value: Any) -> Any:
    """`value` as a settings file states it: dataclasses as dicts without the fields left at their defaults
    (a reward's `name` kept)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            item = getattr(value, f.name)
            if f.name == "name" or f.default is dataclasses.MISSING or item != f.default:
                out[f.name] = as_written(item)
        return out
    if isinstance(value, tuple | list):
        return [as_written(item) for item in value]
    return value


def view(data: dict[str, Any]) -> dict[str, Any]:
    """One reward as the editor shows it: every setting present, defaults filled in."""
    name = str(data.get("name", ""))
    cls = KINDS.get(name)
    own = custom_kinds().get(name)
    params = {}
    if cls is not None:
        for f in dataclasses.fields(cls):
            if f.name not in _SHARED:
                params[f.name] = data.get(f.name, f.default)
    elif own is not None:  # a reward of one's own: its settings as written (None = left out)
        params = {param: data.get(param) for param, *_ in own.params}
    out: dict[str, Any] = {
        "name": name,
        "known": cls is not None or own is not None,
        "enabled": data.get("enabled", True),
        "weight": data.get("weight", 1.0),
        "zero_sum": _zero_sum(data.get("zero_sum")),
        "params": params,
    }
    if name == "all_of":
        out["parts"] = [dict(part) for part in data.get("parts", []) if isinstance(part, dict)]
    return out


def _zero_sum(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    return {
        "team_spirit": data.get("team_spirit", 0.0),
        "opponent_scale": data.get("opponent_scale", 1.0),
    }


def data_of(item: dict[str, Any]) -> dict[str, Any]:
    """An editor view as its `[[rewards]]` block writes it, settings at their defaults left out."""
    out: dict[str, Any] = {"name": item["name"]}
    if item.get("enabled") is False:
        out["enabled"] = False
    out["weight"] = _number(item.get("weight", 1.0))
    for key, value in (item.get("params") or {}).items():
        if value is not None and value != "":  # left out: its default (rewards of one's own)
            out[key] = value if isinstance(value, bool) else _number(value)
    if item.get("name") == "all_of":
        out["parts"] = [{k: _number(v) for k, v in part.items()} for part in item.get("parts") or []]
    zero_sum = item.get("zero_sum")
    if zero_sum:
        shared: dict[str, Any] = {"team_spirit": _number(zero_sum.get("team_spirit", 0.0))}
        if zero_sum.get("opponent_scale", 1.0) != 1.0:
            shared["opponent_scale"] = _number(zero_sum["opponent_scale"])
        out["zero_sum"] = shared
    return out


def _number(value: Any) -> Any:
    """Numbers as a person writes them (4, not 4.0); text that is not a number stays text, for the settings
    check to name."""
    if isinstance(value, str):
        cleaned = value.strip().replace("_", "").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    return value


def written(
    text: str,
    entries: list[dict[str, Any]],
    key: str = "rewards",
    data: Callable[[dict[str, Any]], dict[str, Any]] = data_of,
    kinds: dict[str, Any] | None = None,
) -> str:
    """`text` with its rewards replaced by `entries` (each {"from": the block it was, or None for a new one,
    "view": the reward}). A block keeps its comments and every line that did not change; new blocks go after
    the last one, and the comments that followed the last block (about the next section) stay last. Other
    lists of blocks are written the same way: `key` names them, `data` gives a view's lines and `kinds` the
    classes whose defaults are left out (the game starts, starts.py)."""
    doc = tomlkit.parse(text)
    old = doc.get(key)
    tables: list[Table] = list(old) if isinstance(old, AoT) else []
    kept = [e["from"] for e in entries if isinstance(e.get("from"), int) and 0 <= e["from"] < len(tables)]
    in_order = kept == sorted(set(kept)) and all(isinstance(e.get("from"), int) for e in entries[: len(kept)])
    ending = _comments_after(tables[-1]) if tables else []
    if isinstance(old, AoT) and in_order:
        for gone in sorted(set(range(len(tables))) - set(kept), reverse=True):
            del old[gone]
        aot = old
    else:  # reordered (or written some other way): the blocks are laid out again
        aot = tomlkit.aot()
    for number, entry in enumerate(entries, 1):
        index = entry.get("from")
        fresh = not (isinstance(index, int) and index in kept)
        table = tables[index] if isinstance(index, int) and not fresh else tomlkit.table()
        _patch(table, data(entry["view"]), fresh=fresh, kinds=KINDS if kinds is None else kinds)
        if fresh or aot is not old:
            aot.append(table)
        spaced = any(isinstance(item, Whitespace) for _, item in _tail(table))
        if aot is old and not spaced and (number < len(entries) or fresh or ending):
            table.add(tomlkit.nl())  # a blank line between blocks (a new list of blocks gets them anyway)
    if len(aot):
        aot[-1].value.body.extend(ending)
    if aot is not old:
        if key in doc:
            doc[key] = aot
        else:
            doc.append(key, aot)
    return tomlkit.dumps(doc)


def _tail(table: Table) -> list[tuple[Any, Any]]:
    """The blank lines and comments after a block's last setting."""
    body = table.value.body
    start = len(body)
    while start and body[start - 1][0] is None:
        start -= 1
    return body[start:]


def _comments_after(table: Table) -> list[tuple[Any, Any]]:
    """Takes the comments after the last block's settings (and the blank lines between them) out of it."""
    body = table.value.body
    tail = _tail(table)
    first = next((i for i, (_, item) in enumerate(tail) if isinstance(item, Comment)), len(tail))
    taken = tail[first:]
    del body[len(body) - len(taken) :]
    return taken


def _with_tail_kept(table: Table, change: Callable[[], None]) -> None:
    """Runs `change` (new lines are added at the block's end) with the block's blank lines and comments after
    its settings kept after them."""
    body = table.value.body
    tail = _tail(table)
    del body[len(body) - len(tail) :]
    try:
        change()
    finally:
        body.extend(tail)


_SAME: dict[str, Callable[[Any], Any]] = {"zero_sum": _zero_sum}


def _patch(table: Table, data: dict[str, Any], fresh: bool, kinds: dict[str, Any]) -> None:
    """Makes `table` say `data`, touching only the lines that change (a `fresh` block gets every line)."""
    _with_tail_kept(table, lambda: _patch_lines(table, data, fresh, kinds))


def _patch_lines(table: Table, data: dict[str, Any], fresh: bool, kinds: dict[str, Any]) -> None:
    cls = kinds.get(str(data.get("name")))
    defaults = {f.name: f.default for f in dataclasses.fields(cls)} if cls else {}
    for key, value in data.items():
        if key in table:
            now = table[key].unwrap() if hasattr(table[key], "unwrap") else table[key]
            same = _SAME.get(key, lambda v: v)
            if same(now) != same(value):
                table[key] = _item(value)
        elif fresh or value != defaults.get(key, dataclasses.MISSING):
            table[key] = _item(value)
    for key in ("enabled", "zero_sum"):
        if key in table and key not in data:
            now = table[key].unwrap() if hasattr(table[key], "unwrap") else table[key]
            if now != defaults.get(key):
                del table[key]


def _item(value: Any) -> Any:
    """`value` as a settings line holds it, written as the presets write theirs ({ a = 1 }, 500_000_000)."""
    return tomlkit.parse(f"v = {_toml(value)}\n")["v"]  # the newline ends its line


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:_}" if abs(value) >= 100_000 else str(value)
    if isinstance(value, float):
        return (
            repr(value)
            if math.isfinite(value)
            else ("nan" if value != value else f"{'-' if value < 0 else ''}inf")
        )
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml(inner)}" for key, inner in value.items()) + " }"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml(inner) for inner in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def canon(value: Any) -> Any:
    """A view compared as the file means it (4 and 4.0 alike)."""
    if isinstance(value, dict):
        return {k: canon(v) for k, v in value.items() if k != "known"}
    if isinstance(value, list):
        return [canon(v) for v in value]
    return _number(value)


def identity(view: dict[str, Any]) -> str | None:
    """What makes a reward the same as another: its name. An "all of" group is the same as another only
    with the same parts (two groups of different parts are two rewards); one without parts yet is none."""
    name = str(view.get("name", ""))
    if name != "all_of":
        return name or None
    parts = [canon(part) for part in view.get("parts") or [] if isinstance(part, dict)]
    if not parts:
        return None
    return "all_of:" + json.dumps(sorted(parts, key=lambda part: json.dumps(part, sort_keys=True)))


def duplicates(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rewards listed more than once (each copy pays on its own, so the reward counts that many times):
    {"name", "indexes": the place of every copy, the first one kept}."""
    found: dict[str, list[int]] = {}
    for index, view in enumerate(views):
        key = identity(view)
        if key is not None:
            found.setdefault(key, []).append(index)
    return [{"name": views[at[0]]["name"], "indexes": at} for at in found.values() if len(at) > 1]


def added_copies(before: list[dict[str, Any]], views: list[dict[str, Any]]) -> list[str]:
    """The names of the rewards `views` lists more often than once and more often than the settings file
    (`before`) does: a copy added to a reward the bot already has."""
    had: dict[str, int] = {}
    for view in before:
        if (key := identity(view)) is not None:
            had[key] = had.get(key, 0) + 1
    now: dict[str, int] = {}
    names: dict[str, str] = {}
    for view in views:
        if (key := identity(view)) is not None:
            now[key] = now.get(key, 0) + 1
            names[key] = str(view.get("name"))
    return [names[key] for key, count in now.items() if count > max(1, had.get(key, 0))]


def labels(views: list[dict[str, Any]]) -> list[str | None]:
    """Each reward's name in metrics.csv (`raw_<label>`, `paid_<label>`), None for one switched off."""
    on = numbered([item["name"] for item in views if item.get("enabled", True) is not False])
    found = iter(on)
    return [next(found) if item.get("enabled", True) is not False else None for item in views]


def changes(before: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[str]:
    """The waiting reward changes in plain words, one line each."""
    lines = []
    kept = set()
    for entry in entries:
        item, index = entry["view"], entry.get("from")
        if not isinstance(index, int) or not 0 <= index < len(before):
            lines.append(f"{item['name']}: added, weight {_show(item.get('weight'))}")
            continue
        kept.add(index)
        old = before[index]
        name = item["name"]
        if old.get("enabled", True) != item.get("enabled", True):
            lines.append(f"{name}: switched {'on' if item.get('enabled', True) else 'off'}")
        if _number(old.get("weight")) != _number(item.get("weight")):
            lines.append(f"{name}: weight {_show(old.get('weight'))} -> {_show(item.get('weight'))}")
        for key, value in (item.get("params") or {}).items():
            was = (old.get("params") or {}).get(key)
            if _number(was) != _number(value):
                lines.append(f"{name}: {describe.label(key).lower()} {_show(was)} -> {_show(value)}")
        lines.extend(_shared_changes(name, old, item))
    lines.extend(f"{old['name']}: removed" for i, old in enumerate(before) if i not in kept)
    return lines


def _shared_changes(name: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines = []
    was, now = old.get("zero_sum"), new.get("zero_sum")
    if bool(was) != bool(now):
        lines.append(f"{name}: zero-sum {'on' if now else 'off'}")
    elif was and now:
        for key, words in (("team_spirit", "team spirit"), ("opponent_scale", "opponent scale")):
            if _number(was.get(key)) != _number(now.get(key)):
                lines.append(f"{name}: {words} {_show(was.get(key))} -> {_show(now.get(key))}")
    if (old.get("parts") or []) != (new.get("parts") or []):
        parts = ", ".join(str(part.get("name")) for part in new.get("parts") or [])
        lines.append(f"{name}: parts -> {parts or '(none)'}")
    return lines


def _show(value: Any) -> str:
    value = _number(value)
    if isinstance(value, int) and abs(value) >= 10_000:
        return f"{value:,}"
    return f"{value:g}" if isinstance(value, float) else str(value)


def shares(folder: RunFolder, seconds: float = 60.0) -> dict[str, Any]:
    """Each reward's share of all the reward paid (by size: a penalty counts too) over the last `seconds` of
    training, with the share over the `seconds` before for comparison, from metrics.csv's `paid_` columns,
    and how often it fires (`fired_`): the share of agent steps and the times a minute of one car's
    play. An iteration's time is its agent steps over its speed."""
    m = bots.metrics(folder)
    if "agent_steps" not in m.columns or "steps_per_second" not in m.columns:
        return {"rewards": [], "seconds": 0.0, "note": "No training yet."}
    found = [c[len("paid_abs_") :] for c in m.header if c.startswith("paid_abs_")]
    if not found:
        return {
            "rewards": [],
            "seconds": 0.0,
            "note": "Shares appear once the bot trains with this version of NitroLearn.",
        }
    windows = _windows(m, seconds)
    now, before = (_window_sums(m, rows, found) for rows in windows)
    total = sum(item["size"] for item in now["by_label"].values()) or 0.0
    total_before = sum(item["size"] for item in before["by_label"].values()) or 0.0
    total_own = sum(item["own"] or 0.0 for item in now["by_label"].values())
    minute = decisions_a_minute(folder)
    out = []
    for label in found:
        sums, earlier = now["by_label"][label], before["by_label"][label]
        steps = now["steps"] or 1.0
        fire_rate = sums["fired"] / sums["fired_steps"] if sums["fired_steps"] else None
        out.append(
            {
                "label": label,
                "share": sums["size"] / total if total else 0.0,
                "share_before": earlier["size"] / total_before if total_before else None,
                "paid_per_step": sums["paid"] / steps,
                "size_per_step": sums["size"] / steps,
                "raw_per_step": sums["raw"] / steps if sums["raw_seen"] else None,
                # Before zero-sum (what the bots earn themselves, by size), and its share of all rewards so.
                "own_per_step": sums["own"] / steps if sums["own"] is not None else None,
                "own_share": sums["own"] / total_own if sums["own"] is not None and total_own else None,
                "fire_rate": fire_rate,
                "fires_per_minute": fire_rate * minute if fire_rate is not None else None,
                "seen": sums["seen"],
            }
        )
    out = [item for item in out if item["seen"]]
    out.sort(key=lambda item: item["share"], reverse=True)
    return {
        "rewards": out,
        "seconds": now["seconds"],
        "agent_steps": now["steps"],
        "updated": folder.metrics.stat().st_mtime if folder.metrics.is_file() else None,
        "alerts": alerts(out),
        "note": "" if out else "No reward paid in this window yet.",
        "fires_note": ""
        if any(item["fire_rate"] is not None for item in out) or not out
        else "Trigger counts appear once the bot trains with this version of NitroLearn.",
    }


def decisions_a_minute(folder: RunFolder) -> float:
    """One car's decisions in a minute of play at the run's tick skip (the default when unreadable)."""
    try:
        written = (
            tomllib.loads(folder.settings.read_text(encoding="utf-8-sig")).get("env", {}).get("tick_skip")
        )
    except (OSError, ValueError):
        written = None
    tick_skip = written if isinstance(written, int) and written > 0 else _DEFAULT_TICK_SKIP
    return 60 * 120 / tick_skip


def _windows(m: bots.Metrics, seconds: float) -> tuple[list[Any], list[Any]]:
    """The newest rows covering at least `seconds` of training, and the rows covering as much before them,
    read from the newest back (never the whole file). Each: (row, agent steps trained, how long it took)."""

    def timed() -> Any:
        later: int | None = None
        for row in range(m.count - 1, -1, -1):
            steps = m.value("agent_steps", row)
            if steps is None:
                continue
            if later is not None:
                trained = (m.value("agent_steps", later) or 0.0) - steps
                speed = m.value("steps_per_second", later)
                if trained > 0 and speed:
                    yield later, trained, trained / speed
            later = row

    rows = timed()
    now: list[Any] = []
    spent = 0.0
    for item in rows:
        if spent >= seconds:
            break
        now.append(item)
        spent += item[2]
    else:
        return now, []
    before = [item]
    spent_before = item[2]
    for item in rows:
        if spent_before >= max(spent, seconds):
            break
        before.append(item)
        spent_before += item[2]
    return now, before


def _window_sums(m: bots.Metrics, rows: list[Any], labels_found: list[str]) -> dict[str, Any]:
    by_label = {
        label: {
            "size": 0.0,
            "paid": 0.0,
            "raw": 0.0,
            "raw_seen": False,
            "seen": False,
            "own": None,
            "fired": 0.0,
            "fired_steps": 0.0,  # the steps of rows with a fired_ count
        }
        for label in labels_found
    }
    steps = sum(item[1] for item in rows)
    for row, trained, _ in rows:
        for label in labels_found:
            size = m.value(f"paid_abs_{label}", row)
            if size is None:
                continue
            sums = by_label[label]
            sums["seen"] = True
            sums["size"] += size * trained
            sums["paid"] += (m.value(f"paid_{label}", row) or 0.0) * trained
            own = m.value(f"own_abs_{label}", row)
            if own is not None:
                sums["own"] = (sums["own"] or 0.0) + own * trained
            raw = m.value(f"raw_{label}", row)
            if raw is not None:
                sums["raw"] += raw * trained
                sums["raw_seen"] = True
            fired = m.value(f"fired_{label}", row)
            if fired is not None:
                sums["fired"] += fired * trained
                sums["fired_steps"] += trained
    return {"by_label": by_label, "steps": steps, "seconds": sum(item[2] for item in rows)}


def title(label: str) -> str:
    """A metrics label as the page names it: `velocity_player_to_ball_2` -> 'Velocity player to ball 2'."""
    return describe.label(label)


def alerts(ranked: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Signs of a reward being farmed: the training's own warning (one reward over 90% of all paid while
    others pay too), and a reward whose share at least doubled, to a quarter or more, since the window
    before."""
    found = []
    paying = [item for item in ranked if item["share"] > 0]
    if len(paying) > 1 and paying[0]["share"] > DOMINANT_SHARE:
        top = paying[0]
        found.append(
            {
                "label": top["label"],
                "level": "warning",
                "text": f"'{title(top['label'])}' makes up {top['share']:.0%} of all reward paid: the bot "
                "may be farming it instead of playing. Consider lowering its weight.",
            }
        )
    for item in paying:
        before = item.get("share_before")
        if before is not None and item["share"] >= 0.25 and item["share"] >= 2 * before:
            found.append(
                {
                    "label": item["label"],
                    "level": "watch",
                    "text": f"'{title(item['label'])}' went from {before:.0%} to {item['share']:.0%} of all "
                    "reward paid since the window before: watch whether the bot is learning a trick to "
                    "collect it.",
                }
            )
    return found
