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
from nitro.dashboard import bots, describe, edits, reward_guide
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

# A reward built from blocks: its title names it in metrics.csv, its terms say what it pays.
BUILT = "built"
_TERM_DEFAULTS: dict[str, Any] = {
    "when": [],
    "combine": "product",
    "weight": 1,
    "once_per_touch": False,
    "once_per_episode": False,
    "every_step": False,
}
_VAR_DEFAULTS: dict[str, Any] = {"start": 0}
# A built reward's lists of blocks, in the order they are written: its variables, then its terms.
_BUILT_LISTS = ("vars", "terms")


def key_name(view: dict[str, Any]) -> str:
    """A reward's name in metrics.csv and the page's layout: its name, or a built reward's title."""
    name = str(view.get("name", ""))
    return str(view.get("title") or name) if name == BUILT else name


def blocks() -> dict[str, Any]:
    """The building blocks of built rewards with their plain-English help (nitro.config.built), for the
    editor."""
    from nitro.config import built

    return {
        "flags": [{"name": n, "help": h, "event": n in built.EVENT_FLAGS} for n, h in built.FLAGS.items()],
        "numbered": [{"name": n, "help": h} for n, h in built.NUMBERED.items()],
        "measures": [{"name": n, "help": h, "every_step": e} for n, (h, e) in built.MEASURES.items()],
        "term_keys": dict(built.TERM_KEYS),
        "var_keys": dict(built.VAR_KEYS),
        "formula": {
            "car": [{"name": n, "help": h} for n, h in built.CAR_FIELDS.items()],
            "ball": [{"name": n, "help": h} for n, h in built.BALL_FIELDS.items()],
            "names": [{"name": n, "help": h} for n, h in built.NAMES.items()],
            "functions": [{"name": n, "help": h} for n, h in built.FUNCTIONS.items()],
        },
        "templates": templates(),
    }


TEMPLATES = Path(__file__).resolve().parents[1] / "config" / "built_templates.json"


@functools.cache
def _templates_text() -> str:
    return TEMPLATES.read_text(encoding="utf-8")


def templates() -> list[dict[str, Any]]:
    """The builder's templates (python/nitro/config/built_templates.json): each a built reward to start from,
    with what it pays and, for those that copy a reward, which (the core's tests prove the copies equal)."""
    out = []
    for item in json.loads(_templates_text()):
        copies = item.get("copies") or {}
        copied = copies.get("kind") if copies.get("name") == "custom" else copies.get("name")
        out.append(
            {
                "title": item["title"],
                "about": item["about"],
                "copies": copied if copied and copies.get("name") != "test" else "",
                "vars": item.get("vars", []),
                "terms": item["terms"],
            }
        )
    return out


def built_problems(view: dict[str, Any]) -> dict[str, Any]:
    """A built reward's view checked as the settings check and the core will: its problems, each with the
    place it is in ("title", "terms[0].value", "vars[1].next" ...), whether it is paid every step and how
    many numbers it remembers per car (from the core, which alone can judge formulas)."""
    from nitro.config import built
    from nitro.config.rewards import _TITLE_LETTERS, _builtin_names

    data = data_of({**view, "name": BUILT})
    title, terms, variables = data.get("title", ""), data.get("terms", []), data.get("vars", [])
    problems: list[dict[str, str]] = []
    if not title or not all(c in _TITLE_LETTERS for c in title):
        problems.append({"where": "title", "message": "a title of small letters, digits and _ is needed"})
    elif title in _builtin_names() or title in custom_kinds():
        problems.append({"where": "title", "message": f"'{title}' is already a reward's name"})
    if not terms:
        problems.append({"where": "terms", "message": "a built reward needs at least one term"})
    known = frozenset(str(v.get("name")) for v in variables if isinstance(v, dict))
    for i, var in enumerate(variables):
        for key, message in built.var_problems(var):
            problems.append({"where": f"vars[{i}]" + (f".{key}" if key else ""), "message": message})
    for i, term in enumerate(terms):
        for key, message in built.term_problems(term, known):
            problems.append({"where": f"terms[{i}]" + (f".{key}" if key else ""), "message": message})
    every_step, memory = False, 0
    if not problems:
        where, message, every_step, memory = built.core_check(title, variables, terms)
        if message:
            problems.append({"where": where or "terms", "message": message})
    return {"problems": problems, "every_step": every_step, "memory": memory}


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
    out.append(
        {
            "name": BUILT,
            "title": "Build your own (no code)",
            "about": "A reward made from building blocks, no code: each term pays a measure (speed toward "
            "the ball, a goal, the ball's height, ...) times its weight on the steps where all of its "
            "conditions hold (in the air, touching the ball, ball above a height, ...). It can be added more "
            "than once.",
            "every_step": False,  # worked out from its terms, reward by reward, on the page
            "group": False,
            "built": True,
            "repeatable": True,
            "params": [],
            "more": reward_guide.MORE.get(BUILT, ""),
            "nerd": _nerd(BUILT),
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
        "known": cls is not None or own is not None or name == BUILT,
        "enabled": data.get("enabled", True),
        "weight": data.get("weight", 1.0),
        "zero_sum": _zero_sum(data.get("zero_sum")),
        "params": params,
    }
    if name == "all_of":
        out["parts"] = [dict(part) for part in data.get("parts", []) if isinstance(part, dict)]
    if name == BUILT:
        out["title"] = str(data.get("title", ""))
        for kind in _BUILT_LISTS:
            items = data.get(kind)
            out[kind] = (
                [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []
            )
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
    if item.get("name") == BUILT:
        out["title"] = str(item.get("title") or "")
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
    if item.get("name") == BUILT:  # last: they are written as [[rewards.vars]] and [[rewards.terms]] blocks
        out["vars"] = [_var_data(var) for var in item.get("vars") or [] if isinstance(var, dict)]
        out["terms"] = [_term_data(term) for term in item.get("terms") or [] if isinstance(term, dict)]
    return out


def _var_data(var: dict[str, Any]) -> dict[str, Any]:
    """A built reward's variable as its block writes it: name, start (left out at 0) and next."""
    out: dict[str, Any] = {"name": str(var.get("name", "")).strip()}
    start = var.get("start", 0)
    start = start if isinstance(start, bool) else _number(start if start not in (None, "") else 0)
    if start != 0:
        out["start"] = start
    out["next"] = str(var.get("next", "")).strip()
    for key, value in var.items():  # anything else stays, for the settings check to name
        if key not in out and key != "start":
            out[key] = value
    return out


def _term_data(term: dict[str, Any]) -> dict[str, Any]:
    """A built reward's term as its block writes it: one measure as text, several as a list, the rest left
    out at their defaults; numbers as a person writes them."""
    out: dict[str, Any] = {}
    when = [_condition_data(c) for c in _as_list(term.get("when"))]
    if when:
        out["when"] = when
    value = term.get("value")
    if isinstance(value, str) and value.strip():  # a formula pays: no measures, no combine
        out["value"] = value.strip()
    else:
        measures = [str(m) for m in _as_list(term.get("measure")) if m not in (None, "")]
        out["measure"] = measures[0] if len(measures) == 1 else measures
    for key in ("combine", "weight", "once_per_touch", "once_per_episode", "every_step"):
        if key == "combine" and "value" in out:
            continue
        number = term.get(key, _TERM_DEFAULTS[key])
        number = number if isinstance(number, bool) else _number(number)
        if number != _TERM_DEFAULTS[key]:
            out[key] = number
    for key in ("cap_step", "cap_episode"):
        if term.get(key) not in (None, ""):
            out[key] = _number(term[key])
    for key, kept in term.items():  # anything else stays, for the settings check to name
        if (
            key not in out
            and key not in _TERM_DEFAULTS
            and key not in ("measure", "value", "cap_step", "cap_episode")
        ):
            out[key] = kept
    return out


def _condition_data(condition: Any) -> Any:
    if isinstance(condition, str):
        return condition.strip()
    if isinstance(condition, dict):
        return {
            key: _condition_data(value) if key == "not" else _number(value)
            for key, value in condition.items()
        }
    return condition


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _number(value: Any) -> Any:
    """Numbers as a person writes them (4, not 4.0); text that is not a number stays text, for the settings
    check to name."""
    if isinstance(value, str):
        try:
            value = float(edits.ungrouped(value))
        except ValueError:  # a decimal comma (0,5) included: never read as a bigger number
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
    ending = _comments_after(_deepest(tables[-1])) if tables else []
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
        last = _deepest(table)
        spaced = any(isinstance(item, Whitespace) for _, item in _tail(last))
        if aot is old and not spaced and (number < len(entries) or fresh or ending):
            last.add(tomlkit.nl())  # a blank line between blocks (a new list of blocks gets them anyway)
    if len(aot):
        _deepest(aot[-1]).value.body.extend(ending)
    if aot is not old:
        if key in doc:
            doc[key] = aot
        else:
            doc.append(key, aot)
    return tomlkit.dumps(doc)


def _deepest(table: Table) -> Table:
    """The block whose lines come last in the file for `table`: a built reward's last term, else itself."""
    body = table.value.body
    while body and isinstance(body[-1][1], AoT) and len(body[-1][1]):
        table = body[-1][1][-1]
        body = table.value.body
    return table


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
    if data.get("name") == BUILT and "terms" in data:  # its blocks first; its own lines go above them
        lists = {kind: data.get(kind) or [] for kind in _BUILT_LISTS}
        data = {key: value for key, value in data.items() if key not in _BUILT_LISTS}
        if fresh and not any(kind in table for kind in _BUILT_LISTS):
            _with_tail_kept(table, lambda: _patch_lines(table, data, fresh, kinds))
            table.add(tomlkit.nl())
            for kind, items in lists.items():
                _patch_blocks(table, kind, items)
            return
        for kind, items in lists.items():
            _patch_blocks(table, kind, items)
    if any(isinstance(item, AoT | Table) for _, item in table.value.body):
        _patch_lines(table, data, fresh, kinds)
    else:
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
            _add_line(table, key, _item(value))
    for key in ("enabled", "zero_sum"):
        if key in table and key not in data:
            now = table[key].unwrap() if hasattr(table[key], "unwrap") else table[key]
            if now != defaults.get(key):
                del table[key]


def _add_line(table: Table, key: str, item: Any) -> None:
    """A new line at the end of the block's own lines: before the blank lines and comments that lead into a
    built reward's term blocks, if it has any."""
    body = table.value.body
    first = next((i for i, (_, it) in enumerate(body) if isinstance(it, AoT | Table)), None)
    if first is None:
        table[key] = item
        return
    while first and body[first - 1][0] is None:
        first -= 1
    table.value._insert_at(first, key, item)


def _patch_blocks(table: Table, kind: str, terms: list[dict[str, Any]]) -> None:
    """A built reward's `[[rewards.terms]]` (or `[[rewards.vars]]`, `kind`) blocks made to say `terms`: a
    block's unchanged lines and comments stay, a line at its default stays as written, removed blocks go and
    new ones come last."""
    defaults = _TERM_DEFAULTS if kind == "terms" else _VAR_DEFAULTS
    old = table.get(kind)
    if not terms:
        if old is not None:
            del table[kind]
        return
    if not isinstance(old, AoT):  # none yet, or written as one list: written again as blocks
        if old is not None and _plain(old) == terms:
            return
        aot = tomlkit.aot()
        for term in terms:
            aot.append(_term_table(term))
        if kind == "vars" and "terms" in table:  # variables go above the terms they are read in
            terms_now = table["terms"]
            del table["terms"]
            table[kind] = aot
            table["terms"] = terms_now
        else:
            table[kind] = aot
        return
    ending = _comments_after(old[-1]) if len(old) else []  # about what follows the blocks: stays last
    was_last = old[-1] if len(old) else None
    for gone in reversed(_removed_blocks([_block_of(block, defaults) for block in old], terms)):
        # The comments right above the removed block go with it; those after its lines (above the next
        # block) stay, in their place.
        after = _comments_after(old[gone])
        if gone:
            body = old[gone - 1].value.body
            del body[len(body) - len(_heading(body)) :]
            body.extend(after)
        else:
            _swap_heading(table, kind, after)
        del old[gone]
    for i, term in enumerate(terms):
        if i >= len(old):
            if not any(isinstance(it, Whitespace) for _, it in _tail(old[i - 1])):
                old[i - 1].add(tomlkit.nl())
            old.append(_term_table(term))
            continue
        block = old[i]
        for key, value in term.items():
            if key not in block or _plain(block[key]) != value:
                block[key] = _item(value)
        for key in [k for k in block if k not in term]:
            if _plain(block[key]) != defaults.get(key, dataclasses.MISSING):
                del block[key]
    last = old[-1]
    if ending and last is not was_last and not any(isinstance(item, Whitespace) for _, item in _tail(last)):
        last.add(tomlkit.nl())
    last.value.body.extend(ending)


def _heading(body: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """The comment lines at the end of `body` with no blank line after them: they head what comes next."""
    start = len(body)
    while start and body[start - 1][0] is None and isinstance(body[start - 1][1], Comment):
        start -= 1
    return body[start:]


def _swap_heading(table: Table, kind: str, lines: list[tuple[Any, Any]]) -> None:
    """The comments heading a built reward's first `kind` block replaced by `lines`: they end the blocks
    written above it (its variables, for its first term), else the reward's own lines."""
    body = table.value.body
    at = next(i for i, (key, item) in enumerate(body) if isinstance(item, AoT) and key and key.key == kind)
    above = body[at - 1][1] if at else None
    if isinstance(above, AoT) and len(above):
        held = above[-1].value.body
        del held[len(held) - len(_heading(held)) :]
        held.extend(lines)
        return
    start = at - len(_heading(body[:at]))
    body[start:at] = lines
    shift = len(lines) - (at - start)
    index = table.value._map
    for key, value in index.items():
        if isinstance(value, tuple):
            index[key] = tuple(v + shift if v >= at else v for v in value)
        elif value >= at:
            index[key] = value + shift


def _block_of(block: Table, defaults: dict[str, Any]) -> dict[str, Any]:
    """A block's settings as `_term_data` (or `_var_data`) gives them: lines at their defaults left out."""
    out = {key: _plain(block[key]) for key in block}
    return {key: value for key, value in out.items() if value != defaults.get(key, dataclasses.MISSING)}


def _removed_blocks(old: list[dict[str, Any]], terms: list[dict[str, Any]]) -> list[int]:
    """Which of the `old` blocks were removed to leave `terms`: the blocks that still say a term unchanged
    keep it (the longest such match in order); between them, edited terms keep the first blocks and the
    rest are removed."""
    extra = len(old) - len(terms)
    if extra <= 0:
        return []
    # Longest common subsequence of unchanged terms.
    best = [[0] * (len(terms) + 1) for _ in range(len(old) + 1)]
    for i in range(len(old) - 1, -1, -1):
        for j in range(len(terms) - 1, -1, -1):
            best[i][j] = best[i + 1][j + 1] + 1 if old[i] == terms[j] else max(best[i + 1][j], best[i][j + 1])
    removed: list[int] = []
    i = j = 0
    gap_old: list[int] = []
    gap_terms = 0

    def close_gap() -> None:
        removed.extend(gap_old[gap_terms:])

    while i < len(old):
        if j < len(terms) and old[i] == terms[j] and best[i][j] == best[i + 1][j + 1] + 1:
            close_gap()
            gap_old, gap_terms = [], 0
            i, j = i + 1, j + 1
        elif j < len(terms) and best[i][j + 1] >= best[i + 1][j]:
            gap_terms, j = gap_terms + 1, j + 1
        else:
            gap_old.append(i)
            i += 1
    gap_terms += len(terms) - j
    close_gap()
    return removed[:extra]


def _term_table(term: dict[str, Any]) -> Table:
    block = tomlkit.table()
    for key, value in term.items():
        block[key] = _item(value)
    return block


def _plain(item: Any) -> Any:
    """A settings line's value as plain Python (lists and tables too), numbers as a person writes them."""
    value = item.unwrap() if hasattr(item, "unwrap") else item
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value if isinstance(value, bool) else _number(value)


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
    if name == BUILT:  # built rewards are told apart by their titles
        return f"{BUILT}:{view.get('title') or ''}"
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
    (`before`) does: a copy added to a reward the bot already has. A built reward is "built:<its title>"."""
    had: dict[str, int] = {}
    for view in before:
        if (key := identity(view)) is not None:
            had[key] = had.get(key, 0) + 1
    now: dict[str, int] = {}
    names: dict[str, str] = {}
    for view in views:
        if (key := identity(view)) is not None:
            now[key] = now.get(key, 0) + 1
            names[key] = (
                f"{BUILT}:{view.get('title') or ''}" if view.get("name") == BUILT else str(view.get("name"))
            )
    return [names[key] for key, count in now.items() if count > max(1, had.get(key, 0))]


def labels(views: list[dict[str, Any]]) -> list[str | None]:
    """Each reward's name in metrics.csv (`raw_<label>`, `paid_<label>`), None for one switched off."""
    on = numbered([key_name(item) for item in views if item.get("enabled", True) is not False])
    found = iter(on)
    return [next(found) if item.get("enabled", True) is not False else None for item in views]


def changes(before: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[str]:
    """The waiting reward changes in plain words, one line each."""
    lines = []
    kept = set()
    for entry in entries:
        item, index = entry["view"], entry.get("from")
        if not isinstance(index, int) or not 0 <= index < len(before):
            lines.append(f"{key_name(item)}: added, weight {_show(item.get('weight'))}")
            continue
        kept.add(index)
        old = before[index]
        name = key_name(item)
        if old.get("enabled", True) != item.get("enabled", True):
            lines.append(f"{name}: switched {'on' if item.get('enabled', True) else 'off'}")
        if _number(old.get("weight")) != _number(item.get("weight")):
            lines.append(f"{name}: weight {_show(old.get('weight'))} -> {_show(item.get('weight'))}")
        for key, value in (item.get("params") or {}).items():
            was = (old.get("params") or {}).get(key)
            if _number(was) != _number(value):
                lines.append(f"{name}: {describe.label(key).lower()} {_show(was)} -> {_show(value)}")
        lines.extend(_shared_changes(name, old, item))
    lines.extend(f"{key_name(old)}: removed" for i, old in enumerate(before) if i not in kept)
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
    if old.get("title") != new.get("title") and new.get("name") == BUILT:
        lines.append(f"{key_name(old)}: renamed to {key_name(new)}")
    if canon(old.get("terms") or []) != canon(new.get("terms") or []):
        count = len(new.get("terms") or [])
        lines.append(f"{name}: terms changed ({count} term{'' if count == 1 else 's'})")
    if canon(old.get("vars") or []) != canon(new.get("vars") or []):
        count = len(new.get("vars") or [])
        lines.append(f"{name}: variables changed ({count} variable{'' if count == 1 else 's'})")
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
