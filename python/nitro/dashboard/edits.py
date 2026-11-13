"""Changes made in the dashboard, written into a bot's settings file only on Apply (the user chooses
when). The file keeps its comments and layout: only the changed lines change."""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.items import Table

from nitro.config.loader import ConfigError, ConfigIssue, parse_toml
from nitro.config.schema import NitroConfig


class EditError(ValueError):
    """A value the setting cannot take, in plain English."""


def convert(item: dict[str, Any], raw: Any) -> Any:
    """The dashboard's input for `item` (one setting described by `describe`) as the settings file holds it;
    None leaves the setting unwritten (its default)."""
    kind = item["kind"]
    empty = raw is None or (isinstance(raw, str) and raw.strip() == "" and kind != "text")
    if empty and item["nullable"]:
        return None
    if empty and kind != "bool":
        # Left empty, the line (and its comment) would go and the setting take its default unseen.
        raise EditError("it needs a value (Undo puts back the one in the file)")
    try:
        if kind == "bool":
            return bool(raw)
        if kind == "int":
            number = _finite(float(str(raw).replace("_", "").replace(",", "")))
            if number != int(number):
                raise EditError("a whole number is needed")
            return _bounded(item, int(number))
        if kind == "float":
            return _bounded(item, _finite(float(str(raw).replace("_", "").replace(",", ""))))
        if kind == "numbers":
            text = raw if isinstance(raw, list) else str(raw).strip("[] ")
            if isinstance(text, str) and ", " in text and re.search(r"\d,\d", text):
                # "1,024, 1,024": a comma inside a number as well as between them.
                raise EditError("write each number without commas (1024 or 1_024), with ', ' between them")
            parts = text if isinstance(text, list) else text.replace(",", " ").split()
            numbers = [_finite(float(str(p).replace("_", ""))) for p in parts]
            return [int(n) if n == int(n) else n for n in numbers]
        if kind == "words":
            words = raw if isinstance(raw, list) else str(raw).strip("[] ").split(",")
            return [w.strip(" '\"") for w in words if w.strip(" '\"")]
        if kind == "choice":
            for option in item["options"]:
                if str(option) == str(raw):
                    return option
            raise EditError("pick one of " + ", ".join(str(o) for o in item["options"]))
        if kind == "path":
            return str(raw).strip().strip('"').strip("'")  # Windows' "Copy as path" adds quotes
        if kind == "text":
            return str(raw)
    except (TypeError, ValueError, OverflowError) as err:
        if isinstance(err, EditError):
            raise
        raise EditError(f"'{raw}' is not {_kind_words(kind)}") from err
    raise EditError("this setting is changed in the settings file itself for now")


def _finite(number: float) -> float:
    if not math.isfinite(number):
        raise EditError("a finite number is needed")
    return number


def _bounded(item: dict[str, Any], number: float) -> Any:
    low, high = item.get("minimum"), item.get("maximum")
    if low is not None and number < low:
        raise EditError(f"the smallest allowed is {low:g}")
    if high is not None and number > high:
        raise EditError(f"the largest allowed is {high:g}")
    return number


def _kind_words(kind: str) -> str:
    return {
        "int": "a whole number",
        "float": "a number",
        "numbers": "a list of numbers like 1024, 1024",
    }.get(kind, "a value this setting takes")


def written(text: str, changes: dict[str, Any]) -> str:
    """`text` (a settings file) with `changes` (dotted path -> value, None to remove the line) written in."""
    doc = tomlkit.parse(text)
    for path, value in changes.items():
        *sections, key = path.split(".")
        table: Any = doc
        for name in sections:
            if name not in table:
                if value is None:
                    break
                table[name] = tomlkit.table()
            table = table[name]
        else:
            if value is None:
                if key in table:
                    del table[key]
            else:
                table[key] = value
    return tomlkit.dumps(doc)


def replaced(text: str, name: str, values: dict[str, Any], comment: str) -> str:
    """`text` with its table [name] (of plain values) written anew as `values` under a comment line, at the
    end: the old table goes with its comments, those on its lines and those just above its header (they were
    about the old values)."""
    lines = text.splitlines(keepends=True)
    header = re.compile(rf"^\s*\[\s*{re.escape(name)}\s*\]\s*(#.*)?$")
    first = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if first is not None:
        end = next((i for i in range(first + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        while end > first + 1 and lines[end - 1].lstrip().startswith("#"):
            end -= 1  # the next table's own comments
        while first > 0 and lines[first - 1].lstrip().startswith("#"):
            first -= 1
        lines[first:end] = []
    doc = tomlkit.parse("".join(lines))
    if name in doc:  # written as dotted keys or inline
        del doc[name]
    table = tomlkit.table()
    table.add(tomlkit.comment(comment))
    for key, value in values.items():
        table.add(key, value)
    doc.add(name, table)
    return tomlkit.dumps(doc)


def problems(text: str, file: Path, base_dir: Path) -> list[str]:
    """What is wrong with `text` as a settings file, one plain-English line each (empty when fine)."""
    return [issue.describe() for issue in issues(text, file, base_dir)]


def issues(text: str, file: Path, base_dir: Path) -> list[ConfigIssue]:
    """What is wrong with `text` as a settings file, each problem with the setting it is about."""
    try:
        parse_toml(text, NitroConfig, file=file, base_dir=base_dir)
    except ConfigError as err:
        return list(err.issues)
    except Exception as err:  # a file tomlkit wrote but the reader refuses: say it, never crash the window
        return [ConfigIssue("", str(err))]
    return []


def save(file: Path, text: str) -> None:
    """Writes `text` to `file` in one step, so a crash never leaves half a settings file. Windows can refuse
    the swap for a moment while another program (a virus scanner) reads the file: tried again then."""
    partial = file.with_name(file.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(partial, file)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def table_paths(text: str) -> set[str]:
    """Every dotted key written in `text` (for tests and the editor's 'written here' marks)."""
    found: set[str] = set()

    def walk(table: Any, prefix: str) -> None:
        for key, value in table.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Table | dict):
                walk(value, path)
            else:
                found.add(path)

    walk(tomlkit.parse(text), "")
    return found
