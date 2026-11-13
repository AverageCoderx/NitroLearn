"""Turns a TOML settings file into typed, validated, frozen config objects with plain-English errors."""

from __future__ import annotations

import dataclasses
import difflib
import math
import re
import tomllib
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")

_TYPE_NAMES = {
    bool: "true or false",
    int: "a whole number (like 8)",
    float: "a number (like 0.5)",
    str: 'text in quotes (like "abc")',
    Path: 'a folder or file path in quotes (like "collision_meshes")',
}


@dataclass(frozen=True)
class Rule:
    """Validation metadata attached to one setting."""

    help: str
    minimum: float | None = None
    maximum: float | None = None


def setting(default: Any, help: str, *, minimum: float | None = None, maximum: float | None = None) -> Any:
    """Declares a setting with a default value, an explanation, and optional numeric bounds."""
    return field(default=default, metadata={"rule": Rule(help, minimum, maximum)})


def section(cls: type[Any], help: str) -> Any:
    """Declares a `[section]` whose settings all have defaults."""
    return field(default_factory=cls, metadata={"rule": Rule(help)})


@dataclass(frozen=True)
class ConfigIssue:
    """One problem found in a settings file."""

    path: str
    message: str
    line: int | None = None
    hint: str | None = None

    def describe(self) -> str:
        where = f"line {self.line}" if self.line is not None else "file"
        name = f", {self.path}" if self.path else ""
        text = f"{where}{name}: {self.message}"
        return f"{text}\n      hint: {self.hint}" if self.hint else text


class ConfigError(Exception):
    """Raised with every problem found, so all of them can be fixed in one go."""

    def __init__(self, file: Path, issues: list[ConfigIssue]) -> None:
        self.file = file
        # In the order of the file's lines (problems of the whole file first), so they read top to bottom.
        self.issues = sorted(issues, key=lambda issue: -1 if issue.line is None else issue.line)
        super().__init__(str(self))

    def __str__(self) -> str:
        count = len(self.issues)
        header = f"{count} problem{'s' if count != 1 else ''} in {self.file}:"
        return "\n".join([header, *(f"  - {issue.describe()}" for issue in self.issues)])


class KeyLocator:
    """Maps dotted setting paths (e.g. `rewards[1].weight`) to the line where they are written."""

    _ARRAY_HEADER = re.compile(r"^\[\[\s*([^\]]+?)\s*\]\]")
    _TABLE_HEADER = re.compile(r"^\[\s*([^\]]+?)\s*\]")

    def __init__(self, text: str) -> None:
        self._lines: dict[str, int] = {}
        table = ""
        array_counts: dict[str, int] = {}
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if match := self._ARRAY_HEADER.match(line):
                name = match.group(1)
                index = array_counts.get(name, 0)
                array_counts[name] = index + 1
                self._lines.setdefault(name, lineno)
                table = f"{name}[{index}]"
            elif match := self._TABLE_HEADER.match(line):
                table = match.group(1)
            elif "=" in line:
                key = line.split("=", 1)[0].strip().strip("\"'")
                self._lines.setdefault(f"{table}.{key}" if table else key, lineno)
                continue
            else:
                continue
            self._lines.setdefault(table, lineno)

    def line_of(self, path: str) -> int | None:
        """Line of `path`, or of the nearest enclosing table that is written in the file."""
        while path:
            if path in self._lines:
                return self._lines[path]
            path = path[: path.rindex("[")] if path.endswith("]") else path.rpartition(".")[0]
        return None


@dataclass
class _Context:
    locator: KeyLocator
    base_dir: Path
    issues: list[ConfigIssue] = field(default_factory=list)

    def report(self, path: str, message: str, hint: str | None = None) -> None:
        self.issues.append(ConfigIssue(path, message, self.locator.line_of(path), hint))


def load_toml(file: Path, schema: type[T]) -> T:
    """Reads `file` and builds `schema` from it. Relative paths are resolved against the file's folder."""
    try:
        # utf-8-sig: Windows tools (PowerShell 5.1, some editors) often add a byte-order mark.
        text = file.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ConfigError(file, [ConfigIssue("", "the settings file does not exist")]) from None
    except OSError as err:
        raise ConfigError(
            file, [ConfigIssue("", f"the settings file cannot be read ({err.strerror})")]
        ) from None
    return parse_toml(text, schema, file=file, base_dir=file.resolve().parent)


def parse_toml(text: str, schema: type[T], *, file: Path, base_dir: Path) -> T:
    """Builds `schema` from TOML text; `file` is only used in error messages."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        line = getattr(err, "lineno", None)
        message = str(err)
        if line is None and (match := re.search(r"line (\d+)", message)):
            line = int(match.group(1))
        message = re.sub(r"\s*\(at line \d+, column \d+\)", "", message)
        plain = "this line cannot be read: it is not written the way a settings file is"
        raise ConfigError(file, [ConfigIssue("", f"{plain} (TOML says: {message})", line)]) from None

    context = _Context(KeyLocator(text), base_dir)
    result = _build(schema, data, "", context)
    if context.issues:
        raise ConfigError(file, context.issues)
    return result


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    """A settings class's field types, read once (reading them compiles each annotation: half a parse)."""
    found = _HINTS.get(cls)
    if found is None:
        found = _HINTS[cls] = get_type_hints(cls)
    return found


def _build(cls: type[T], table: dict[str, Any], prefix: str, context: _Context) -> T:
    hints = _hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    for key in table:
        if key not in known:
            close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
            hint = f"did you mean '{close[0]}'?" if close else f"valid settings here: {', '.join(known)}"
            context.report(_join(prefix, key), f"'{key}' is not a known setting", hint)

    values: dict[str, Any] = {}
    for name, f in known.items():
        path = _join(prefix, name)
        if name in table:
            values[name] = _convert(hints[name], table[name], path, f, context)
        elif _is_section(hints[name]):
            values[name] = _build(hints[name], {}, path, context)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            context.report(path, "this setting is required but missing")
            values[name] = None
        elif isinstance(f.default, Path) and not f.default.is_absolute():
            values[name] = (context.base_dir / f.default).resolve()
            if fallback := getattr(cls, "default_path", None):  # where an unwritten path points
                values[name] = fallback(name, values[name])
    result = cls(**values)
    check = getattr(result, "problems", None)
    if check is not None and not any(issue.path.startswith(prefix) for issue in context.issues):
        for key, message in check():
            context.report(_join(prefix, key), message)
    return result


def _convert(tp: Any, raw: Any, path: str, f: dataclasses.Field[Any], context: _Context) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        inner = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(inner) > 1:
            return _convert_named(inner, raw, path, context)
        return _convert(inner[0], raw, path, f, context)
    if _is_section(tp):
        if not isinstance(raw, dict):
            context.report(path, f"must be a section [{path}], not a single value")
            return _build(tp, {}, path, context)
        return _build(tp, raw, path, context)
    if origin is tuple:
        item_type = get_args(tp)[0]
        if item_type is Path and isinstance(raw, str):  # one folder where a list of them is allowed
            raw = [raw]
        if not isinstance(raw, list):
            context.report(path, "must be a list")
            return ()
        return tuple(_convert(item_type, item, f"{path}[{i}]", f, context) for i, item in enumerate(raw))
    if origin is Literal:
        choices = get_args(tp)
        if raw not in choices:
            listed = ", ".join(repr(choice) for choice in choices)
            one = "" if len(choices) == 1 else "one of "
            context.report(path, f"must be {one}{listed}, but is {raw!r}")
            return choices[0]
        return raw
    return _convert_scalar(tp, raw, path, f, context)


def _convert_named(options: list[Any], raw: Any, path: str, context: _Context) -> Any:
    """Picks the section type whose `name` matches `raw["name"]` (e.g. which reward a `[[rewards]]` is). A
    type without a fixed name takes the names its `accepts` knows (rewards of one's own)."""
    by_name = {_name_of(option): option for option in options if _name_of(option)}
    if not isinstance(raw, dict) or "name" not in raw:
        context.report(path, "needs a name", f"one of: {', '.join(by_name)}")
        return None
    name = raw["name"]
    for option in options:
        if name not in by_name and not _name_of(option) and option.accepts(name):
            return option.build(
                raw, path, lambda cls, table, at: _build(cls, table, at, context), context.report
            )
    if name not in by_name:
        known = list(by_name) + [n for option in options if not _name_of(option) for n in option.names()]
        close = difflib.get_close_matches(str(name), known, n=1, cutoff=0.6)
        hint = f"did you mean '{close[0]}'?" if close else f"valid names: {', '.join(known)}"
        context.report(_join(path, "name"), f"'{name}' is not a known name here", hint)
        return None
    return _build(by_name[name], raw, path, context)


def _name_of(cls: type[Any]) -> str:
    name_field = next(f for f in dataclasses.fields(cls) if f.name == "name")
    return str(name_field.default)


def _convert_scalar(tp: Any, raw: Any, path: str, f: dataclasses.Field[Any], context: _Context) -> Any:
    ok = {
        bool: isinstance(raw, bool),
        int: isinstance(raw, int) and not isinstance(raw, bool),
        float: isinstance(raw, int | float) and not isinstance(raw, bool),
        str: isinstance(raw, str),
        Path: isinstance(raw, str),
    }.get(tp)
    if ok is None:
        raise TypeError(f"unsupported setting type {tp!r} at {path}")
    if not ok:
        context.report(path, f"must be {_TYPE_NAMES[tp]}, but is {raw!r}")
        return f.default if f.default is not dataclasses.MISSING else None
    if tp is Path:
        given = Path(raw)
        return given if given.is_absolute() else (context.base_dir / given).resolve()
    value = float(raw) if tp is float else raw
    if isinstance(value, float) and not math.isfinite(value):
        context.report(path, f"must be an ordinary number, but is {value}")
        return f.default if f.default is not dataclasses.MISSING else None
    rule: Rule | None = f.metadata.get("rule")
    if rule is not None and isinstance(value, int | float):
        if rule.minimum is not None and value < rule.minimum:
            context.report(path, f"must be at least {rule.minimum:g}, but is {value:g}", rule.help)
        if rule.maximum is not None and value > rule.maximum:
            context.report(path, f"must be at most {rule.maximum:g}, but is {value:g}", rule.help)
    return value


def _is_section(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key
