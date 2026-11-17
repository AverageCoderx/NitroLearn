"""Reads a GigaLearn project's C++ setup: the rewards and weights, the observation builder,
action parser, state setter and terminal conditions its environment function makes, and the `LearnerConfig`
values its main function sets. Only plain code is understood (literals, simple arithmetic, local variables);
anything else is listed as not read, never guessed."""

from __future__ import annotations

import ast
import contextlib
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A C++ value as read: a number, a bool, a string (an enum name such as "RELU"), a list, or a constructor.
Value = Any


@dataclass(frozen=True)
class Call:
    """`new Name(args)` (or `Name(args)`): the class name without namespaces, its arguments, its line."""

    name: str
    args: tuple[Value, ...]
    line: int
    text: str  # as written, for reports


@dataclass(frozen=True)
class WeightedEntry:
    reward: Call
    weight: Value  # a number, or None when the weight could not be read
    line: int


@dataclass
class Source:
    """What `read` found; `unread` lists what was seen but could not be read, with its line."""

    file: Path
    rewards: list[WeightedEntry] = field(default_factory=list)
    obs: Call | None = None
    action: Call | None = None
    setter: Call | None = None
    terminals: list[Call] = field(default_factory=list)
    players_per_team: int | None = None
    settings: dict[str, Value] = field(default_factory=dict)  # "tickSkip", "ppo.policyLR", ...
    unread: list[str] = field(default_factory=list)
    classes: dict[str, str] = field(default_factory=dict)  # class name -> "file:line" in the project
    # Class name -> its code without comments, whitespace collapsed (to recognise known versions).
    class_code: dict[str, str] = field(default_factory=dict)


_IDENT = r"[A-Za-z_][A-Za-z0-9_:]*"


def strip_comments(text: str) -> str:
    """The code without comments (kept line for line, so line numbers stay right)."""

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    def keep_strings(match: re.Match[str]) -> str:
        return match.group(0) if match.group(0).startswith('"') else blank(match)

    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', keep_strings, text, flags=re.S)


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _matching(text: str, start: int, open_: str, close: str) -> int:
    """The position just after the bracket closing the one at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_:
            depth += 1
        elif text[i] == close:
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError(f"no closing {close!r}")


def _split_args(inner: str) -> list[str]:
    """Top-level comma-separated parts of an argument list."""
    return [part.strip() for _, part in _parts_with_offsets(inner)]


class Reader:
    """Evaluates the C++ expressions of one file, with the local variables and config values set so far."""

    def __init__(self, text: str, config_names: set[str]) -> None:
        self.text = text
        self.variables: dict[str, Value] = {}
        self.config_names = config_names
        self.settings: dict[str, Value] = {}

    def value(self, expr: str, line: int) -> Value:
        """`expr` as a value; a `new X(...)` is a Call; raises ValueError when it is not plain code."""
        expr = expr.strip()
        call = re.fullmatch(rf"(?:new\s+)?({_IDENT})\s*(<[^()]*>)?\s*\((.*)\)", expr, flags=re.S)
        if (
            call
            and not re.fullmatch(r"[\d.]+f?", call.group(1))
            and call.group(1) not in ("RS_MIN", "RS_MAX")
        ):
            name = call.group(1).split("::")[-1] + (call.group(2) or "")
            args = tuple(self.value(arg, line) for arg in _split_args(call.group(3)))
            return Call(name, args, line, " ".join(expr.split()))
        if expr.startswith("{") and expr.endswith("}") and "new " in expr:
            return [self.value(item, line) for item in _split_args(expr[1:-1])]
        if re.fullmatch(rf"new\s+{_IDENT}", expr):
            name = expr.split()[-1].split("::")[-1]
            return Call(name, (), line, expr)
        python = re.sub(r"(?<=\d)'(?=\d)", "", expr)  # 50'000
        python = re.sub(r"(\d(?:\.\d*)?(?:e[+-]?\d+)?)[fF]\b", r"\1", python)
        python = re.sub(r"(\d)[uU]?[lL]{0,2}\b", r"\1", python)
        python = python.replace("{", "[").replace("}", "]").replace("&&", " and ").replace("||", " or ")
        python = re.sub(r"!(?!=)", " not ", python)
        python = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", python))
        python = re.sub(r"\(\s*(?:float|double|int|int64_t)\s*\)", "", python)
        python = re.sub(rf"({_IDENT})", lambda m: m.group(1).replace("::", "__ENUM__"), python)
        try:
            tree = ast.parse(python, mode="eval")
        except SyntaxError as err:
            raise ValueError(f"not plain code: {expr}") from err
        return self._eval(tree.body, expr)

    def _eval(self, node: ast.AST, expr: str) -> Value:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.List):
            return [self._eval(item, expr) for item in node.elts]
        if isinstance(node, ast.UnaryOp):
            inner = self._eval(node.operand, expr)
            if isinstance(node.op, ast.USub):
                return -inner
            if isinstance(node.op, ast.UAdd):
                return +inner
            if isinstance(node.op, ast.Not):
                return not inner
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left, expr), self._eval(node.right, expr)
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
            if not all(isinstance(v, (int, float)) for v in (a, b)):
                raise ValueError(f"not plain arithmetic: {expr}")
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ValueError(f"a division by zero: {expr}")
                both_int = isinstance(a, int) and isinstance(b, int)
                return int(a / b) if both_int else a / b
            if type(node.op) in ops:
                return ops[type(node.op)](a, b)
        if isinstance(node, ast.Name):
            if "__ENUM__" in node.id:
                return node.id.split("__ENUM__")[-1]  # ModelActivationType::RELU -> "RELU"
            if node.id in self.variables:
                return self.variables[node.id]
        if isinstance(node, ast.Attribute):
            path = _dotted(node)
            if path and path[0] in self.config_names and ".".join(path[1:]) in self.settings:
                return self.settings[".".join(path[1:])]
        raise ValueError(f"not plain code: {expr}")


def _dotted(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else [*head, node.attr]
    return None


def _calls_in(reader: Reader, text: str, start: int, stop: int) -> list[Call]:
    """Every top-level `new X(...)` between `start` and `stop`."""
    found: list[Call] = []
    pattern = re.compile(rf"\bnew\s+{_IDENT}\s*(<[^()]*>)?\s*\(")
    pos = start
    for match in pattern.finditer(text, start, stop):
        if match.start() < pos:
            continue
        end = _matching(text, match.end() - 1, "(", ")")
        found.append(reader.value(text[match.start() : end], _line(text, match.start())))
        pos = end
    return found


def _entries(reader: Reader, text: str, source: Source) -> None:
    """The reward lists: `std::vector<WeightedReward> name = { { new X(...), weight }, ... }`,
    `.rewards = {...}`,
    and `name.push_back({ new X(...), weight })` / `emplace_back(new X(...), weight)` on such a list."""
    names = set(re.findall(r"vector\s*<\s*(?:RLGC::)?WeightedReward\s*>\s*(\w+)", text))
    blocks = [
        *re.finditer(r"vector\s*<\s*(?:RLGC::)?WeightedReward\s*>\s*\w+\s*=\s*(\{)", text),
        *re.finditer(r"\.rewards\s*=\s*(\{)", text),
    ]
    spans: list[tuple[int, str]] = []
    for match in blocks:
        start = match.start(1)
        end = _matching(text, start, "{", "}")
        spans += [
            (start + 1 + offset, part) for offset, part in _parts_with_offsets(text[start + 1 : end - 1])
        ]
    for match in re.finditer(r"\b(\w+)\s*\.\s*(?:push_back|emplace_back)\s*\(", text):
        if match.group(1) in names:
            end = _matching(text, match.end() - 1, "(", ")")
            spans.append((match.end(), text[match.end() : end - 1]))
    for pos, part in sorted(spans):
        line = _line(text, pos + len(part) - len(part.lstrip()))
        inner = part.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1]
        elif inner.startswith("WeightedReward(") and inner.endswith(")"):
            inner = inner[len("WeightedReward(") : -1]
        args = _split_args(inner)
        if len(args) != 2:
            source.unread.append(f"line {line}: reward entry {' '.join(part.split())}")
            continue
        try:
            reward = reader.value(args[0], line)
        except ValueError:
            source.unread.append(f"line {line}: reward {' '.join(args[0].split())}")
            continue
        if not isinstance(reward, Call):
            source.unread.append(f"line {line}: reward entry {' '.join(part.split())}")
            continue
        try:
            weight = reader.value(args[1], line)
        except ValueError:
            weight = None
            source.unread.append(f"line {line}: the weight '{args[1]}' of {reward.name}")
        source.rewards.append(WeightedEntry(reward, weight, line))


def _parts_with_offsets(inner: str) -> list[tuple[int, str]]:
    """`_split_args` keeping where each part starts in `inner`."""
    out, depth, start = [], 0, 0
    for i, char in enumerate(inner):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            out.append((start, inner[start:i]))
            start = i + 1
    if inner[start:].strip():
        out.append((start, inner[start:]))
    return [(offset, part) for offset, part in out if part.strip()]


def _assignments(reader: Reader, text: str, config_names: set[str], source: Source) -> None:
    """Local variables and config values, in the order the file sets them."""
    types = rf"int|float|double|bool|auto|int64_t|uint64_t|size_t|{_IDENT}"
    decl = rf"(?:(?:const|constexpr|static)\s+)*(?:{types})\s+"
    pattern = re.compile(rf"(?:^|[;{{}}\n])\s*(?:{decl})?({_IDENT}(?:\.\w+)*)\s*=\s*([^;=][^;]*);", re.M)
    for match in pattern.finditer(text):
        target, expr = match.group(1), match.group(2)
        line = _line(text, match.start(1))
        head, _, rest = target.partition(".")
        if "new " in expr or "->" in target or "[" in target:
            continue
        if head in config_names and rest:
            try:
                reader.settings[rest] = reader.value(expr, line)
            except ValueError:
                source.unread.append(f"line {line}: {target} = {' '.join(expr.split())}")
        elif not rest and "::" not in target:
            with contextlib.suppress(ValueError):  # a variable the settings never read
                reader.variables[target] = reader.value(expr, line)


def read(path: Path) -> Source:
    """Reads a GigaLearn project: `path` is its main .cpp, or a folder searched for the .cpp that sets a
    `LearnerConfig`. Custom classes (rewards, observations, ...) defined in the folder are listed by name."""
    folder = path if path.is_dir() else path.parent
    every = sorted(p for p in folder.rglob("*") if p.suffix in (".cpp", ".h", ".hpp", ".cc"))
    files = [p for p in every if _ours(p, folder)]
    if path.is_dir():
        mains = [
            p
            for p in files
            if p.suffix in (".cpp", ".cc") and "LearnerConfig" in p.read_text(errors="replace")
        ]
        if not mains:
            raise FileNotFoundError(f"no .cpp file in {folder} sets up a GigaLearn LearnerConfig")
        main = max(mains, key=lambda p: p.read_text(errors="replace").count("WeightedReward"))
    else:
        main = path
    try:
        return _read_main(main, every, files, folder)
    except ValueError as err:  # brackets that never close: not code this reader can follow
        raise ValueError(f"{main}: {err}") from None


def _read_main(main: Path, every: list[Path], files: list[Path], folder: Path) -> Source:
    text = strip_comments(main.read_text(encoding="utf-8", errors="replace"))
    source = Source(main)
    config_names = set(re.findall(r"\bLearnerConfig\s+(\w+)", text)) or {"cfg"}
    reader = Reader(text, config_names)
    _assignments(reader, text, config_names, source)
    source.settings = dict(reader.settings)
    _entries(reader, text, source)
    for attr, slot in (("obsBuilder", "obs"), ("actionParser", "action"), ("stateSetter", "setter")):
        match = re.search(rf"\.{attr}\s*=\s*", text)
        if not match:
            continue
        end = text.index(";", match.end())
        try:
            setattr(source, slot, reader.value(text[match.end() : end], _line(text, match.start())))
        except ValueError:
            source.unread.append(
                f"line {_line(text, match.start())}: {attr} = {text[match.end() : end].strip()}"
            )
    match = re.search(r"TerminalCondition\s*\*\s*>\s*\w+\s*=\s*\{|\.terminalConditions\s*=\s*\{", text)
    if match:
        end = _matching(text, match.end() - 1, "{", "}")
        source.terminals = _calls_in(reader, text, match.end(), end)
    players = re.search(r"\bint\s+playersPerTeam\s*=\s*(\d+)\s*;", text)
    if players:
        source.players_per_team = int(players.group(1))
    # Rewards and observations people add often go into the project's copy of GigaLearn itself
    # (RLGymCPP/Rewards/CommonRewards.h), so classes are looked for there too.
    for file in (p for p in every if _ours(p, folder, keep_gigalearn=True)):
        body = strip_comments(file.read_text(encoding="utf-8", errors="replace"))
        for match in re.finditer(
            r"\b(?:class|struct)\s+(\w+)\s*(?:final\s*)?:\s*public\s+(?:RLGC::)?(\w+)\s*\{", body
        ):
            name = match.group(1)
            if name in source.classes and file not in files:
                continue  # the project's own copy wins over the framework's
            source.classes[name] = f"{file.relative_to(folder).as_posix()}:{_line(body, match.start())}"
            try:
                code = body[match.start() : _matching(body, match.end() - 1, "{", "}")]
            except ValueError:  # preprocessor branches and the like: listed, its code not known
                code = ""
            source.class_code[name] = " ".join(code.split())
    return source


def _ours(path: Path, root: Path, keep_gigalearn: bool = False) -> bool:
    """Files of the project itself, not GigaLearn's own copy (unless `keep_gigalearn`), RocketSim, LibTorch or
    build output."""
    skip = {
        *(() if keep_gigalearn else ("GigaLearnCPP",)),
        "RocketSim",
        "RLBotCPP",
        "libtorch",
        "build",
        "out",
        ".git",
        "pybind11",
        "thread_pool",
    }
    return not any(part in skip for part in path.relative_to(root).parts[:-1])
