"""Reads an RLGym-PPO (or rlgym-learn) project's Python without running it: the Learner's
settings, and what the environment is made of (`rlgym_sim.make(...)` or `rlgym.make(...)`: observation
builder, action parser, rewards and weights, terminal conditions, state setter, tick skip, team size).
Plain code only (literals, arithmetic, local variables, RLGym's common values); anything else is listed as not
read, never guessed."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nitro.rlgym_reference import common_values

Value = Any


@dataclass(frozen=True)
class PyCall:
    """`Name(args, kw=...)`: the called name (last part), its arguments read, its line, the module it was
    imported from ("" when not seen), and the code as written."""

    name: str
    args: tuple[Value, ...]
    kwargs: dict[str, Value]
    line: int
    module: str
    text: str

    def arg(self, index: int, name: str, default: Value) -> Value:
        """The argument given by position `index` or keyword `name`, else `default`."""
        if name in self.kwargs:
            return self.kwargs[name]
        return self.args[index] if len(self.args) > index else default


class Unread:
    """A value the reader could not work out; `text` is the code."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"<not read: {self.text}>"


@dataclass
class RlgymSource:
    file: Path
    learner: dict[str, Value] = field(default_factory=dict)  # Learner(...)'s keyword settings
    learner_line: int = 0
    make: PyCall | None = None  # rlgym_sim.make / rlgym.make / RLGym(...)
    obs: Value = None
    action: Value = None
    reward: Value = None
    terminals: list[Value] = field(default_factory=list)
    setter: Value = None
    classes: dict[str, str] = field(default_factory=dict)  # the project's own classes -> "file:line"
    unread: list[str] = field(default_factory=list)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}
_FUNCTIONS: dict[str, Any] = {
    "int": int,
    "float": float,
    "round": round,
    "abs": abs,
    "bool": bool,
    "min": min,
    "max": max,
}
_ARRAYS = {"asarray", "array"}  # np.asarray([...]) -> a list


class Reader:
    def __init__(self, text: str) -> None:
        self.text = text
        self.env: dict[str, Value] = {}
        self.modules: dict[str, str] = {}  # imported name -> its module

    def code(self, node: ast.AST) -> str:
        return " ".join((ast.get_source_segment(self.text, node) or "?").split())

    def value(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.value(item) for item in node.elts]
        if isinstance(node, ast.UnaryOp):
            inner = self.value(node.operand)
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                if isinstance(node.op, ast.USub):
                    return -inner
                if isinstance(node.op, ast.UAdd):
                    return inner
            if isinstance(node.op, ast.Not) and isinstance(inner, bool):
                return not inner
            return Unread(self.code(node))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = self.value(node.left), self.value(node.right)
            try:
                if _number(a) and _number(b):
                    return _arith(node.op, a, b)
                if isinstance(a, list) and _number(b):  # np.asarray([...]) / 2300 and the like
                    return (
                        [_arith(node.op, x, b) for x in a]
                        if all(_number(x) for x in a)
                        else Unread(self.code(node))
                    )
                if _number(a) and isinstance(b, list) and all(_number(x) for x in b):
                    return [_arith(node.op, a, x) for x in b]
            except (ArithmeticError, ValueError):  # 1 / 0, an overflow, a huge power: not read
                pass
            return Unread(self.code(node))
        if isinstance(node, ast.Name):
            return self.env.get(node.id, Unread(node.id))
        if isinstance(node, ast.Attribute):
            path = _dotted(node)
            if path and path[0] in ("math", "np", "numpy") and path[-1] == "pi":
                return math.pi
            if path and path[-1] in vars(common_values) and not path[-1].startswith("_"):
                return getattr(common_values, path[-1])
            return Unread(self.code(node))
        if isinstance(node, ast.Call):
            return self.call(node)
        return Unread(self.code(node))

    def call(self, node: ast.Call) -> Value:
        path = _dotted(node.func) or []
        name = path[-1] if path else ""
        args = [self.value(arg) for arg in node.args]
        kwargs = {kw.arg: self.value(kw.value) for kw in node.keywords if kw.arg}
        if name in _FUNCTIONS and len(path) == 1 and all(_number(a) for a in args) and args:
            return _FUNCTIONS[name](*args)
        if name in _ARRAYS and path[0] in ("np", "numpy") and len(args) == 1:
            return args[0]
        module = self.modules.get(path[0], "") if path else ""
        if len(path) > 1:
            module = ".".join(path[:-1]) if not module else f"{module}.{'.'.join(path[1:-1])}"
        return PyCall(name, tuple(args), kwargs, node.lineno, module.rstrip("."), self.code(node))

    def visit(self, body: list[ast.stmt]) -> None:
        """Walks statements in order, keeping imports and plain assignments; into functions and ifs too."""
        for stmt in body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module:
                for alias in stmt.names:
                    self.modules[alias.asname or alias.name] = stmt.module
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    self.modules[alias.asname or alias.name] = alias.name
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    self.env[target.id] = self.value(stmt.value)
                elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Tuple):
                    for t, v in zip(target.elts, stmt.value.elts, strict=False):
                        if isinstance(t, ast.Name):
                            self.env[t.id] = self.value(v)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
                self.env[stmt.target.id] = self.value(stmt.value)
            for inner in ("body", "orelse", "finalbody"):
                nested = getattr(stmt, inner, None)
                if isinstance(nested, list) and not isinstance(stmt, ast.ClassDef):
                    self.visit(nested)


def _arith(op: ast.operator, a: Value, b: Value) -> Value:
    if isinstance(op, ast.Pow) and (abs(b) > 64 or abs(a) > 2**64):  # no huge whole numbers to work out
        raise OverflowError("too large")
    result = _OPS[type(op)](a, b)
    if not _number(result):  # (-1) ** 0.5 is complex
        raise ValueError("not a plain number")
    return result


def _number(value: Value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dotted(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else [*head, node.attr]
    return None


MAKERS = ("make", "RLGym")


class SourceError(Exception):
    """The project cannot be read; the message says why."""


def _calls(file: Path, wanted: Callable[[list[str]], bool]) -> bool:
    """Whether `file` calls something `wanted` accepts (by its dotted name); unparsable files do not."""
    try:
        tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, RecursionError):
        return False
    return any(isinstance(n, ast.Call) and wanted(_dotted(n.func) or []) for n in ast.walk(tree))


def _makes_learner(path: list[str]) -> bool:
    return bool(path) and path[-1] == "Learner"


def _makes_env(path: list[str]) -> bool:
    return bool(path) and (path[-1] == "RLGym" or (path[-1] == "make" and path[0] in ("rlgym_sim", "rlgym")))


def read(path: Path) -> RlgymSource:
    """Reads the project at `path`: a .py file, or a folder searched for the .py that makes a Learner (or the
    one that makes the environment). Classes defined in the folder are listed by name."""
    folder = path if path.is_dir() else path.parent
    files = sorted(
        p
        for p in folder.rglob("*.py")
        if not any(
            part.startswith(".") or part in ("venv", ".venv", "site-packages")
            for part in p.relative_to(folder).parts
        )
    )
    if path.is_dir():
        chosen = [p for p in files if _calls(p, _makes_learner)] or [
            p for p in files if _calls(p, _makes_env)
        ]
        if not chosen:
            raise FileNotFoundError(
                f"no .py file in {folder} makes an RLGym-PPO Learner or an RLGym environment"
            )
        if len(chosen) > 1:
            names = ", ".join(p.relative_to(folder).as_posix() for p in chosen)
            raise SourceError(
                f"several .py files in {folder} could be the bot's ({names}); point at the one that trains it"
            )
        main = chosen[0]
    else:
        main = path
    text = main.read_text(encoding="utf-8", errors="replace")
    source = RlgymSource(main)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as err:
        raise SourceError(f"{main} is not Python this reader can read ({err})") from None
    reader = Reader(text)
    reader.visit(tree.body)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path_ = _dotted(node.func) or []
        name = path_[-1] if path_ else ""
        if name == "Learner":
            call = reader.call(node)
            assert isinstance(call, PyCall)
            source.learner, source.learner_line = dict(call.kwargs), node.lineno
            source.learner.update({f"#{i}": a for i, a in enumerate(call.args)})
        elif name in MAKERS and (name == "RLGym" or (len(path_) > 1 and path_[0] in ("rlgym_sim", "rlgym"))):
            call = reader.call(node)
            assert isinstance(call, PyCall)
            source.make = call
    make = source.make
    if make is not None:
        source.obs = make.kwargs.get("obs_builder")
        source.action = make.kwargs.get("action_parser")
        source.reward = make.kwargs.get("reward_fn")
        source.setter = make.kwargs.get("state_setter", make.kwargs.get("state_mutator"))
        terminals = make.kwargs.get("terminal_conditions", make.kwargs.get("termination_cond"))
        source.terminals = (
            terminals if isinstance(terminals, list) else [] if terminals is None else [terminals]
        )
        for key, value in make.kwargs.items():
            if isinstance(value, Unread):
                source.unread.append(f"line {make.line}: {key} = {value.text}")
    for key, value in source.learner.items():
        if isinstance(value, Unread) and not key.startswith("#"):  # by position: the env maker
            source.unread.append(f"line {source.learner_line}: Learner {key} = {value.text}")
    for file in files:
        try:
            module = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                source.classes.setdefault(node.name, f"{file.relative_to(folder).as_posix()}:{node.lineno}")
    return source
