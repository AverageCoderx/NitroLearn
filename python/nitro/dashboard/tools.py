"""The Tools of a bot's page: commands started for a bot from the window (`nitro eval`, `skills`, `ab`,
`surgery`, `check-settings`, `export`, `pack`), each as its own process with its output in the bot's
console, and what the page shows when one ends. Also what the bot's page shows of its progress: whether a
game of it was recorded, whether its RLBot bot was made, and its charts over the newest part of its training.

`Tools` is a part of `nitro.dashboard.api.Api`. Nothing here loads PyTorch: a command that needs it runs as
a child process and its result is read from what it printed."""

from __future__ import annotations

import dataclasses
import functools
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nitro import pack
from nitro.config.loader import ConfigError, load_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import bots, charts, edits, livery, resources
from nitro.dashboard.jobs import CONSOLE, Job, Jobs
from nitro.runs import RunFolder

# The tools that need the graphics card, or the bot standing still: one at a time in the window, never beside
# the bot's own training. Each with how the page names one that runs, and why training must stop for it.
HEAVY = {
    "eval": ("a test against older saves", "a match beside training would slow both"),
    "skills": ("a skill check", "a check beside training would slow both"),
    "ab": ("a comparison of two settings", "it trains bots of its own on the same graphics card"),
    "surgery": ("a change of network size", "so the wider bot starts from its newest save"),
}
KINDS = (*HEAVY, "settings", "export", "pack")
# What each skill of `nitro skills` is, in plain words: the drill, and what its score counts.
SKILLS = {
    "ball_touch": "Touch a ball at rest",
    "recovery": "Land on its wheels",
    "boost_pickup": "Collect boost",
    "shot": "Score in an empty net",
    "save": "Save a shot",
    "aerial_touch": "Touch a ball in the air",
}
MEASURES = {
    "time_to_touch": "seconds until it touches",
    "time_to_land": "seconds until it lands",
    "boost_collected": "boost collected in 20 seconds",
    "goal_rate": "share of tries scored",
    "save_rate": "share of shots kept out",
    "time_to_first_touch": "seconds until the first touch",
}
SPANS = {"hour": 3600.0, "minutes": 600.0}  # the Overview's ranges besides "all", in seconds of training
_TREND = "touches_per_minute"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# The name RLBot shows goes into bot.toml between quotes as it is.
_SHOWN_NAME = re.compile(r'^[^"\\\x00-\x1f]{1,40}$')
_HEADER = "\n$ nitro "  # jobs.spawn writes one before each command's output


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """`nitro.dashboard.api._safe`, looked up when called: that module loads this one."""

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from nitro.dashboard.api import _safe as safe

        return safe(method)(*args, **kwargs)

    return wrapped


@dataclass
class _Plan:
    """A tool's command, worked out from the page's options but not started."""

    args: list[str]  # what follows `nitro`
    blocked: str = ""  # why it cannot start now ("" when it can)
    files: dict[Path, str] = field(default_factory=dict)  # written before it starts
    extra: dict[str, Any] = field(default_factory=dict)  # for the page and for reading the result


@dataclass
class _Run:
    """A tool's command started from here."""

    job: Job
    command: str  # as typed in a terminal
    offset: int  # where its output starts in the bot's console file
    extra: dict[str, Any]
    end: int | None = None  # where the console ended when the command was first seen ended
    ended: float = 0.0  # and when that was


class Tools:
    """The part of `Api` behind a bot's Tools, sectors and chart ranges. It uses what `Api` holds (the
    project and runs folders, the started commands, the library) and its helpers named below."""

    _project: Path
    _runs: Path
    _jobs: Jobs
    _library: resources.Library

    if TYPE_CHECKING:

        def _folder(self, name: str) -> RunFolder: ...
        def _busy(self, name: str) -> str: ...
        def _check_new_name(self, name: str) -> None: ...

    # ---------- what the bot's page shows ----------
    @_safe
    def bot_tools(self, name: str) -> dict[str, Any]:
        """What a bot's page shows besides `bot`: whether a game of it was recorded and its RLBot bot made
        (the page's sectors), its saves and network sizes (the tools' choices), the tools running or done,
        and how its ball touches changed over the last hour of training."""
        folder = self._folder(name)
        heavy = self._heavy_run()
        return {
            "recorded": _recorded(folder),
            "rlbot": self._rlbot_bot(name),
            "saves": [int(p.name) for p in reversed(folder.complete_checkpoints())],
            "network": _network(folder),
            "jobs": {kind: self._brief(name, kind) for kind in KINDS if (name, kind) in self._tool_runs()},
            "heavy": None
            if heavy is None
            else {"bot": heavy[0], "kind": heavy[1], "what": HEAVY[heavy[1]][0]},
            "trend": _trend(bots.metrics(folder)),
            "liveries": [list(pair) for pair in livery.PALETTE],
        }

    @_safe
    def bot_charts(self, name: str, span: str = "all", ids: list[str] | None = None) -> dict[str, Any]:
        """The Overview's charts: the folders, every chart the bot's metrics hold, and the points of the
        charts `ids` (all when None) over all of the training (`span` "all") or its newest part ("hour", or
        "minutes": the last ten). Time is training time, worked out from each row's steps and speed, so a
        pause does not count. `seconds` is how much training a range covers (less than asked for early on)."""
        folder = self._folder(name)
        if span not in SPANS:
            return {"span": "all", **charts.view(folder, 0, ids)}
        m = bots.metrics(folder)
        first, seconds = _since(m, SPANS[span])
        return {"span": span, "seconds": seconds, **charts.view(folder, first, ids, m=m)}

    @_safe
    def check_livery(self, number: Any, main: Any, contrast: Any) -> dict[str, Any]:
        """A race number and two colours as `set_livery` would keep them (for the preview while choosing), or
        why it would refuse them."""
        try:
            return {"livery": livery.checked(number, main, contrast)}
        except livery.LiveryError as err:
            return {"error": str(err)}

    @_safe
    def tool_choices(self, name: str) -> dict[str, Any]:
        """What the tools' forms offer: the other bots with a save (to play against), every settings file
        (each bot's own, the presets, my-settings, the library; `broken` marks one with a mistake, which only
        the check takes), the practice tasks, and the comparisons already finished."""
        self._folder(name)
        shown = ("id", "title", "group", "about", "broken")
        return {
            "bots": bots.start_choices(self._runs, name),
            "files": [{key: file[key] for key in shown} for file in self._files()],
            "tasks": _tasks(),
            "comparisons": self._comparisons(),
        }

    # ---------- starting a tool, and how it goes ----------
    @_safe
    def tool_command(self, name: str, kind: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """The command a tool would run with the form's `options`, as one would type it, starting nothing;
        `blocked` says why it cannot start now. An option it cannot take comes back as the error."""
        plan = self._plan(name, kind, dict(options or {}))
        return {"command": self._typed(plan.args), "blocked": plan.blocked, **plan.extra}

    @_safe
    def tool_start(self, name: str, kind: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Starts a tool's command for the bot; its output goes to the bot's console."""
        folder = self._folder(name)
        plan = self._plan(name, kind, dict(options or {}))
        if plan.blocked:
            return {"error": plan.blocked}
        for file, text in plan.files.items():
            file.parent.mkdir(parents=True, exist_ok=True)
            edits.save(file, text)
        if kind == "pack":
            Path(plan.extra["folder"]).mkdir(exist_ok=True)
        try:
            offset = (folder.root / CONSOLE).stat().st_size
        except OSError:
            offset = 0
        job = self._jobs.start(name, kind, plan.args, folder.root)
        command = self._typed(plan.args)
        self._tool_runs()[(name, kind)] = _Run(job, command, offset, plan.extra)
        return {"started": True, "command": command, **plan.extra}

    @_safe
    def tool_state(self, name: str, kind: str) -> dict[str, Any]:
        """How the bot's newest run of a tool is doing: "none", "running", "done", "stopped" or "failed"; its
        command, how long it has run, its last lines, and once it ended its result in plain words (or, failed,
        why)."""
        self._folder(name)
        run = self._tool_runs().get((name, kind))
        if run is None:
            return {"state": "none"}
        state = _state(run.job)  # before its output is read: an ended command has printed everything
        text = _output(run, state != "running")
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith("$ nitro ")]
        out: dict[str, Any] = {
            "state": state,
            "kind": kind,
            "command": run.command,
            "seconds": (run.ended or time.time()) - run.job.started,
            "tail": "\n".join(lines[-8:]),
            **run.extra,
        }
        if kind == "ab":
            out["progress"] = read_progress(text)
        if state == "failed":
            out["why"] = "\n".join(lines[-8:]) or "it stopped; the Console tab says why"
        if state == "done" or (kind in ("settings", "export") and state == "failed"):
            out["result"] = _result(kind, text, run.extra)
        return out

    @_safe
    def tool_stop(self, name: str, kind: str) -> dict[str, Any]:
        """Ends a tool's command that is running. A comparison carries on where it was when started again."""
        run = self._tool_runs().get((name, kind))
        if run is None or run.job.done:
            return {"error": "it is not running"}
        run.job.stopped = True
        run.job.process.terminate()
        return {"stopped": True}

    def _tool_busy(self, name: str) -> str:
        """Why the bot cannot train or try rewards now: one of its heavy tools is running."""
        for kind in HEAVY:
            run = self._tool_runs().get((name, kind))
            if run is not None and not run.job.done:
                return f"{name} is busy with {HEAVY[kind][0]}: wait until it is done, or stop it in Tools"
        return ""

    def _tool_runs(self) -> dict[tuple[str, str], _Run]:
        """The tools' commands started from this window, newest by bot and tool (made at first use: `Api`
        sets up its own state)."""
        runs: dict[tuple[str, str], _Run] = self.__dict__.setdefault("_tools_started", {})
        return runs

    def _heavy_run(self) -> tuple[str, str] | None:
        """The heavy tool running now, of any bot: (bot, kind)."""
        for (bot, kind), run in self._tool_runs().items():
            if kind in HEAVY and not run.job.done:
                return bot, kind
        return None

    def _brief(self, name: str, kind: str) -> dict[str, Any]:
        run = self._tool_runs()[(name, kind)]
        return {"state": _state(run.job), "seconds": time.time() - run.job.started}

    def _blocked(self, name: str, kind: str) -> str:
        """Why a heavy tool cannot start for the bot now: the bot trains or is about to, or another heavy tool
        runs."""
        if bots.running(self._folder(name)) is not None:
            return f"{name} is training: stop it first ({HEAVY[kind][1]})"
        if busy := self._busy(name):
            return busy
        if (heavy := self._heavy_run()) is not None:
            what = HEAVY[heavy[1]][0]
            return (
                f"{heavy[0]} is busy with {what}: wait until it is done, or stop it (one heavy job at a time)"
            )
        return ""

    # ---------- each tool's command ----------
    def _plan(self, name: str, kind: str, options: dict[str, Any]) -> _Plan:
        folder = self._folder(name)
        plans: dict[str, Callable[[RunFolder, dict[str, Any]], _Plan]] = {
            "eval": self._plan_eval,
            "skills": self._plan_skills,
            "ab": self._plan_ab,
            "surgery": self._plan_surgery,
            "settings": self._plan_settings,
            "export": self._plan_export,
            "pack": self._plan_pack,
        }
        if kind not in plans:
            raise ValueError(f"there is no tool called {kind}")
        plan = plans[kind](folder, options)
        if kind in HEAVY and not plan.blocked:
            plan.blocked = self._blocked(name, kind)
        return plan

    def _plan_eval(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro eval`: the newest save against an earlier one of the bot, or against another bot's save.
        Both sides play each colour from the same starts, unless they play as the RLBot bot does."""
        name = folder.root.name
        saves = _saves(folder)
        if not saves:
            raise ValueError(f"{name} has no save yet to play")
        other, at = str(options.get("opponent") or ""), _whole(options.get("save"), "the opponent's save")
        if other and other != name:
            theirs = _saves(self._folder(other))
            if not theirs:
                raise ValueError(f"{other} has no save yet to play against")
            if at and at not in theirs:
                raise ValueError(f"{other} has no save at {at:,} steps")
            against = f"{other}'s save at {bots._short(at or theirs[-1])} steps"
        else:
            other = ""
            if len(saves) < 2:
                raise ValueError(
                    f"{name} has only one save, so there is nothing older to play against: train on to its "
                    "next save, or pick another bot"
                )
            if at and (at not in saves or at == saves[-1]):
                raise ValueError(f"pick one of {name}'s earlier saves to play against")
            against = f"its save at {bots._short(at or saves[0])} steps"
        points = _within(options.get("points", 512), 8, 100_000, "the points to play")
        args = ["eval", name, "--runs", str(self._runs), "--games", str(points)]
        if other:
            args += ["--vs", other]
        if at:
            args += ["--vs-at", str(at)]
        args += ["--as-bot"] if options.get("as_bot") else ["--both-ways", "--log"]
        first = f"{name}'s newest save ({bots._short(saves[-1])} steps)"
        return _Plan(args, extra={"first": first, "second": against})

    def _plan_skills(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro skills`: a save's scores on the practice skills."""
        name = folder.root.name
        saves = _saves(folder)
        if not saves:
            raise ValueError(f"{name} has no save yet to score")
        at = _whole(options.get("save"), "the save")
        if at and at not in saves:
            raise ValueError(f"{name} has no save at {at:,} steps")
        games = _within(options.get("situations", 512), 2, 100_000, "the situations a skill")
        args = ["skills", name, "--runs", str(self._runs), "--games", str(games)]
        if at:
            args += ["--at", str(at)]
        if options.get("as_bot"):
            args += ["--temperature", "0"]
        return _Plan(args, extra={"save": at or saves[-1]})

    def _plan_ab(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro ab`: two settings files compared on the same seeds, on a practice task or by speed. A bot's
        own settings go in as a copy that reads the same from the comparison's folder."""
        files: dict[Path, str] = {}
        labels, paths = [], []
        for side in ("a", "b"):
            label, path, text = self._settings_file(str(options.get(side) or ""))
            if text is None:
                load_toml(path, NitroConfig)  # a file with a mistake says so here, not minutes into the run
            if text is not None:
                files[path] = text
            labels.append(label)
            paths.append(path)
        if paths[0] == paths[1]:
            raise ValueError("pick two different settings files to compare")
        what = str(options.get("on") or "")
        tasks = [task["name"] for task in _tasks()]
        if what != "speed" and what not in tasks:
            raise ValueError("pick what to compare them on: a practice task, or training speed")
        title = str(options.get("name") or "").strip() or f"{labels[0]}-vs-{labels[1]}-{what}"
        if not _NAME.match(title):
            raise ValueError("the comparison's name uses letters, digits, '-', '_' and '.'")
        args = ["ab", str(paths[0]), str(paths[1])]
        if what == "speed":
            repeats = _within(options.get("repeats", 6), 2, 50, "the runs a setting")
            args += ["--speed", "--repeats", str(repeats)]
            training = [f.root.name for f in bots.bot_folders(self._runs) if bots.running(f) is not None]
            blocked = (
                f"A speed comparison needs a quiet PC: stop {', '.join(training)} first" if training else ""
            )
        else:
            seeds = _within(options.get("seeds", 10), 4, 50, "the seeds a setting")
            args += ["--task", what, "--seeds", str(seeds)]
            if options.get("with_rewards"):
                args += ["--with-rewards"]
            blocked = ""
        folder_ab = self._project / "ab"
        args += ["--folder", str(folder_ab), "--name", title]
        extra = {"a": labels[0], "b": labels[1], "name": title, "folder": str(folder_ab / title)}
        return _Plan(args, blocked, files, extra)

    def _plan_surgery(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro surgery`: a new bot from the newest save with every hidden layer some times as wide."""
        name = folder.root.name
        network = _network(folder)
        if network is None:
            raise ValueError(f"{name}'s settings cannot be read: fix them first")
        if network["problem"]:
            raise ValueError(network["problem"])
        factor = _within(options.get("widen", 2), 2, 8, "how many times wider")
        made = str(options.get("new_name") or "").strip() or f"{name}-x{factor}"
        self._check_new_name(made)
        args = ["surgery", name, "--widen", str(factor), "--as", made, "--runs", str(self._runs)]
        return _Plan(args, extra={"made": made})

    def _plan_settings(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro check-settings`: a settings file's mistakes, its networks' size and the notes on it."""
        _, path, text = self._settings_file(str(options.get("file") or f"bot:{folder.root.name}"))
        return _Plan(["check-settings", str(path)], files={} if text is None else {path: text})

    def _plan_export(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro export`: the newest save as an RLBot bot folder, under the name RLBot shows for it."""
        name = folder.root.name
        if not folder.complete_checkpoints():
            raise ValueError(f"{name} has no save yet to make a bot of")
        shown = str(options.get("bot_name") or "").strip()
        if shown and not _SHOWN_NAME.match(shown):
            raise ValueError("the name RLBot shows is up to 40 characters, without quotes or backslashes")
        args = ["export", name, "--runs", str(self._runs)]
        if shown and shown != name:
            args += ["--bot-name", shown]
        making = self._jobs.get(name, "export")
        blocked = "Its RLBot bot is being made now" if making is not None and not making.done else ""
        return _Plan(args, blocked, extra={"folder": str(self._project / "bots" / name)})

    def _plan_pack(self, folder: RunFolder, options: dict[str, Any]) -> _Plan:
        """`nitro pack`: the bot in one file in the packs folder, for another computer."""
        name = folder.root.name
        out = self._project / "packs" / f"{name}-{time.strftime('%Y-%m-%d-%H%M%S')}{pack.SUFFIX}"
        args = ["pack", name, "--runs", str(self._runs), "--to", str(out)]
        if options.get("all_saves"):
            args += ["--all-saves"]
        # The arena files (game files) only when asked for.
        args += ["--include-meshes"] if options.get("meshes") else ["--no-meshes"]
        training = bots.running(folder) is not None
        blocked = f"{name} is training: stop it first, so the pack holds its newest save" if training else ""
        return _Plan(args, blocked, extra={"file": str(out), "folder": str(out.parent)})

    def _settings_file(self, ident: str) -> tuple[str, Path, str | None]:
        """A settings file the page chose, by its id in `tool_choices`: a short label, its path, and for a
        bot's own settings ("bot:<name>") the text to write there first. A bot's file names its folders
        relative to where the bot was made, so a command reads a copy with them written in full."""
        if not ident:
            raise ValueError("pick a settings file")
        found = next((file for file in self._files() if file["id"] == ident), None)
        if found is None:
            raise ValueError(f"there is no settings file called {ident}")
        if not ident.startswith("bot:"):
            return found["path"].stem, found["path"], None
        folder = self._folder(ident[len("bot:") :])
        try:
            text = _standalone(folder)
        except ConfigError:  # a file with a mistake goes as it is: the command says what is wrong with it
            text = folder.settings.read_text(encoding="utf-8-sig")
        return folder.root.name, found["path"], text

    def _files(self) -> list[dict[str, Any]]:
        """Every settings file a tool can be pointed at: each bot's own (under "bot:<name>", read as a copy
        kept in the runs folder), then the presets, my-settings and library files. One that
        does not read is `broken`."""
        files: list[dict[str, Any]] = []
        listed: set[Path] = set()

        def add(ident: str, title: str, group: str, path: Path, about: str = "") -> None:
            listed.add(path.resolve())
            entry = {"id": ident, "title": title, "group": group, "path": path}
            files.append({**entry, "about": about or "has a mistake", "broken": not about})

        for folder in sorted(bots.bot_folders(self._runs), key=lambda f: f.root.name.lower()):
            name = folder.root.name
            try:
                about = resources.describe_settings(folder.load_settings())
            except (ConfigError, OSError, ValueError, KeyError):
                about = ""
            copy = self._runs / ".dashboard" / "compare" / f"{name}.toml"
            add(f"bot:{name}", f"{name}'s settings", "bot", copy, about)
        presets = self._project / "configs" / "presets"
        for source in resources.settings_sources(self._project, self._library):
            ident = source["id"]
            path = Path(ident[len("file:") :]) if ident.startswith("file:") else presets / f"{ident}.toml"
            add(ident, source["title"], source["group"], path, source["about"])
        # The files those lists leave out because they do not read.
        mine = self._project / "my-settings"
        folders = {"preset": presets, "my-settings": mine}
        others = [
            (group, path) for group, folder in folders.items() for path in sorted(folder.glob("*.toml"))
        ]
        others += [
            ("library", Path(entry["path"]))
            for entry in self._library.entries()
            if entry["kind"] == "settings" and entry["problem"]
        ]
        for group, path in others:
            if path.resolve() not in listed:
                add("file:" + str(path.resolve()), path.stem.replace("_", " "), group, path)
        return files

    def _typed(self, args: list[str]) -> str:
        """`nitro <args>` as one would type it in NitroLearn's folder: without `--runs` when it is the usual
        runs folder, and with files inside the project named from there."""
        shown: list[str] = []
        usual = (self._project / "runs").resolve()
        skip = False
        for i, arg in enumerate(args):
            if skip:
                skip = False
            elif arg == "--runs" and Path(args[i + 1]).resolve() == usual:
                skip = True
            else:
                shown.append(self._short(arg))
        return "nitro " + " ".join(f'"{arg}"' if " " in arg else arg for arg in shown)

    def _short(self, arg: str) -> str:
        path = Path(arg)
        if not path.is_absolute():
            return arg
        try:
            # Forward slashes on every OS: the command reads the same, and Windows takes them too.
            return path.resolve().relative_to(self._project.resolve()).as_posix()
        except ValueError:
            return arg

    def _rlbot_bot(self, name: str) -> dict[str, Any] | None:
        """The bot's RLBot bot folder as `nitro export` made it (bots/<name> in the project): the name RLBot
        shows, the steps of the save it holds, and whether its program is there. None when not made."""
        folder = self._project / "bots" / name
        try:
            text = (folder / "bot.toml").read_text(encoding="utf-8")
            made = (folder / "policy.nwt").stat().st_mtime
        except OSError:
            return None
        shown = re.search(r'^name = "(.*)"$', text, re.MULTILINE)
        steps = re.search(r"([\d,]+) agent steps", text)
        programs = re.findall(r'^run_command\w* = "(?:\./)?(\S+)', text, re.MULTILINE)
        return {
            "folder": str(folder),
            "name": shown.group(1) if shown else name,
            "steps": int(steps.group(1).replace(",", "")) if steps else None,
            "program": any((folder / program).is_file() for program in programs),
            "at": made,
        }

    def _comparisons(self) -> list[dict[str, Any]]:
        """The comparisons with a result in the project's ab folder, newest first."""
        found = []
        for result in (self._project / "ab").glob("*/result.txt"):
            try:
                verdict = read_verdict(result.read_text(encoding="utf-8"))
                at = result.stat().st_mtime
            except OSError:
                continue
            if verdict:
                found.append({"name": result.parent.name, "at": at, "folder": str(result.parent), **verdict})
        return sorted(found, key=lambda entry: -entry["at"])[:20]


# ---------- reading what a command printed ----------
def _state(job: Job) -> str:
    if not job.done:
        return "running"
    return "stopped" if job.stopped else "done" if job.code == 0 else "failed"


def _output(run: _Run, ended: bool) -> str:
    """What the bot's console holds from the run's start: to its end while it runs, and once it has `ended`
    to where the console ended then. Another command of the bot (its trainer) may print there meanwhile, so
    the readers below pick the run's own lines out by what they say."""
    try:
        with run.job.console.open("rb") as handle:
            handle.seek(run.offset)
            data = handle.read(2_000_000 if run.end is None else run.end - run.offset)
    except OSError:
        return ""
    if ended and run.end is None:
        run.end, run.ended = run.offset + len(data), time.time()
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return text.partition(_HEADER)[2].partition("\n")[2]


def _result(kind: str, text: str, extra: dict[str, Any]) -> dict[str, Any] | None:
    if kind == "eval":
        return read_eval(text, extra["first"], extra["second"])
    if kind == "skills":
        return read_skills(text)
    if kind == "ab":
        try:
            text = (Path(extra["folder"]) / "result.txt").read_text(encoding="utf-8")
        except OSError:
            return None
        verdict = read_verdict(text, extra["a"], extra["b"])
        return None if verdict is None else {**verdict, "text": text}
    if kind == "surgery":
        made = re.search(r"^Made '([^']+)' from ", text, re.MULTILINE)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return {"made": made.group(1), "lines": lines} if made else None
    if kind == "settings":
        return read_check(text)
    if kind == "export":
        return read_export(text, extra["folder"])
    if kind == "pack":
        notes = [line[len("Note: ") :] for line in text.splitlines() if line.startswith("Note: ")]
        return {"file": extra["file"], "folder": extra["folder"], "notes": notes}
    return None


def read_eval(text: str, first: str, second: str) -> dict[str, Any] | None:
    """What `nitro eval` printed, in plain words: who took more of the goals, by how much and how sure.
    `first` and `second` name the two sides (the command's Blue and Orange; with --both-ways each plays
    both colours). None when the output holds no result."""
    reading = re.search(r"^\s*Reading: (.+)$", text, re.MULTILINE)
    both = re.search(
        r"As Blue it scored (\d+) and let in (\d+); as Orange it\s+scored (\d+) and let in (\d+); "
        r"(\d+) ended without a goal",
        text,
    )
    one = re.search(r"Blue scored (\d+), Orange (\d+); (\d+) ended without a goal", text)
    if reading is None or (both is None and one is None):
        return None
    if both is not None:
        scored, let_in = int(both[1]) + int(both[3]), int(both[2]) + int(both[4])
        no_goal = int(both[5])
    else:
        assert one is not None
        scored, let_in, no_goal = int(one[1]), int(one[2]), int(one[3])
    points = scored + let_in + no_goal
    share = re.search(r"(\d+)% \(95% interval (\d+)% to (\d+)%\)", text)
    said = reading[1].strip()
    out: dict[str, Any] = {
        "first": first,
        "second": second,
        "scored": scored,
        "let_in": let_in,
        "no_goal": no_goal,
        "points": points,
        "reading": said,
        "share": None,
    }
    if share is None or said.startswith("Nobody scored"):
        out["winner"] = "nobody"
        out["headline"] = "Nobody scored, so there is nothing to compare."
        out["lines"] = [
            f"All {points:,} points ran out of time. Bots early in training often cannot score yet: try "
            "again after more training, or with more points.",
        ]
        return out
    low, high = int(share[2]), int(share[3])
    out |= {"share": int(share[1]), "low": low, "high": high}
    counted = (
        f"{_capital(first)} scored {scored:,} goals and let in {let_in:,}: {share[1]}% of the goals, over "
        f"{points:,} points ({no_goal:,} ended with no goal)."
    )
    sure = f"How sure: allowing for luck, its real share is between {low}% and {high}% (95 times in 100)."
    if said.startswith(("The first is clearly better", "Blue is clearly better")):
        out["winner"] = "first"
        out["headline"] = f"{_capital(first)} is clearly better than {second}."
        sure += " All of that range is above half, so this is not luck."
    elif said.startswith(("The second is clearly better", "Orange is clearly better")):
        out["winner"] = "second"
        out["headline"] = f"{_capital(second)} is clearly better than {first}."
        sure += " All of that range is below half, so this is not luck."
    else:
        out["winner"] = "neither"
        out["headline"] = "No clear difference between the two."
        sure += " That range includes an even split, so this many points cannot tell them apart; more "
        sure += "points narrow it."
    out["lines"] = [counted, sure]
    return out


_SKILL_LINE = re.compile(
    r"^(?P<task>\w+)\s+(?P<metric>\w+) (?P<value>[^\s()]+?)(?P<unit>s?) \(bar (?P<bar>[^\s;]+?)s?; "
    r"(?P<side>above|below|under|over) the bar: [^)]*\)",
    re.MULTILINE,
)


def read_skills(text: str) -> dict[str, Any] | None:
    """The table `nitro skills` printed: each skill with its score, its bar and whether the bot has it."""
    rows = []
    for match in _SKILL_LINE.finditer(text):
        task, metric = match["task"], match["metric"]
        rows.append(
            {
                "task": task,
                "skill": SKILLS.get(task, task.replace("_", " ")),
                "measure": MEASURES.get(metric, metric.replace("_", " ")),
                "value": match["value"],
                "bar": match["bar"],
                "unit": match["unit"],
                "has": match["side"] in ("above", "under"),
                "lower_is_better": metric.startswith("time_"),
            }
        )
    if not rows:
        return None
    has = sum(1 for row in rows if row["has"])
    note = re.search(r"The bar is .+$", text, re.MULTILINE)
    who = re.search(r"^Scoring (.+?) on \d+ skills?, (\d+) situations each", text, re.MULTILINE)
    return {
        "rows": rows,
        "headline": f"It has {has} of the {len(rows)} skills." if len(rows) > 1 else rows[0]["skill"],
        "who": who.group(1) if who else "",
        "situations": int(who.group(2)) if who else None,
        "note": note.group(0) if note else "",
    }


_VERDICTS = {
    "B IS BETTER": ("b", "{b} did clearly better than {a}."),
    "B IS WORSE": ("a", "{b} did clearly worse than {a}."),
    "BORDERLINE": ("", "Borderline: a hint that they differ, not a result."),
    "NO MEASURABLE DIFFERENCE": ("", "No measurable difference between the two."),
}


def read_verdict(text: str, a: str = "A", b: str = "B") -> dict[str, Any] | None:
    """The verdict of a comparison's result (what `nitro ab` printed last and kept in result.txt)."""
    found = re.search(r"^Verdict: ([A-Z ]+?) - (.+)$", text, re.MULTILINE)
    if found is None or found[1] not in _VERDICTS:
        early = re.search(r"^Too few finished runs.*$", text, re.MULTILINE)
        return {"verdict": "", "better": "", "headline": early[0], "meaning": ""} if early else None
    better, words = _VERDICTS[found[1]]
    headline = words.format(a=a, b=b)
    return {"verdict": found[1].capitalize(), "better": better, "headline": headline, "meaning": found[2]}


def read_progress(text: str) -> dict[str, int] | None:
    """How far a comparison is, from its newest "(3 of 20); about 85 min left" line."""
    found = re.findall(r"\((\d+) of (\d+)\); about (\d+) min left", text)
    if not found:
        return None
    done, of, minutes = (int(n) for n in found[-1])
    return {"done": done, "of": of, "minutes_left": minutes}


def read_check(text: str) -> dict[str, Any]:
    """What `nitro check-settings` printed: fine or not, its mistakes, the networks' size and the notes."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    fine = any(line.endswith(": no problems found.") for line in lines)
    count = re.search(r"^(\d+) problems? in .*:$", text, re.MULTILINE)
    problems = []
    if count:
        for line in text[count.end() :].lstrip("\n").splitlines():
            if not line.startswith("  "):
                break
            problems.append(line.strip())
    if fine:
        headline = "No problems found."
    elif count:
        headline = f"{count[1]} {'mistake' if count[1] == '1' else 'mistakes'} to fix."
    else:
        headline = "The check stopped before it was done."
    return {
        "fine": fine,
        "headline": headline,
        "networks": next((line for line in lines if line.startswith("Networks: ")), ""),
        "notes": [line[len("Note: ") :] for line in lines if line.startswith("Note: ")],
        "problems": problems,
    }


def read_export(text: str, folder: str) -> dict[str, Any]:
    """What `nitro export` printed: where the bot folder is, and whether its program was built."""
    missing = re.search(r"^The bot program is missing: (.+)$", text, re.MULTILINE)
    written = "Exported to " in text
    return {
        "folder": folder,
        "written": written,
        "program": written and missing is None,
        "missing": missing.group(1) if missing else "",
    }


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


# ---------- facts read from the bot's folder ----------
def _saves(folder: RunFolder) -> list[int]:
    return [int(p.name) for p in folder.complete_checkpoints()]


def _whole(value: Any, what: str) -> int:
    """A whole number from the page (0 for none), or ValueError naming `what`."""
    if value in (None, "", False):
        return 0
    try:
        number = float(str(value).replace(",", "").replace("_", ""))
    except ValueError:
        raise ValueError(f"{what} is a whole number") from None
    if number != int(number) or number < 0:
        raise ValueError(f"{what} is a whole number")
    return int(number)


def _within(value: Any, low: int, high: int, what: str) -> int:
    number = _whole(value, what)
    if not low <= number <= high:
        raise ValueError(f"{what}: between {low:,} and {high:,}")
    return number


def _recorded(folder: RunFolder) -> dict[str, Any] | None:
    """When a game of the bot was last recorded (`nitro watch` writes its page and its recording into the
    bot's folder); None when never."""
    times = []
    for file in (folder.root / "watch.html", folder.root / "watch.nlrec"):
        try:
            times.append(file.stat().st_mtime)
        except OSError:
            continue
    return {"at": max(times)} if times else None


def _network(folder: RunFolder) -> dict[str, Any] | None:
    """The bot's hidden layers as its settings have them, and why `nitro surgery` cannot widen them ("" when
    it can). None when its settings cannot be read."""
    try:
        model = folder.load_settings().model
    except (ConfigError, OSError, ValueError, KeyError):
        return None
    name = folder.root.name
    problem = ""
    if not folder.complete_checkpoints():
        problem = f"{name} has no save yet: widening starts from a save"
    elif model.uses_attention:
        problem = f"{name} has attention networks: only plain networks can be widened"
    elif model.residual or model.value_heads_apart:
        problem = f"{name} has residual blocks or value heads: only plain networks can be widened"
    elif not (model.shared or model.policy or model.value):
        problem = f"{name} has no hidden layers to widen"
    return {
        "shared": list(model.shared),
        "policy": list(model.policy),
        "value": list(model.value),
        "problem": problem,
    }


def _standalone(folder: RunFolder) -> str:
    """The bot's settings file as it reads from any folder: the folders it names (relative to where the bot
    was made) written in full."""
    config = folder.load_settings()
    changes: dict[str, Any] = {}
    for section in dataclasses.fields(config):
        part = getattr(config, section.name)
        if not dataclasses.is_dataclass(part):
            continue
        for item in dataclasses.fields(part):
            value = getattr(part, item.name)
            if isinstance(value, Path):
                changes[f"{section.name}.{item.name}"] = value.as_posix()
            elif isinstance(value, tuple) and value and all(isinstance(p, Path) for p in value):
                changes[f"{section.name}.{item.name}"] = [p.as_posix() for p in value]
    return edits.written(folder.settings.read_text(encoding="utf-8-sig"), changes)


def _tasks() -> list[dict[str, str]]:
    """The practice tasks `nitro ab` compares on, each with what it is (its file's first words)."""
    found = []
    for file in sorted((Path(__file__).parents[1] / "tasks").glob("*.toml")):
        first = file.read_text(encoding="utf-8").split("\n", 1)[0]
        about = re.match(r"^# Practice task: ([^.]+)\.", first)
        found.append({"name": file.stem, "about": about.group(1) if about else file.stem.replace("_", " ")})
    return found


def _since(m: bots.Metrics, seconds: float) -> tuple[int, float]:
    """The first row of the newest `seconds` of training in `m`, and the seconds of training from it to the
    newest row: each row's steps over its speed, added up from the newest back."""
    steps, speed = m.columns.get("agent_steps"), m.columns.get("steps_per_second")
    if steps is None or speed is None or not m.count:
        return 0, 0.0
    covered, row = 0.0, m.count - 1
    while row > 0 and covered < seconds:
        gained, rate = steps[row] - steps[row - 1], speed[row]
        if gained > 0 and rate > 0:  # an empty cell (nan) is neither
            covered += gained / rate
        row -= 1
    return row, covered


def _trend(m: bots.Metrics) -> dict[str, float] | None:
    """Ball touches a minute over the newest ten minutes of training and over the ten minutes an hour of
    training before; None until the metrics hold that much."""
    if _TREND not in m.columns:
        return None
    last, covered = _since(m, 600.0)
    hour, back = _since(m, 3600.0)
    before, further = _since(m, 4200.0)
    if covered < 600.0 or further < 4200.0 or before >= hour:
        return None
    now = _mean(m, _TREND, last, m.count)
    then = _mean(m, _TREND, before, hour)
    return None if now is None or then is None else {"now": now, "before": then, "minutes": back / 60}


def _mean(m: bots.Metrics, column: str, first: int, end: int) -> float | None:
    values = [v for row in range(first, end) if (v := m.value(column, row)) is not None]
    return sum(values) / len(values) if values else None
