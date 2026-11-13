"""The dashboard's Python side: what the window asks for and the actions its buttons take. Every public method
takes and returns plain JSON-able values (the window calls them through pywebview); a problem comes back as
{"error": "..."} in plain English, never as a crash of the window."""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nitro import pack, rebuild, reward_check, rlbot_find, rlbot_install
from nitro.config.loader import ConfigError, load_toml
from nitro.config.rewards import custom_kinds
from nitro.config.schema import NitroConfig, large_networks
from nitro.dashboard import (
    bots,
    checks,
    describe,
    edits,
    guide,
    livery,
    machine,
    replay_tab,
    resources,
    reward_layout,
    rewards,
    starts,
)
from nitro.dashboard.garage import GarageApi, preset_cards
from nitro.dashboard.info import InfoApi
from nitro.dashboard.jobs import CONSOLE, Jobs, read_console
from nitro.dashboard.tools import Tools
from nitro.dashboard.updating import Updating
from nitro.dashboard.watching import Watching
from nitro.doctor import find_rlbot
from nitro.runs import RunFolder

# How long a trainer started again after an Apply may take to be training before it counts as failed.
RESTART_GRACE = 10 * 60

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Names Windows keeps for devices (and drops a trailing dot), so no folder can have them.
_RESERVED = {"con", "prn", "aux", "nul", *(f"{port}{i}" for port in ("com", "lpt") for i in range(1, 10))}
_BUILD = ".build"  # the jobs list's name for a build of the core with one's own Rust rewards
_CONVERT = ".convert"  # the jobs list's name for converting replays only
_RLBOT = ".rlbot"  # the jobs list's name for installing RLBot v5 (`nitro install-rlbot`)
PENDING = "pending.json"  # the waiting changes, in the runs folder's .dashboard
NEW_ACTION_DELAY = 1  # ticks, for new bots
# How a bot gets its first save: one copying replays saves when the copy is ready.
_FIRST_SAVE = (
    "a bot that copies replays saves every pretrain.save_every replays and when the copy is ready; others "
    "15 minutes into training by default (run.checkpoint_minutes)"
)
# Where a bot starts from is chosen up to its first save.
_STARTED = (
    "Read when the bot first started. It has its own trained networks now (its saves), so where it started "
    "from can no longer change; New bot from this one makes a bot that starts from somewhere else."
)
# The new-bot form's steps after the name and game size: (title, what it is about, the settings asked).
WIZARD = (
    (
        "How it plays",
        "How often the bot decides, and what it sees of the game.",
        ("env.tick_skip", "env.observation", "env.obs_team_size"),
    ),
    (
        "Its brain",
        "The networks' kind and sizes. These stay fixed once the bot has saves.",
        (
            "model.encoder",
            "model.shared",
            "model.policy",
            "model.value",
            "model.value_head_layers",
            "model.attention_width",
            "model.attention_blocks",
            "model.attention_heads",
            "model.transformer_layers",
            "model.transformer_width",
            "model.transformer_heads",
            "model.transformer_ff",
        ),
    ),
    (
        "How it learns",
        "Changeable later in the bot's Settings tab.",
        (
            "ppo.policy_lr",
            "ppo.value_lr",
            "ppo.entropy",
            "ppo.discount_half_life",
            "ppo.steps_per_iteration",
            "ppo.epochs",
            "ppo.minibatches",
        ),
    ),
)
# Asked by the new-bot form only of a starting file that sets them to something.
_WIZARD_WHEN_SET = frozenset({"model.value_head_layers"})


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """Turns any exception into {"error": ...}, so one bad file never breaks the window; and a number that is
    not finite into None, since the window's JSON reader refuses Infinity and NaN (a settings file can hold
    `weight = inf`)."""

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return _finite(method(*args, **kwargs))
        except ConfigError as err:
            return {"error": str(err)}
        except (OSError, ValueError, KeyError, ArithmeticError) as err:
            return {"error": str(err) or type(err).__name__}

    return wrapped


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _newest_finished(folder: RunFolder) -> dict[str, Any] | None:
    """The newest finished check of rewards kept in the bot's folder, or None."""
    for path in sorted((folder.root / reward_check.CHECKS).glob("*.json"), reverse=True):
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("finished"):
            return {**data, "file": path.name}
    return None


class Api(InfoApi, GarageApi, Tools, Watching, Updating):
    def __init__(self, project: Path, runs: Path) -> None:
        self._project = project
        self._runs = runs
        self._jobs = Jobs(project)
        self._pending: dict[str, dict[str, Any]] = {}  # bot -> {setting path: value as the file will hold it}
        self._reward_pending: dict[
            str, list[dict[str, Any]]
        ] = {}  # bot -> its rewards as Apply will write them
        self._start_pending: dict[str, list[dict[str, Any]]] = {}  # bot -> its game starts, likewise
        # Kept in the runs folder, so closing the window loses no waiting change.
        self._pending_file = runs / ".dashboard" / PENDING
        self._load_pending()
        self._applying: dict[str, dict[str, Any]] = {}  # bot -> what an Apply does now, and since when
        self._stopping: dict[str, float] = {}  # bot -> when Stop and save was pressed
        self._window: Any = None  # set by nitro.dashboard.app, for file dialogs
        self._library = resources.Library(runs, project / "configs" / "reward-sets")
        self._lock = threading.Lock()

    def _load_pending(self) -> None:
        try:
            data = json.loads(self._pending_file.read_text(encoding="utf-8"))
            self._pending = {str(k): dict(v) for k, v in data.get("settings", {}).items()}
            self._reward_pending = {str(k): list(v) for k, v in data.get("rewards", {}).items()}
            self._start_pending = {str(k): list(v) for k, v in data.get("starts", {}).items()}
        except (OSError, ValueError, AttributeError, TypeError):
            pass

    def _keep_pending(self) -> None:
        data = {
            "settings": self._pending,
            "rewards": self._reward_pending,
            "starts": self._start_pending,
        }
        try:
            self._pending_file.parent.mkdir(parents=True, exist_ok=True)
            edits.save(self._pending_file, json.dumps(data, indent=1, ensure_ascii=False))
        except OSError:
            pass  # kept in memory; only a closed window loses them then

    # ---------- the bots ----------
    @_safe
    def home(self) -> dict[str, Any]:
        return {
            "bots": [self._summary(folder) for folder in bots.bot_folders(self._runs)],
            "runs": str(self._runs),
            "presets": self._presets(),
            "prefs": self._prefs(),
            "rlbot_ready": self._rlbot_ready(),
            "machine": machine.facts(),
        }

    def _rlbot_ready(self) -> bool:
        """Whether RLBot v5 is there: the program chosen in the dashboard, else RLBot v5 where its installer
        puts it (as Check this PC finds it). An older RLBot does not run NitroLearn's bots."""
        return find_rlbot([self._runs], rlbot_install.local_app_data()) is not None

    @_safe
    def bot(self, name: str) -> dict[str, Any]:
        folder = self._folder(name)
        out = self._summary(folder)
        start = out["start"]
        out["transfer"] = {
            "choices": [] if start["done"] else bots.start_choices(self._runs, name),
            "note": "",
        }
        try:
            text = folder.settings.read_text(encoding="utf-8-sig")
            config = folder.load_settings()
            saved = bool(folder.complete_checkpoints()) and folder.copy_so_far() is None
            out["settings"] = describe.describe(config, text, locked=saved, fixed=_start_locks(start))
            note = json.loads(folder.info.read_text(encoding="utf-8")).get("start_note", "")
            if not start["done"] and config.start_from.source and config.start_from.parts == "policy":
                out["transfer"]["note"] = note
        except ConfigError as err:
            out["settings"] = []
            out["problem"] = str(err)
        except UnicodeDecodeError:
            out["settings"] = []
            out["problem"] = (
                f"{folder.settings} is not saved as UTF-8 (an editor saved it as ANSI). Open it (the button "
                "below), save it again choosing UTF-8, and come back."
            )
        out["pending"] = dict(self._pending.get(name, {}))
        out["reward_changes"] = self._reward_changes(folder, name)
        out["start_changes"] = self._start_changes(folder, name)
        out["notes"] = self._notes(folder)
        out["newest"] = bots.newest(folder)
        out["wandb"] = bots.wandb_url(folder)
        out["report"] = folder.report.is_file()
        out["copy"] = bots.copy_view(folder)
        return out

    @_safe
    def console_text(self, name: str, after: int = 0) -> dict[str, Any]:
        """The bot's console after byte `after`. Not called `console`: pywebview's WebView2 window takes a
        call of that name as its own console forwarding and prints it, so it never came here."""
        text, end = read_console(self._folder(name).root, int(after))
        return {"text": text, "next": end}

    # ---------- changes, applied only on Apply ----------
    @_safe
    def stage(self, name: str, path: str, value: Any) -> dict[str, Any]:
        """Keeps a change to setting `path` for Apply (nothing is written yet); refuses a value the setting
        cannot take, saying why."""
        folder = self._folder(name)
        item = self._item(folder, path)
        if item is None:
            return {"error": f"there is no setting {path}"}
        if item["fixed"]:
            return {"error": item["why_fixed"]}
        try:
            converted = edits.convert(item, value)
        except edits.EditError as err:
            return {"error": f"{item['label']}: {err}"}
        with self._lock:
            pending = dict(self._pending.get(name, {}))
            pending[path] = converted
            text = self._written(folder, name, pending)
            found = edits.problems(text, folder.settings, self._settings_dir(folder))
            mine = [p for p in found if path in p] or found
            if mine:
                return {"error": "; ".join(mine), "pending": self._pending.get(name, {})}
            self._pending[name] = pending
            self._keep_pending()
        return {"pending": pending}

    @_safe
    def unstage(self, name: str, path: str) -> dict[str, Any]:
        with self._lock:
            self._pending.get(name, {}).pop(path, None)
            self._keep_pending()
            return {"pending": dict(self._pending.get(name, {}))}

    @_safe
    def discard(self, name: str) -> dict[str, Any]:
        with self._lock:
            self._pending.pop(name, None)
            self._reward_pending.pop(name, None)
            self._start_pending.pop(name, None)
            self._keep_pending()
        return {"pending": {}}

    # ---------- rewards ----------
    @_safe
    def rewards(self, name: str, seconds: float = 60.0) -> dict[str, Any]:
        """The bot's rewards as the editor shows them (with any waiting changes), the kinds of reward there
        are, and each reward's share of all reward paid over the last `seconds` of training."""
        folder = self._folder(name)
        out = self._reward_state(folder, name)
        out["catalog"] = rewards.catalog()
        out["help"] = rewards.help_texts()
        out["guide"] = guide.under("reward.")
        out["shares"] = rewards.shares(folder, float(seconds))
        out["layout"] = reward_layout.read(folder.root, self._reward_keys(folder, name))
        return out

    @_safe
    def stage_rewards(self, name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Keeps the edited rewards (each {"from": its place in the file or None, "view": ...}) for Apply.
        They are kept even with a problem (a reward is filled in one box at a time); Apply waits until
        fixed."""
        folder = self._folder(name)
        cleaned = [_entry(e) for e in entries]
        before, _ = rewards.read(folder.settings.read_text(encoding="utf-8-sig"))
        if copies := rewards.added_copies(before, [e["view"] for e in cleaned]):
            listed = ", ".join(f"'{describe.label(copy)}'" for copy in copies)
            verb = "is" if len(copies) == 1 else "are"
            raise ValueError(
                f"{listed} {verb} already in this bot's rewards: a reward can be added only once"
            )
        with self._lock:
            if not rewards.changes(before, cleaned) and len(cleaned) == len(before):
                self._reward_pending.pop(name, None)
            else:
                self._reward_pending[name] = cleaned
            self._keep_pending()
        return self._reward_state(folder, name)

    @_safe
    def reward_shares(self, name: str, seconds: float = 60.0) -> dict[str, Any]:
        return rewards.shares(self._folder(name), float(seconds))

    @_safe
    def try_rewards(self, name: str, agent_steps: int = reward_check.DEFAULT_STEPS) -> dict[str, Any]:
        """Plays the bot's newest save, learning nothing, with the rewards as the editor shows them (waiting
        changes in), and keeps what each would pay (`nitro reward-check`)."""
        folder = self._folder(name)
        if bots.running(folder) is not None:
            return {"error": f"{name} is training: stop it first (a check beside training would slow both)"}
        if not folder.complete_checkpoints():
            return {"error": f"{name} has no save yet: start it once ({_FIRST_SAVE})"}
        if busy := self._busy(name):
            return {"error": busy}
        checks = folder.root / reward_check.CHECKS
        checks.mkdir(parents=True, exist_ok=True)
        tried = checks / f"{time.strftime('%Y%m%d-%H%M%S')}.rewards.toml"
        tried.write_text(self._written(folder, name), encoding="utf-8")
        steps = str(max(1, int(agent_steps)))
        args = ["reward-check", name, "--runs", str(self._runs), "--rewards", str(tried), "--steps", steps]
        self._jobs.start(name, "check", args, folder.root)
        return {"started": True}

    @_safe
    def reward_check(self, name: str) -> dict[str, Any]:
        """The newest check of rewards on the frozen bot: playing (with its progress), its result, or why the
        last one failed."""
        folder = self._folder(name)
        job = self._jobs.get(name, "check")
        playing = job is not None and not job.done
        newest = reward_check.newest(folder)
        out: dict[str, Any] = {"playing": playing, "result": None, "progress": None, "error": ""}
        if newest is not None and newest.get("finished"):
            out["result"] = newest
        elif newest is not None and playing:
            out["progress"] = newest.get("done", 0) / max(newest.get("rounds", 1), 1)
        elif newest is not None:
            out["result"] = _newest_finished(folder)  # a check stopped part-way leaves the one before
        if playing and job is not None:
            out["seconds"] = time.time() - job.started
            if out["progress"]:  # the time left, from how long the rounds so far took (setting up included)
                out["seconds_left"] = out["seconds"] * (1 - out["progress"]) / out["progress"]
        if job is not None and job.done and job.stopped:
            out["stopped"] = True
        elif job is not None and job.done and job.code != 0:
            text, _ = read_console(folder.root, 0)
            last = text.rsplit("$ nitro reward-check", 1)[-1].strip().splitlines()
            out["error"] = last[-1] if last else "the check stopped; the Console tab says why"
        return out

    def _reward_state(self, folder: RunFolder, name: str) -> dict[str, Any]:
        text = folder.settings.read_text(encoding="utf-8-sig")
        before, in_file = rewards.read(text)
        entries = self._reward_pending.get(name) or [{"from": i, "view": v} for i, v in enumerate(before)]
        found = edits.issues(self._written(folder, name), folder.settings, self._settings_dir(folder))
        per_reward: dict[int, list[str]] = {}
        general = []
        for issue in found:
            match = re.match(r"^rewards\[(\d+)\]\.?(.*)$", issue.path)
            if match:
                where = describe.label(match.group(2).replace(".", " ")) if match.group(2) else ""
                per_reward.setdefault(int(match.group(1)), []).append(
                    f"{where}: {issue.message}" if where else issue.message
                )
            else:
                general.append(f"{issue.path}: {issue.message}" if issue.path else issue.message)
        labels = rewards.labels([e["view"] for e in entries])
        return {
            "rewards": [
                {**entry, "label": labels[i], "problems": per_reward.get(i, [])}
                for i, entry in enumerate(entries)
            ],
            "original": before,
            "in_file": in_file,
            "changes": rewards.changes(before, entries) if name in self._reward_pending else [],
            "general": general,
            "pending": name in self._reward_pending,
            "duplicates": rewards.duplicates([e["view"] for e in entries]),
        }

    def _reward_keys(self, folder: RunFolder, name: str) -> list[str]:
        """The keys (reward_layout.keys) of the bot's rewards in its settings file and in its waiting
        changes."""
        before, _ = rewards.read(folder.settings.read_text(encoding="utf-8-sig"))
        waiting = self._reward_pending.get(name) or []
        return reward_layout.keys(view["name"] for view in before) + reward_layout.keys(
            str(entry["view"].get("name", "")) for entry in waiting
        )

    @_safe
    def set_reward_layout(self, name: str, layout: Any) -> dict[str, Any]:
        """Keeps how the Rewards tab lays out the bot's rewards (folders and order, reward_layout): a view
        only, kept in the bot's folder, so it needs no Apply and leaves the settings file as it is."""
        folder = self._folder(name)
        return {"layout": reward_layout.write(folder.root, layout, self._reward_keys(folder, name))}

    def _reward_changes(self, folder: RunFolder, name: str) -> list[str]:
        entries = self._reward_pending.get(name)
        if entries is None:
            return []
        before, _ = rewards.read(folder.settings.read_text(encoding="utf-8-sig"))
        return rewards.changes(before, entries) or ["rewards: reordered"]

    def _written(self, folder: RunFolder, name: str, pending: dict[str, Any] | None = None) -> str:
        """The bot's settings file as Apply would write it: the waiting setting and reward changes in."""
        text = folder.settings.read_text(encoding="utf-8-sig")
        changes = self._pending.get(name, {}) if pending is None else pending
        if changes:
            text = edits.written(text, changes)
        if name in self._reward_pending:
            text = rewards.written(text, self._reward_pending[name])
        if name in self._start_pending:
            text = starts.written(text, self._start_pending[name])
        return text

    # ---------- game starts (state setters) ----------
    @_safe
    def starts(self, name: str) -> dict[str, Any]:
        """The bot's game starts as the editor shows them (waiting changes in) and the kinds there are."""
        out = self._start_state(self._folder(name), name)
        out["catalog"] = starts.catalog()
        out["guide"] = guide.under("start.")
        return out

    @_safe
    def stage_starts(self, name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Keeps the edited starts (each {"from": its place in the file or None, "view": ...}) for Apply."""
        folder = self._folder(name)
        cleaned = [_entry(e) for e in entries]
        before, _ = starts.read(folder.settings.read_text(encoding="utf-8-sig"))
        with self._lock:
            if not starts.changes(before, cleaned) and len(cleaned) == len(before):
                self._start_pending.pop(name, None)
            else:
                self._start_pending[name] = cleaned
            self._keep_pending()
        return self._start_state(folder, name)

    def _start_state(self, folder: RunFolder, name: str) -> dict[str, Any]:
        before, in_file = starts.read(folder.settings.read_text(encoding="utf-8-sig"))
        entries = self._start_pending.get(name) or [{"from": i, "view": v} for i, v in enumerate(before)]
        found = edits.issues(self._written(folder, name), folder.settings, self._settings_dir(folder))
        per_start: dict[int, list[str]] = {}
        general = []
        for issue in found:
            match = re.match(r"^state_setters\[(\d+)\]\.?(.*)$", issue.path)
            if match:
                where = describe.label(match.group(2).replace(".", " ")) if match.group(2) else ""
                per_start.setdefault(int(match.group(1)), []).append(
                    f"{where}: {issue.message}" if where else issue.message
                )
            elif issue.path.startswith("state_setters"):
                general.append(issue.message)
        shares = starts.shares([e["view"] for e in entries])
        return {
            "starts": [
                {**entry, "share": shares[i], "problems": per_start.get(i, [])}
                for i, entry in enumerate(entries)
            ],
            "original": before,
            "in_file": in_file,
            "changes": starts.changes(before, entries) if name in self._start_pending else [],
            "general": general,
            "pending": name in self._start_pending,
        }

    def _start_changes(self, folder: RunFolder, name: str) -> list[str]:
        entries = self._start_pending.get(name)
        if entries is None:
            return []
        before, _ = starts.read(folder.settings.read_text(encoding="utf-8-sig"))
        return starts.changes(before, entries) or ["starts: reordered"]

    @_safe
    def apply(self, name: str) -> dict[str, Any]:
        """Writes the waiting changes into the bot's settings file. A training bot saves, stops and carries on
        with them; a stopped bot just keeps them for its next start. New rewards for a bot not started yet
        that takes everything from another bot switch it to that bot's policy only (`switched` says so)."""
        folder = self._folder(name)
        with self._lock:
            pending = self._pending.get(name, {})
            reward_changes = self._reward_changes(folder, name) + self._start_changes(folder, name)
            if not pending and not reward_changes:
                return {"applied": 0}
            switch, switched = self._new_rewards_switch(folder, name, pending)
            changes = {**pending, **switch}
            info = json.loads(folder.info.read_text(encoding="utf-8"))
            moved = "start_from" in info and any(path.startswith("start_from.") for path in changes)
            if moved:
                # A transfer kept in run.json (nitro train --from) moves into the file with
                # the change, so the file alone says where the bot starts from.
                given = {f"start_from.{key}": value for key, value in info.pop("start_from").items()}
                changes = {**given, **changes}
            text = self._written(folder, name, changes)
            found = edits.problems(text, folder.settings, self._settings_dir(folder))
            if found:
                return {"error": "fix these first: " + "; ".join(found)}
            edits.save(folder.settings, text)
            if switched:
                info["start_note"] = switched
            if moved or switched:
                edits.save(folder.info, json.dumps(info))
            count = len(pending) + len(reward_changes)
            self._pending.pop(name, None)
            self._reward_pending.pop(name, None)
            self._start_pending.pop(name, None)
            self._keep_pending()
        restarting = bots.running(folder) is not None
        if restarting:
            # The trainer starts itself again after its save (nitro.cli._run), so closing this window in
            # between cannot leave it stopped; Stop meanwhile cancels that.
            self._applying[name] = {"text": "Saving, then starting again with the changes", "at": time.time()}
            (folder.root / bots.STOP_REQUEST).write_text("apply\n", encoding="utf-8")
        out: dict[str, Any] = {"applied": count, "restarting": restarting}
        if switched:
            out["switched"] = switched
        return out

    def _new_rewards_switch(
        self, folder: RunFolder, name: str, pending: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        """With changed rewards, a bot not started yet that takes everything (or, the matching parts)
        from another bot takes only its policy instead: the other bot's value network learned what its own
        rewards are worth. A "matching" bot whose model.value is not the other's keeps "matching" (the policy
        alone would not fit). Nothing changes when the user set that choice in this same Apply."""
        if not self._reward_changes(folder, name) or "start_from.parts" in pending:
            return {}, ""
        if folder.complete_checkpoints() or bots.running(folder) is not None:
            return {}, ""  # it has its own networks, or took the other bot's already
        try:
            config = folder.load_settings(text=self._written(folder, name, pending))
        except ConfigError:
            return {}, ""  # Apply refuses it and says why
        start = config.start_from
        if not start.source or start.parts not in ("all", "matching") or config.pretrain.enabled:
            return {}, ""
        who = bots.find_source(start.source, start.at, self._runs)["source"]
        after = " Change this in its Settings tab (Transfer learning) if you like."
        if start.parts == "matching":
            theirs = bots.source_setting(start.source, start.at, self._runs, "model.value")
            if theirs is not None and list(theirs) != list(config.model.value):  # "policy" refuses it
                return {}, ""
        note = (
            f"The rewards changed, so {name} will take only {who}'s policy, with a fresh value network "
            f"({who}'s learned what the old rewards are worth)."
        )
        return {"start_from.parts": "policy"}, note + after

    # ---------- copying replays while training ([replay_mix]) ----------
    @_safe
    def replay_mix(self, name: str) -> dict[str, Any]:
        """The Replays in training tab: where the copy steps' replays come from and how many there are (with
        the waiting changes in), the settings check's advice about them, their explanations, and the run's
        newest copy steps. Its settings themselves are the bot's (`bot`), changed with `stage` and Apply."""
        folder = self._folder(name)
        try:
            config = folder.load_settings(text=self._written(folder, name))
        except ConfigError:  # a waiting change with a problem: what the file says
            config = folder.load_settings()
        return {
            "sources": replay_tab.sources(config),
            "advice": [note for note in config.advice() if "replay_mix" in note],
            "guide": guide.under("replay_mix."),
            "progress": replay_tab.progress(folder),
        }

    @_safe
    def replay_mix_progress(self, name: str) -> dict[str, Any]:
        """Only the run's newest copy steps (redrawn while the tab is open)."""
        return {"progress": replay_tab.progress(self._folder(name))}

    @_safe
    def choose_replay_folder(self) -> dict[str, Any]:
        """Asks for a folder of replays (the replay copy's card)."""
        if self._window is None:
            return {"error": "no window to ask from"}
        import webview

        chosen = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if not chosen:
            return {"cancelled": True}
        return {"folder": str(Path(chosen[0]))}

    @_safe
    def count_replays(self, folders: list[str]) -> dict[str, Any]:
        """How many replays (.replay files) each folder holds, as the copy looks for them (not in folders
        inside it), and in all, a name in two folders counted once."""
        counts, names = [], set()
        default = not folders
        if default:  # none chosen: the game's own folders, as the copy takes them
            from nitro import replays

            folders = [str(f) for f in replays.game_folders(replays.documents())]
        for folder in folders:
            path = Path(folder)
            found = [p.stem for p in path.glob("*.replay")] if path.is_dir() else []
            counts.append({"folder": folder, "replays": len(found), "missing": not path.is_dir()})
            names.update(found)
        return {"folders": counts, "total": len(names), "default": default}

    # ---------- running ----------
    @_safe
    def start(self, name: str) -> dict[str, Any]:
        folder = self._folder(name)
        if (state := bots.running(folder)) is not None:
            self._note_running(name, state)
            return {"error": f"{name} is already training"}
        if busy := self._busy(name):
            return {"error": busy}
        waiting = (self._pending, self._reward_pending, self._start_pending)
        if self._pending.get(name) or any(name in kind for kind in waiting[1:]):
            return {"error": "apply or discard the waiting changes first"}
        self._jobs.start(name, "train", ["resume", name, "--runs", str(self._runs)], folder.root)
        return {"started": True}

    @staticmethod
    def _restart_problem(folder: RunFolder, since: float) -> str:
        """Why the trainer did not start again after an Apply, once that is known: it stopped with a
        problem instead of saving, or it started again and that start ended before training."""
        try:
            outcome = json.loads((folder.root / bots.APPLY_RESTART).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        at = float(outcome.get("at", 0))
        if at < since or bots.running(folder) is not None:
            return ""
        if not outcome.get("restarted"):
            return (
                f"It saved the changes but did not start again: {outcome.get('why', 'see its Console tab')}"
            )
        if time.time() - at > RESTART_GRACE:
            return (
                "It saved and started again with the changes, but that start ended before training (its "
                "Console tab says why); press Start to try again"
            )
        return ""

    def _busy(self, name: str) -> str:
        """Why `name` cannot start now though nothing trains: a start under way (a double click, or a trainer
        still loading), its restart after an Apply, or a check of rewards playing. A trainer seen
        training whose process is only ending (its running.json already gone) does not count."""
        train = self._jobs.get(name, "train")
        if train is not None and not train.done and not train.seen_running:
            return f"{name} is starting already"
        check = self._jobs.get(name, "check")
        if check is not None and not check.done:
            return f"{name} is trying rewards; wait until that is done"
        if tool := self._tool_busy(name):
            return tool
        return f"{name} is starting again with the applied changes" if name in self._applying else ""

    def _note_running(self, name: str, state: dict[str, Any] | None) -> None:
        train = self._jobs.get(name, "train")
        if (
            state
            and train is not None
            and (state.get("pid") == train.process.pid or (train.token and state.get("job") == train.token))
        ):
            train.seen_running = True

    @_safe
    def stop(self, name: str) -> dict[str, Any]:
        folder = self._folder(name)
        if (state := bots.running(folder)) is None:
            return {"error": f"{name} is not training"}
        self._note_running(name, state)
        self._applying.pop(name, None)  # the trainer then saves and stops, without starting again
        (folder.root / bots.STOP_REQUEST).write_text("stop\n", encoding="utf-8")
        self._stopping[name] = time.time()
        return {"stopping": True}

    # ---------- deleting and renaming a bot ----------
    def _held(self, name: str) -> str:
        """Why the bot `name` cannot be deleted or renamed now: it trains, a command of its runs, or a bot
        not started yet takes its networks from it ("" when nothing holds it)."""
        folder = self._folder(name)
        if bots.running(folder) is not None:
            return f"{name} is training: stop it first"
        if self._jobs.active(name) or self._busy(name):
            return f"{name} is busy (a command of its runs): wait until it is done"
        live = self._live
        if live is not None and live.bot == name and live.process.poll() is None:
            return f"{name} is playing live in the viewer: close the viewer first"
        for other in bots.bot_folders(self._runs):
            if other.root.name == name or other.complete_checkpoints():
                continue
            start = bots.summary(other)["start"]
            if start["kind"] == "from" and start["source"] == name:
                return (
                    f"{other.root.name} will start from {name}'s networks and has not started yet: start it "
                    f"first, or switch its transfer learning off in its Settings tab"
                )
        return ""

    @_safe
    def delete_bot(self, name: str, with_rlbot: bool = False) -> dict[str, Any]:
        """Deletes the bot `name`: its folder in the runs folder (settings, saves, notes, recordings) and,
        with `with_rlbot`, its RLBot bot's folder. Nothing else is touched: a linked folder inside it is
        unlinked, never emptied."""
        folder = self._folder(name)
        if held := self._held(name):
            return {"error": held}
        shutil.rmtree(folder.root)
        removed = [str(folder.root)]
        rlbot = self._project / "bots" / name
        if with_rlbot and rlbot.is_dir() and not rlbot.is_symlink():
            shutil.rmtree(rlbot)
            removed.append(str(rlbot))
        with self._lock:
            for kept in (self._pending, self._reward_pending, self._start_pending):
                kept.pop(name, None)
            self._keep_pending()
        return {"deleted": name, "removed": removed}

    @_safe
    def rename_bot(self, name: str, new_name: str) -> dict[str, Any]:
        """Renames the bot `name` (its folder in the runs folder) to `new_name` while it is not training.
        Its RLBot bot keeps the folder and name it was made with."""
        folder = self._folder(name)
        new_name = (new_name or "").strip()
        if new_name == name:
            return {"renamed": name}
        self._check_new_name(new_name)
        if held := self._held(name):
            return {"error": held}
        target = self._runs / new_name
        folder.root.rename(target)
        info_file = RunFolder(target).info
        info = json.loads(info_file.read_text(encoding="utf-8"))
        moved = False
        for key in ("settings_dir", "settings_file", "settings_copied", "collision_meshes"):
            value = info.get(key)
            if isinstance(value, str) and Path(value).is_relative_to(folder.root):
                info[key] = str(target / Path(value).relative_to(folder.root))  # inside the bot's own folder
                moved = True
        if moved:
            edits.save(info_file, json.dumps(info))
        with self._lock:
            for kept in (self._pending, self._reward_pending, self._start_pending):
                if name in kept:
                    kept[new_name] = kept.pop(name)  # type: ignore[assignment]
            self._keep_pending()
        return {"renamed": new_name}

    @_safe
    def new_bot(self, name: str, preset: str, start_from: str = "") -> dict[str, Any]:
        """A new bot from one of the presets (the full new-bot form comes next), trained from
        scratch or from another bot's networks (transfer learning)."""
        self._check_new_name(name)
        presets = {p["name"]: p["path"] for p in self._presets()}
        if preset not in presets:
            return {"error": f"there is no preset called {preset}"}
        args = ["train", presets[preset], "--name", name, "--runs", str(self._runs)]
        if start_from:
            self._folder(start_from)
            args += ["--from", start_from]
        self._jobs.start(name, "train", args, self._runs / name)
        return {"started": True, "name": name}

    @_safe
    def settings_sources(self) -> dict[str, Any]:
        """What a new bot can start from: the presets, my-settings files and settings files in the library."""
        return {"sources": resources.settings_sources(self._project, self._library)}

    @_safe
    def wizard(self, preset: str) -> dict[str, Any]:
        """The new-bot form's questions for a starting settings file (a preset's name or a source's id), each
        with that file's value, its explanation and its choices."""
        file, config, text = self._preset(preset)
        items = _items(describe.describe(config, text, locked=False))
        asked = {p for p in items if p not in _WIZARD_WHEN_SET or items[p]["value"] not in (None, "", [])}
        steps = [
            {"title": title, "about": about, "items": [items[path] for path in paths if path in asked]}
            for title, about, paths in WIZARD
        ]
        return {
            "preset": preset,
            "team_size": config.env.team_size,
            "steps": steps,
            "file": str(file),
            "about": resources.describe_settings(config),
        }

    @_safe
    def check_bot(self, preset: str, values: dict[str, Any]) -> dict[str, Any]:
        """The new-bot form's values checked as Make would check them, making nothing: each step's Next asks,
        so a mistake is shown on its own step."""
        errors = self._bot_changes(preset, values)[3]
        return {"errors": errors} if errors else {}

    @_safe
    def create_bot(self, name: str, preset: str, values: dict[str, Any]) -> dict[str, Any]:
        """A new bot from `preset` with the form's `values` (setting path -> what was typed), not started yet,
        so its rewards can be set up first. Its settings file keeps the preset's comments."""
        self._check_new_name(name)
        file, text, changes, errors, adjusted = self._bot_changes(preset, values)
        if errors:
            return {"errors": errors}
        made = edits.written(text, changes)
        folder = RunFolder(self._runs / name)
        folder.checkpoints.mkdir(parents=True)
        edits.save(folder.settings, made)
        folder.info.write_text(json.dumps({"settings_dir": str(file.resolve().parent)}), encoding="utf-8")
        advice = folder.load_settings().advice()
        return {"made": name, "changes": changes, "adjusted": adjusted, "advice": advice}

    def _bot_changes(
        self, preset: str, values: dict[str, Any]
    ) -> tuple[Path, str, dict[str, Any], dict[str, str], list[str]]:
        """The settings file a new bot starts from, its text, the form's changes to it, the form's mistakes
        (setting path -> why) and what was adjusted to fit."""
        file, config, text = self._preset(preset)
        items = _items(describe.describe(config, text, locked=False))
        changes: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for path, raw in (values or {}).items():
            item = items.get(path)
            if item is None:
                errors[path] = f"there is no setting {path}"
                continue
            try:
                value = edits.convert(item, raw)
            except edits.EditError as err:
                errors[path] = f"{item['label']}: {err}"
                continue
            if value != describe.plain(item["value"]):
                changes[path] = value
        adjusted = []
        tick_skip = changes.get("env.tick_skip")
        if tick_skip is not None and "env.action_delay" not in changes:
            delay = items["env.action_delay"]["value"]
            if delay >= tick_skip:  # the delay must stay below the tick skip; new bots use 1
                changes["env.action_delay"] = min(NEW_ACTION_DELAY, tick_skip - 1)
                adjusted.append(f"Action delay set to {changes['env.action_delay']}, below the tick skip.")
        sizes = {
            key: changes.get(f"model.{key}", getattr(config.model, key))
            for key in ("shared", "policy", "value")
        }
        precision_asked = "run.precision" in (values or {})
        if not precision_asked and config.run.precision == "fp32" and large_networks(**sizes):
            # Big networks: FP16 trained about twice as fast and learned as well.
            changes["run.precision"] = "fp16"
            note = "Precision set to fp16: networks this big train about twice as fast with it"
            norm_ok = config.model.layer_norm and config.model.activation in ("relu", "leaky_relu")
            if norm_ok and "model.fused_norm" not in (values or {}):
                changes["model.fused_norm"] = True
                note += ", with model.fused_norm (11% more)"
            adjusted.append(note + ". Change it in the Settings tab before the first save if you like.")
        encoder = changes.get("model.encoder", config.model.encoder)
        if (
            encoder == "attention"
            and not config.model.attention_fast
            and "model.attention_fast" not in (values or {})
        ):
            # The same answers to rounding, 10% faster.
            changes["model.attention_fast"] = True
            adjusted.append("attention_fast turned on: the same attention network, trained about 10% faster.")
        if not errors:
            found = edits.issues(edits.written(text, changes), file, file.parent)
            errors = {issue.path or "settings": issue.message for issue in found}
        return file, text, changes, errors, adjusted

    @_safe
    def copy_bot(self, name: str, new_name: str, transfer: bool) -> dict[str, Any]:
        """A new bot with this bot's settings, not started yet (so its settings can still change, network
        sizes included). Its [start_from] names this bot and its newest save, switched on with `transfer`
        (it then takes everything: the rewards are the same) and off without, so it can be switched on in
        its Settings tab until its first start."""
        source = self._folder(name)
        self._check_new_name(new_name)
        saves = source.complete_checkpoints()
        if transfer and not saves:
            return {"error": f"{name} has no save yet, so there are no trained networks to start from"}
        values = _start_section(name, int(saves[-1].name) if saves else 0, transfer, "all")
        text = _with_start(source.settings.read_text(encoding="utf-8-sig"), values)
        notes = []
        if transfer and source.load_settings().pretrain.enabled:
            text = edits.written(text, {"pretrain.enabled": False})  # both choose what a bot starts from
            notes.append(f"It starts from {name}, so it does not copy replays first ([pretrain] is off).")
        target = RunFolder(self._runs / new_name)
        target.checkpoints.mkdir(parents=True)
        edits.save(target.settings, text)
        info: dict[str, Any] = {"settings_dir": str(self._settings_dir(source))}
        meshes = json.loads(source.info.read_text(encoding="utf-8")).get("collision_meshes")
        if meshes:  # an unpacked bot's: this computer's meshes
            info["collision_meshes"] = meshes
        target.info.write_text(json.dumps(info), encoding="utf-8")
        return {"made": new_name, "notes": notes}

    @_safe
    def export(self, name: str) -> dict[str, Any]:
        folder = self._folder(name)
        if not folder.complete_checkpoints():
            return {"error": f"{name} has no save yet to make a bot of"}
        self._jobs.start(name, "export", ["export", name, "--runs", str(self._runs)], folder.root)
        return {"started": True, "folder": str(self._project / "bots" / name)}

    @_safe
    def watch(self, name: str, team_size: int = 0, seconds: int = 60) -> dict[str, Any]:
        folder = self._folder(name)
        args = ["watch", name, "--runs", str(self._runs)]
        if team_size:
            args += ["--team-size", str(int(team_size))]
        args += ["--seconds", str(int(seconds))]
        self._jobs.start(name, "watch", args, folder.root)
        return {"started": True}

    @_safe
    def job(self, name: str, kind: str) -> dict[str, Any]:
        """How a command started from here is doing; when it failed, its last lines say why."""
        job = self._jobs.get(name, kind)
        if job is None:
            return {"state": "none"}
        state = "running" if not job.done else ("done" if job.code == 0 or job.stopped else "failed")
        out: dict[str, Any] = {"state": state, "code": job.code}
        if state == "failed":
            text, _ = read_console(job.console.parent, 0)
            lines = [line for line in text.rsplit("\n$ nitro ", 1)[-1].splitlines()[1:] if line.strip()]
            out["why"] = "\n".join(lines[-6:]) or "it stopped; the Console tab says why"
        return out

    @_safe
    def stop_job(self, name: str, kind: str) -> dict[str, Any]:
        """Ends a command started from here: converting replays, trying rewards or recording a game."""
        job = self._jobs.get(name, kind)
        if job is None or job.done:
            return {"error": "it is not running"}
        job.stopped = True
        job.process.terminate()
        return {"stopped": True}

    @_safe
    def open_settings(self, name: str) -> dict[str, Any]:
        """Opens the bot's settings file in the PC's editor (for a mistake the Settings tab cannot show)."""
        _open(self._folder(name).settings)
        return {"opened": True}

    # ---------- opening things ----------
    @_safe
    def open_folder(self, name: str, which: str = "bot") -> dict[str, Any]:
        places = {
            "export": self._project / "bots" / name,
            "packs": self._project / "packs",
            "code": self._project,
            "rewards-guide": self._project / "docs" / "your_own_rewards.md",
        }
        path = places[which] if which in places else self._folder(name).root
        if not path.exists():
            return {"error": f"{path} does not exist yet"}
        _open(path)
        return {"opened": str(path)}

    @_safe
    def pack_bot(self, name: str) -> dict[str, Any]:
        """Packs the bot into one file for another computer (`nitro pack`), in the packs folder."""
        folder = self._folder(name)
        if bots.running(folder) is not None:
            return {"error": f"{name} is training: stop it first, so the pack holds its newest save"}
        out = self._project / "packs" / f"{name}-{time.strftime('%Y-%m-%d-%H%M%S')}{pack.SUFFIX}"
        out.parent.mkdir(exist_ok=True)
        args = ["pack", name, "--runs", str(self._runs), "--to", str(out), "--no-meshes"]
        self._jobs.start(name, "pack", args, folder.root)
        return {"started": True, "file": str(out)}

    @_safe
    def unpack_bot(self, path: str = "", replace: bool = False, new_name: str = "") -> dict[str, Any]:
        """Adds a bot packed on another computer (`nitro unpack`), asking for the file when `path` is
        empty: under `new_name` when given, else under its own name. A bot of that name here is replaced
        only with `replace` (and kept in .replaced); an answer with "exists" offers a free name."""
        if not path:
            if self._window is None:
                return {"error": "no window to ask from"}
            import webview

            chosen = self._window.create_file_dialog(
                webview.FileDialog.OPEN, file_types=("NitroLearn packs (*.nitropack.zip)", "All files (*.*)")
            )
            if not chosen:
                return {"cancelled": True}
            path = chosen[0]
        name = pack.manifest(Path(path))["run"]
        if new_name:
            self._check_new_name(new_name)
            name = new_name
        if (self._runs / name).exists() and not replace:
            free = next(f"{name}-{k}" for k in range(2, 10_000) if not (self._runs / f"{name}-{k}").exists())
            return {"exists": name, "path": path, "free": free}
        done = pack.unpack(Path(path), self._runs, replace=replace, project=self._project, name=new_name)
        with self._lock:  # changes waiting for the bot that was replaced are not this one's
            if self._pending.pop(name, None) is not None:
                self._keep_pending()
        for file in done.rust:
            with contextlib.suppress(OSError, ValueError):  # already in the library
                self._library.add(file)
        return {"unpacked": done.folder.name, "notes": done.notes, "rust": [file.name for file in done.rust]}

    @_safe
    def open_report(self, name: str) -> dict[str, Any]:
        report = self._folder(name).report
        if not report.is_file():
            return {"error": "the progress report appears after the first few minutes of training"}
        webbrowser.open(report.resolve().as_uri())
        return {"opened": True}

    @_safe
    def open_url(self, url: str) -> dict[str, Any]:
        if not url.startswith("https://"):
            return {"error": "only web pages open from here"}
        webbrowser.open(url)
        return {"opened": True}

    @_safe
    def open_rlbot(self) -> dict[str, Any]:
        """Starts the chosen RLBot v5 program, else RLBot v5 where its installer puts it. A chosen program
        that is not RLBot v5's (RLBot v4 does not run NitroLearn's bots) is forgotten."""
        chosen = self._prefs().get("rlbot_program", "")
        if chosen and not rlbot_find.is_v5(chosen):
            self.set_pref("rlbot_program", "")
        program = find_rlbot([self._runs], rlbot_install.local_app_data())
        if program is None:
            return {"need_program": True}
        _open(program)
        return {"opened": str(program)}

    @_safe
    def choose_rlbot(self) -> dict[str, Any]:
        """Asks once where RLBot v5 is (the dashboard remembers it)."""
        if self._window is None:
            return {"error": "no window to ask from"}
        import webview

        chosen = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False)
        if not chosen:
            return {"cancelled": True}
        if not rlbot_find.is_v5(str(chosen[0])):
            which = (
                "RLBot v4, an older RLBot"
                if rlbot_find.is_older(str(chosen[0]))
                else "not RLBot v5's program"
            )
            return {
                "error": f"That is {which}. NitroLearn's bots need RLBot v5: choose its program, "
                f"{rlbot_find.WHERE_V5} (get RLBot v5 from rlbot.org if it is not there)."
            }
        self.set_pref("rlbot_program", str(chosen[0]))
        return self.open_rlbot()  # type: ignore[no-any-return]

    @_safe
    def install_rlbot(self) -> dict[str, Any]:
        """Windows: downloads RLBot v5's official installer (RLBot's own releases on GitHub) and opens it;
        RLBot's own windows take it from there. Follow it with install_rlbot_state."""
        if sys.platform != "win32":
            return {"error": rlbot_install.NOT_WINDOWS}
        job = self._jobs.get(_RLBOT, "rlbot")
        if job is not None and not job.done:
            return {"error": "RLBot's installer is already open"}
        folder = self._runs / ".dashboard" / "rlbot"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / CONSOLE).unlink(missing_ok=True)
        self._jobs.start(_RLBOT, "rlbot", ["install-rlbot", "--runs", str(self._runs)], folder)
        return {"started": True, "from": rlbot_install.OFFICIAL + "installer/"}

    @_safe
    def install_rlbot_state(self, after: int = 0) -> dict[str, Any]:
        """How installing RLBot is going: none, running, done or failed, what it printed after byte `after`
        (`next` is where to read on from), and RLBot's program once it is found."""
        job = self._jobs.get(_RLBOT, "rlbot")
        state = (
            "none" if job is None else "running" if not job.done else "done" if job.code == 0 else "failed"
        )
        text, end = read_console(self._runs / ".dashboard" / "rlbot", int(after))
        program = find_rlbot([self._runs], rlbot_install.local_app_data())
        return {"state": state, "text": text, "next": end, "program": str(program or "")}

    # ---------- the resources library ----------
    @_safe
    def resources(self) -> dict[str, Any]:
        """The added files with what each is, the notes on resources, and the built-in rewards, observations
        and action sets (which take notes too)."""
        config = NitroConfig()
        sections = {s["path"]: s for s in describe.describe(config, "", locked=False)}
        found = {i["path"]: i for s in sections.values() for i in s["items"]}
        built = {str(Path(kind.file).resolve()) for kind in custom_kinds().values()}
        files = self._library.entries()
        for entry in files:
            if entry["kind"].startswith("rust_"):
                entry["built"] = str(Path(entry["path"]).resolve()) in built
        return {
            "files": files,
            "own_rewards": sorted(custom_kinds()),
            "build": self.build_state()["state"],
            "notes": self._library.notes(),
            "rewards": rewards.catalog(),
            "observation": found["env.observation"],
            "sources": resources.settings_sources(self._project, self._library),
        }

    @_safe
    def add_resources(self) -> dict[str, Any]:
        """Asks for files (settings files, reward sets, Rust files) and adds each to the library."""
        if self._window is None:
            return {"error": "no window to ask from"}
        import webview

        chosen = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=True,
            file_types=("NitroLearn files (*.toml;*.rs)", "All files (*.*)"),
        )
        if not chosen:
            return {"cancelled": True}
        return self.add_resource_paths([str(path) for path in chosen])  # type: ignore[no-any-return]

    @_safe
    def add_resource_paths(self, paths: list[str]) -> dict[str, Any]:
        added, refused = [], []
        for path in paths:
            try:
                added.append(self._library.add(Path(path)))
            except (OSError, ValueError) as err:
                refused.append(str(err))
        return {"added": added, "refused": refused}

    @_safe
    def build_rust(self) -> dict[str, Any]:
        """Builds NitroLearn again with the Rust rewards in the library (`nitro rebuild`, a few minutes).
        Bots training meanwhile carry on with the old build until they start again."""
        job = self._jobs.get(_BUILD, "build")
        if job is not None and not job.done:
            return {"error": "a build is already running"}
        files = [
            Path(e["path"])
            for e in self._library.entries()
            if e["kind"] == "rust_reward" and not e["problem"]
        ]
        rebuild.write_list(self._project, files)
        folder = self._runs / ".dashboard" / "build"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / CONSOLE).unlink(missing_ok=True)
        self._jobs.start(_BUILD, "build", ["rebuild"], folder)
        return {"started": True, "files": [path.name for path in files]}

    @_safe
    def build_state(self, after: int = 0) -> dict[str, Any]:
        """How the build is going: none, running, done or failed, and what it printed after byte `after`."""
        job = self._jobs.get(_BUILD, "build")
        state = (
            "none" if job is None else "running" if not job.done else "done" if job.code == 0 else "failed"
        )
        text, end = read_console(self._runs / ".dashboard" / "build", int(after))
        return {"state": state, "text": text, "next": end}

    def _convert_folder(self) -> Path:
        return self._runs / ".dashboard" / "convert"

    @_safe
    def converting(self, after: int = 0) -> dict[str, Any]:
        """Converting replays only: the store's replays by game size, the disk it takes and has free,
        the folders chosen, the pass's state and progress, and what it printed after byte `after`."""
        from nitro import replays

        folder = self._convert_folder()
        job = self._jobs.get(_CONVERT, "convert")
        state = (
            "none"
            if job is None
            else "running"
            if not job.done
            else "stopped"
            if job.stopped
            else "done"
            if job.code == 0
            else "failed"
        )
        progress = None
        live = replays.converting_file()
        if live.is_file():
            with contextlib.suppress(OSError, ValueError):
                progress = json.loads(live.read_text(encoding="utf-8"))
        chosen = folder / "folders.json"
        folders: list[str] = []
        if chosen.is_file():
            with contextlib.suppress(OSError, ValueError):
                folders = [str(f) for f in json.loads(chosen.read_text(encoding="utf-8"))]
        text, end = read_console(folder, int(after))
        return {
            "store": replays.store_summary(),
            "state": state,
            "progress": progress,
            "folders": folders,
            "threads": os.cpu_count() or 2,
            "text": text,
            "next": end,
        }

    @_safe
    def convert_start(self, folders: list[str], limit: int = 0) -> dict[str, Any]:
        """Starts converting the replays in `folders` (none: the game's own) into the store, every thread at
        work at low priority, learning nothing (`nitro convert-replays`)."""
        job = self._jobs.get(_CONVERT, "convert")
        if job is not None and not job.done:
            return {"error": "replays are already being converted"}
        missing = [f for f in folders if not Path(f).is_dir()]
        if missing:
            return {"error": f"not a folder: {', '.join(missing)}"}
        folder = self._convert_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "folders.json").write_text(json.dumps(folders), encoding="utf-8")
        (folder / CONSOLE).unlink(missing_ok=True)
        meshes = self._project / "collision_meshes"
        args = ["convert-replays", *folders, "--meshes", str(meshes)]
        if int(limit) > 0:
            args += ["--limit", str(int(limit))]
        self._jobs.start(_CONVERT, "convert", args, folder)
        return {"started": True}

    @_safe
    def convert_folders(self, folders: list[str]) -> dict[str, Any]:
        """Keeps the folders chosen for converting, for the next time the page opens."""
        folder = self._convert_folder()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "folders.json").write_text(json.dumps(folders), encoding="utf-8")
        return {"folders": folders}

    @_safe
    def remove_resource(self, ident: str) -> dict[str, Any]:
        self._library.remove(ident)
        return {"removed": ident}

    @_safe
    def set_library_note(self, key: str, text: str) -> dict[str, Any]:
        """A note on a resource: `reward:<name>`, `observation:<name>`, `actions:<name>` or `file:<id>`."""
        if not re.match(r"^(reward|observation|actions|file):[A-Za-z0-9_.-]{1,80}$", key or ""):
            return {"error": f"'{key}' is not something a note can be kept on"}
        return {"notes": self._library.set_note(key, text)}

    @_safe
    def reward_set(self, ident: str) -> dict[str, Any]:
        """The rewards of a reward set in the library, as the Rewards tab shows them."""
        path = self._library.path_of(ident, "rewards")
        views, _ = rewards.read(path.read_text(encoding="utf-8-sig"))
        return {"rewards": views}

    @_safe
    def save_reward_set(self, name: str, set_name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Saves rewards (a bot's, as its Rewards tab shows them) as a reward set in my-settings\\reward-sets
        and adds it to the library."""
        self._folder(name)
        file = resources.save_reward_set(self._project, set_name, [_entry(e) for e in entries], name)
        return {"saved": str(file), "entry": self._library.add(file)}

    # ---------- notes and preferences ----------
    @_safe
    def set_note(self, name: str, key: str, text: str) -> dict[str, Any]:
        folder = self._folder(name)
        notes = self._notes(folder)
        if text.strip():
            notes[key] = text
        else:
            notes.pop(key, None)
        edits.save(folder.root / "notes.json", json.dumps(notes, indent=1, ensure_ascii=False))
        return {"notes": notes}

    @_safe
    def set_pref(self, key: str, value: Any) -> dict[str, Any]:
        prefs = self._prefs()
        prefs[key] = value
        file = self._runs / ".dashboard" / "prefs.json"
        file.parent.mkdir(parents=True, exist_ok=True)
        edits.save(file, json.dumps(prefs, indent=1))
        return {"prefs": prefs}

    @_safe
    def set_livery(self, name: str, number: Any, main: Any, contrast: Any) -> dict[str, Any]:
        """Changes a bot's race number (one or two digits) and its two livery colours (`#RRGGBB`); the
        answer holds the livery as every bot's data does, the readable text colour included."""
        return {"livery": livery.write(self._folder(name).root, number, main, contrast)}

    # ---------- Check this PC ----------
    @_safe
    def check_pc(self) -> dict[str, Any]:
        """The checks of `nitro doctor`, each with its state, its message and how to fix it."""
        return checks.run(self._project)

    # ---------- helpers ----------
    def _summary(self, folder: RunFolder) -> dict[str, Any]:
        out = bots.summary(folder)
        name = folder.root.name
        out["pending_count"] = len(self._pending.get(name, {})) + len(self._reward_changes(folder, name))
        applying = self._applying.get(name)
        if applying and out.get("running_since", 0) > applying["at"]:
            self._applying.pop(name, None)  # started again with the changes
        elif applying and (problem := self._restart_problem(folder, applying["at"])):
            out["restart_problem"] = problem
            self._applying.pop(name, None)
        elif applying and time.time() - applying["at"] > 30 * 60:
            out["applying"] = "Still saving after 30 minutes; press Start when it has stopped"
            self._applying.pop(name, None)
        elif applying:
            out["applying"] = applying["text"]
        stopping = self._stopping.get(name)
        running = out["status"] in ("training", "copying")
        if stopping is not None and running and out.get("running_since", 0) <= stopping:
            out["stopping"] = True  # Stop was pressed; this trainer saves after its step, then ends
        elif stopping is not None:
            self._stopping.pop(name, None)
        starting = self._jobs.get(name, "train")
        if out["status"] in ("training", "copying"):
            self._note_running(name, bots.running(folder))
        elif starting is not None and not starting.done and not starting.seen_running:
            out["status"] = "starting"
        exporting = self._jobs.get(name, "export")
        out["exporting"] = exporting is not None and not exporting.done
        checking = self._jobs.get(name, "check")
        out["checking"] = checking is not None and not checking.done
        return out

    def _folder(self, name: str) -> RunFolder:
        if not _NAME.match(name or ""):
            raise ValueError(f"'{name}' is not a bot name")
        folder = RunFolder(self._runs / name)
        if not folder.info.is_file():
            raise ValueError(f"there is no bot called {name}")
        return folder

    def _check_new_name(self, name: str) -> None:
        if len(name or "") > 64:
            raise ValueError(f"a bot's name is at most 64 characters ({len(name)} given)")
        if not _NAME.match(name or ""):
            raise ValueError(
                "a bot's name uses letters, digits, '-', '_' and '.', and starts with a letter or digit"
            )
        if name.endswith(".") or name.split(".")[0].lower() in _RESERVED:
            raise ValueError(f"'{name}' cannot name a folder on Windows: pick another name")
        if (self._runs / name).exists():
            raise ValueError(f"there is already a bot called {name}")

    def _item(self, folder: RunFolder, path: str) -> dict[str, Any] | None:
        config = folder.load_settings()
        text = folder.settings.read_text(encoding="utf-8-sig")
        saves = [] if folder.copy_so_far() is not None else folder.complete_checkpoints()
        fixed = _start_locks(bots.start_state(folder, config, saves, 0))
        return _items(describe.describe(config, text, locked=bool(saves), fixed=fixed)).get(path)

    def _settings_dir(self, folder: RunFolder) -> Path:
        info = json.loads(folder.info.read_text(encoding="utf-8"))
        return Path(info["settings_dir"])

    def _preset(self, name: str) -> tuple[Path, Any, str]:
        """A starting settings file: a preset by name, or a settings source by its id ("file:<path>"), only
        from the presets, my-settings and the library."""
        presets = {p["name"]: Path(p["path"]) for p in self._presets()}
        if name in presets:
            file = presets[name]
        elif name.startswith("file:") and name in {
            s["id"] for s in self.settings_sources().get("sources", [])
        }:
            file = Path(name[len("file:") :])
        else:
            raise ValueError(f"there is no preset or settings file called {name}")
        return file, load_toml(file, NitroConfig), file.read_text(encoding="utf-8-sig")

    def _presets(self) -> list[dict[str, Any]]:
        return preset_cards(list((self._project / "configs" / "presets").glob("*.toml")))

    def _notes(self, folder: RunFolder) -> dict[str, str]:
        try:
            notes = json.loads((folder.root / "notes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in notes.items()} if isinstance(notes, dict) else {}

    def _prefs(self) -> dict[str, Any]:
        try:
            prefs = json.loads((self._runs / ".dashboard" / "prefs.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"theme": "dark"}
        return prefs if isinstance(prefs, dict) else {"theme": "dark"}


def _start_section(run: str, at: int, on: bool, parts: str) -> dict[str, Any]:
    """[start_from] as a new bot's settings file writes it, every choice in plain sight."""
    return {"enabled": on, "run": run, "at": at, "parts": parts}


def _with_start(text: str, values: dict[str, Any]) -> str:
    """A copied settings file with its [start_from] written anew: the copied one said how the other bot
    started."""
    note = "Where this bot starts from (transfer learning, read at its first start): see its Settings tab."
    return edits.replaced(text, "start_from", values, note)


def _start_locks(start: dict[str, Any]) -> dict[str, str]:
    """The [start_from] settings that can no longer change (path -> why): all of them after the first save."""
    if not start["done"]:
        return {}
    return {f"start_from.{name}": _STARTED for name in ("enabled", "run", "at", "parts")}


def _items(sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every setting in `describe`'s sections by its path."""
    found: dict[str, dict[str, Any]] = {}
    stack = list(sections)
    while stack:
        section = stack.pop()
        found.update((item["path"], item) for item in section["items"])
        stack.extend(section["sections"])
    return found


def _entry(entry: Any) -> dict[str, Any]:
    """One reward entry from the window, checked for shape (its values are checked as settings)."""
    if not isinstance(entry, dict) or not isinstance(entry.get("view"), dict):
        raise ValueError("a reward entry needs its view")
    origin = entry.get("from")
    view = dict(entry["view"])
    if not isinstance(view.get("name"), str) or not isinstance(view.get("params", {}), dict):
        raise ValueError("a reward needs a name")
    return {
        "from": origin if isinstance(origin, int) and not isinstance(origin, bool) else None,
        "view": view,
    }


def _open(path: Path) -> None:
    """Opens a folder or program the way double-clicking it would."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined, unused-ignore]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
