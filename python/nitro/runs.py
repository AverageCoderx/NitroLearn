"""Run folders and checkpoint files.

A run lives in `runs/<name>/`: its settings, metrics, report and checkpoints. A checkpoint is written to
a temporary folder and renamed into place, so a crash or a power cut never leaves a half-written one
behind; resuming takes the newest complete one that reads (one cut short otherwise, as by a full disk, is
named and passed over). The run's settings may be edited between stopping and resuming, except the ones its
saved networks are built for.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nitro import _core
from nitro.config import NitroConfig
from nitro.config.loader import ConfigError, ConfigIssue, KeyLocator, parse_toml
from nitro.config.schema import PROJECT

_CHECKPOINT = re.compile(r"^\d{12}$")
_KEPT = "kept"  # a checkpoint with this file is never pruned
# A checkpoint with this file holds a copy of replays still being made (pretrain.save_every): it can be
# watched, exported and have rewards tried on it, but the bot's next start carries on the copy.
COPY_SO_FAR = "copy_so_far.json"
# What torch.load raises for a file cut short or damaged (a crash, a power cut or a full disk while it was
# written): an empty file, a zip without its directory, bytes that are not a save, or a failed read.
UNREADABLE = (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError)
# Settings that shape the networks and the learner's saved state; a resumed run must keep them.
FIXED_SETTINGS = (
    "env.team_size", "env.opponents", "env.observation", "env.obs_team_size",
    "model.shared", "model.policy", "model.value", "model.layer_norm", "model.activation", "ppo.popart",
    "run.precision", "model.encoder", "model.attention_width", "model.attention_blocks",
    "model.attention_heads", "model.attention_skip", "model.transformer_layers", "model.transformer_width",
    "model.transformer_heads", "model.transformer_ff", "model.transformer_pads", "model.transformer_readout",
    "model.residual", "model.value_heads", "model.value_head_layers",
)  # fmt: skip
# A padded observation (env.obs_team_size) is the same size for every game up to its room, so the
# team size and opponents can change on a resume as long as the room stays.
_FREE_WHEN_PADDED = ("env.team_size", "env.opponents")
# An attention encoder's weights do not depend on how many cars it reads, so with one the game size and
# the room may change too: a 2v2 bot trains on at 3v3.
_FREE_WITH_ATTENTION = (*_FREE_WHEN_PADDED, "env.obs_team_size")
# Each encoder's sizes (the transformer) shape only networks with that encoder; the others
# ignore them.
ATTENTION_SETTINGS = (
    "model.attention_width", "model.attention_blocks", "model.attention_heads", "model.attention_skip"
)  # fmt: skip
TRANSFORMER_SETTINGS = (
    "model.transformer_layers", "model.transformer_width", "model.transformer_heads", "model.transformer_ff",
    "model.transformer_pads", "model.transformer_readout", "model.attention_skip",
)  # fmt: skip
_ENCODER_SETTINGS = {
    "attention": ATTENTION_SETTINGS,
    "attention_communication": ATTENTION_SETTINGS,
    "transformer": TRANSFORMER_SETTINGS,
}


def encoder_settings(saved: dict[str, Any], config: NitroConfig) -> tuple[str, ...]:
    """The sizes that shape the saved networks' encoder or `config`'s, in order (none for plain networks)."""
    kinds = (saved["model.encoder"], config.model.encoder)
    return tuple(dict.fromkeys(path for kind in kinds for path in _ENCODER_SETTINGS.get(kind, ())))


def fixed_settings(config: NitroConfig) -> dict[str, Any]:
    values = {}
    for path in FIXED_SETTINGS:
        section, name = path.split(".")
        values[path] = getattr(getattr(config, section), name)
    return values


def reads_any_cars(saved: dict[str, Any], config: NitroConfig) -> bool:
    """Whether the networks saved with the fixed settings `saved` and `config`'s have the same attention
    encoder, which reads any number of cars: then the game size and the room may differ."""
    return config.model.uses_attention and saved["model.encoder"] == config.model.encoder


def _free(path: str, saved: dict[str, Any], config: NitroConfig) -> bool:
    if path in ATTENTION_SETTINGS or path in TRANSFORMER_SETTINGS:
        return path not in encoder_settings(saved, config)
    if path in _FREE_WITH_ATTENTION and reads_any_cars(saved, config):
        return True
    room = config.env.obs_team_size
    return path in _FREE_WHEN_PADDED and room > 0 and saved["env.obs_team_size"] == room


def game_size_free(config: NitroConfig) -> tuple[str, ...]:
    """The game-size settings among the fixed ones that a bot saved with `config`'s may change between stop
    and resume: the size and room with an attention encoder, the size with a padded room."""
    saved = fixed_settings(config)
    return tuple(path for path in _FREE_WITH_ATTENTION if _free(path, saved, config))


def policy_rows(model_state: dict[str, Any]) -> int:
    """How many actions a saved policy chooses among: the outputs of its last layer."""
    layers = [int(k.split(".")[1]) for k in model_state if k.startswith("policy.") and k.endswith(".weight")]
    return int(model_state[f"policy.{max(layers)}.bias"].shape[0])


def actions_problem(model_state: dict[str, Any], what: str = "This save") -> str | None:
    """Why a saved policy (`what`) cannot be used here, when it does not choose among the action table's
    actions; None when it does."""
    rows, table = policy_rows(model_state), _core.num_actions()
    if rows == table:
        return None
    return (
        f"{what} chooses among {rows} actions, but this version of NitroLearn has {table}: it was made by "
        "another version and cannot be used here"
    )


def portable(path: Path, run: Path | None = None) -> str:
    """`path` as saves record it: relative to the bot's folder `run` or to NitroLearn's folder when it is
    inside one, so the folders of this computer above them are not recorded; else as it is."""
    if path.is_absolute():
        for folder in filter(None, (run, PROJECT)):
            for base in (folder, folder.resolve()):
                if path.is_relative_to(base):
                    return path.relative_to(base).as_posix() or "."
    return str(path)


def all_settings(config: NitroConfig, run: Path | None = None) -> dict[str, Any]:
    """Every setting as a flat {"section.name": plain value} dict, saved with each checkpoint so a resume
    can say which settings changed; a path as `portable` gives it (`run`: the bot's folder)."""

    def flat(value: Any, prefix: str) -> dict[str, Any]:
        if isinstance(value, dict):
            return {k: v for key, item in value.items() for k, v in flat(item, f"{prefix}{key}.").items()}
        return {prefix.rstrip("."): value}

    def text(value: Any) -> str:
        return portable(value, run) if isinstance(value, Path) else str(value)

    plain = json.loads(json.dumps(dataclasses.asdict(config), default=text))
    return flat(plain, "")


def _items(flat: dict[str, Any]) -> dict[str, Any]:
    """`flat` with lists (the rewards, the state setters) spelled out item by item: `rewards[3].weight`."""
    out: dict[str, Any] = {}

    def add(value: Any, key: str) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                add(item, f"{key}.{name}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                add(item, f"{key}[{i}]")
        else:
            out[key] = value

    for key, value in flat.items():
        add(value, key)
    return out


def changed_settings(saved: dict[str, Any], current: dict[str, Any], moved: bool = False) -> list[str]:
    """'section.name: saved -> current' for every setting that differs (paths aside: they may move). With
    `moved` (a bot unpacked on another computer), a file at another path under its own name is no change."""
    return [f"{key}: {before} -> {after}" for key, before, after in setting_changes(saved, current, moved)]


def setting_changes(
    saved: dict[str, Any], current: dict[str, Any], moved: bool = False
) -> list[tuple[str, str, str]]:
    """(setting, saved value, current value) for every setting that differs, as `changed_settings`."""
    saved, current = _items(saved), _items(current)
    return [
        (key, str(saved.get(key, "(none)")), str(current.get(key, "(none)")))
        for key in sorted(saved.keys() | current.keys())
        if not (
            key.startswith("paths.")
            or saved.get(key) == current.get(key)
            or (moved and _moved(saved.get(key), current.get(key)))
        )
    ]


class UnreadableSave(Exception):
    """A save that cannot be read, in plain words: which, where, and what to do."""


def unreadable_reason(err: BaseException) -> str:
    """Why a save file could not be read, in plain words."""
    if isinstance(err, OSError):
        return f"it could not be read ({err.strerror or err})"
    if isinstance(err, EOFError):
        return "it is empty (cut short while it was written, as by a crash, a power cut or a full disk)"
    return "it is damaged or cut short (as by a crash, a power cut or a full disk while it was written)"


def _warn(line: str) -> None:
    print(f"WARNING: {line}", file=sys.stderr, flush=True)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resume_command(folder: RunFolder) -> str:
    """The command that carries on `folder`'s run, with --runs when its runs folder is not the default."""
    runs = folder.root.parent
    elsewhere = runs.resolve() != Path("runs").resolve()
    return f"nitro resume {folder.root.name}" + (f' --runs "{runs}"' if elsewhere else "")


def _moved(saved: Any, current: Any) -> bool:
    """Paths to files of the same names: the files moved (another computer), nothing changed. A pack
    names a file outside the folders it knows by its name alone (`nitro.pack`)."""
    if isinstance(saved, list | tuple) and isinstance(current, list | tuple):
        return len(saved) == len(current) and all(
            _moved(a, b) or a == b for a, b in zip(saved, current, strict=True)
        )
    if not (isinstance(saved, str) and isinstance(current, str)):
        return False
    parts = [re.split(r"[\\/]", text) for text in (saved, current)]
    return any(len(path) > 1 for path in parts) and parts[0][-1] == parts[1][-1]


def numbered(names: Sequence[str]) -> list[str]:
    """The names as metrics labels: a repeated name gets a number from its second appearance on
    (`all_of`, `touch_ball`, `all_of` -> `all_of`, `touch_ball`, `all_of_2`), so no column is lost."""
    seen: dict[str, int] = {}
    labels = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return labels


def source_paths(name: str, runs: Path) -> list[Path]:
    """Where a run to start from called `name` (start_from.run) is looked for: as given when it is a full
    path, else in the runs folder `runs`, then in the current folder."""
    named = Path(name)
    return [named] if named.is_absolute() else [runs / named, Path.cwd() / named]


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, tuple):
        return f"[{', '.join(map(_toml, value))}]"
    return f'"{value}"' if isinstance(value, str) else str(value)


@dataclass(frozen=True)
class RunFolder:
    root: Path

    @property
    def settings(self) -> Path:
        return self.root / "settings.toml"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.csv"

    @property
    def report(self) -> Path:
        return self.root / "report.html"

    @property
    def log(self) -> Path:
        return self.root / "train.log"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def info(self) -> Path:
        return self.root / "run.json"

    @classmethod
    def create(
        cls, runs: Path, name: str | None, settings: Path, start_from: dict[str, Any] | None = None
    ) -> RunFolder:
        """A new, empty run folder with a copy of the settings file; refuses to reuse an existing name.
        `start_from` (`nitro train --from`: run and save) is kept with it, so a restart before the first
        save still starts from that run."""
        name = name or time.strftime("run-%Y%m%d-%H%M%S")
        folder = cls(runs / name)
        if folder.root.exists():
            again = (
                ""
                if folder.complete_checkpoints() or not folder.info.is_file()
                else f" It has no save yet: to start it again from {settings}, delete the folder "
                f"{folder.root} first."
            )
            raise FileExistsError(
                f"A run called '{name}' already exists in {runs}. To continue it, use: "
                f"{resume_command(folder)}.{again}"
            )
        text = settings.read_text(encoding="utf-8-sig")
        folder.checkpoints.mkdir(parents=True)
        folder.settings.write_text(text, encoding="utf-8")
        # Relative paths in the copy keep meaning what they meant next to the original file. The file's name
        # and what the copy said are kept too, so a run that stops before its first save can read the file
        # again once it is fixed (`refresh_settings`).
        info: dict[str, Any] = {
            "settings_dir": str(settings.resolve().parent),
            "settings_file": settings.name,
            "settings_copied": _digest(text),
        }
        if start_from:
            info["start_from"] = start_from
        folder.info.write_text(json.dumps(info), encoding="utf-8")
        return folder

    def started_with(self) -> Path | None:
        """The settings file this run was made from (`nitro train`, `nitro new`), or None when not known."""
        info = json.loads(self.info.read_text(encoding="utf-8"))
        return Path(info["settings_dir"]) / info["settings_file"] if info.get("settings_file") else None

    def refresh_settings(self) -> str | None:
        """For a run with no save yet: when the settings file it was made from changed since and the run's
        own copy did not, the copy becomes that file again (the run has nothing its networks must keep). A
        sentence saying what was done, or what to do when both changed; None when nothing changed."""
        info = json.loads(self.info.read_text(encoding="utf-8"))
        original = self.started_with()
        if original is None or self.complete_checkpoints():
            return None
        try:
            text = original.read_text(encoding="utf-8-sig")
            copy = self.settings.read_text(encoding="utf-8-sig")
        except OSError:
            return None
        if text == copy:
            return None
        if _digest(copy) != info.get("settings_copied"):
            return (
                f"{original} changed since this run was made, but the run's own copy ({self.settings}) was "
                f"changed too, so the run uses its copy. To start it again from {original} instead, delete "
                f"the folder {self.root} and run: nitro train {original} --name {self.root.name}"
            )
        self.settings.write_text(text, encoding="utf-8")
        self.info.write_text(json.dumps({**info, "settings_copied": _digest(text)}), encoding="utf-8")
        return (
            f"{original} changed since this run was made, and the run has no save yet: it starts with the "
            "new settings."
        )

    def load_settings(self, state: dict[str, Any] | None = None, text: str | None = None) -> NitroConfig:
        """The run's settings (or `text` as its settings file); with `state` (a checkpoint), refuses changes
        its networks cannot follow.

        A start_from kept in run.json (`nitro train --from`) is laid over the file's [start_from], all but
        its switch: `enabled = false` in the file means from scratch."""
        info = json.loads(self.info.read_text(encoding="utf-8"))
        base_dir = Path(info["settings_dir"])
        text = self.settings.read_text(encoding="utf-8-sig") if text is None else text
        config = parse_toml(text, NitroConfig, file=self.settings, base_dir=base_dir)
        if "start_from" in info:
            given = {**info["start_from"], "enabled": config.start_from.enabled}
            config = dataclasses.replace(config, start_from=dataclasses.replace(config.start_from, **given))
        meshes = info.get("collision_meshes")  # this computer's, set by `nitro unpack`
        if meshes and not (config.paths.collision_meshes / "soccar").is_dir():
            paths = dataclasses.replace(config.paths, collision_meshes=Path(meshes))
            config = dataclasses.replace(config, paths=paths)
        if state is not None:
            saved, lines = state["settings"], KeyLocator(text)
            issues = [
                ConfigIssue(
                    path,
                    f"was {_toml(saved[path])} when the run was saved, and its saved networks only fit that",
                    lines.line_of(path),
                    "put it back, or start a new run (nitro train) to change it",
                )
                for path, value in fixed_settings(config).items()
                if saved[path] != value and not _free(path, saved, config)
            ]
            if issues:
                raise ConfigError(self.settings, issues)
        return config

    def save_checkpoint(
        self,
        state: dict[str, Any],
        agent_steps: int,
        keep: int,
        milestone_steps: int,
        kept: bool = False,
        copy_so_far: dict[str, Any] | None = None,
    ) -> Path:
        """Writes `state` as the checkpoint at `agent_steps`; keeps the newest `keep`, the milestones and
        every checkpoint saved `kept` (never pruned). `copy_so_far` marks
        a save of an unfinished copy of replays.

        A milestone is the first checkpoint of every `milestone_steps` agent steps (0: none).
        """
        final = self.checkpoints / f"{agent_steps:012d}"
        partial = final.with_name(final.name + ".partial")
        # A save interrupted by a kill or a power cut leaves a `.partial` folder behind, at any step count.
        for stale in self.checkpoints.glob("*.partial"):
            shutil.rmtree(stale, ignore_errors=True)
        import torch  # here, not at the top: the dashboard reads runs without loading PyTorch (1.4 GB, PC)

        try:
            partial.mkdir(parents=True)
            with (partial / "state.pt").open("wb") as file:
                torch.save(state, file)
                file.flush()
                os.fsync(file.fileno())
            if kept:
                (partial / _KEPT).write_text("kept through pruning\n", encoding="utf-8")
            if copy_so_far is not None:
                (partial / COPY_SO_FAR).write_text(json.dumps(copy_so_far), encoding="utf-8")
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)  # a full disk gets its space back
            raise
        if final.exists():
            shutil.rmtree(final)
        os.replace(partial, final)
        complete = self.complete_checkpoints()
        milestones: dict[int, Path] = {}
        if milestone_steps:
            for path in complete:
                milestones.setdefault(int(path.name) // milestone_steps, path)
        for old in complete[:-keep]:
            if old not in milestones.values() and not (old / _KEPT).is_file():
                shutil.rmtree(old)
        return final

    def complete_checkpoints(self) -> list[Path]:
        """Finished checkpoints, oldest first (folders still named `.partial` were interrupted)."""
        if not self.checkpoints.is_dir():
            return []
        return sorted(
            p for p in self.checkpoints.iterdir() if _CHECKPOINT.match(p.name) and (p / "state.pt").is_file()
        )

    def copy_so_far(self) -> dict[str, Any] | None:
        """How far the copy of replays is in the newest save, when that save holds an unfinished copy:
        {"learned": replays, "limit": replays}; None for any other save, or none."""
        complete = self.complete_checkpoints()
        if not complete:
            return None
        try:
            note = json.loads((complete[-1] / COPY_SO_FAR).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return note if isinstance(note, dict) else None

    def drop_copy_so_far(self) -> None:
        """Deletes the saves of an unfinished copy of replays: the copy is finished or made anew."""
        for path in self.complete_checkpoints():
            if (path / COPY_SO_FAR).is_file():
                shutil.rmtree(path)

    def load_checkpoint(self, agent_steps: int) -> dict[str, Any] | None:
        """The finished checkpoint saved at exactly `agent_steps` agent steps, or None; UnreadableSave when it
        does not read."""
        for path in self.complete_checkpoints():
            if int(path.name) == agent_steps:
                try:
                    return self._load(path)
                except UNREADABLE as err:
                    saves = self._steps(self.complete_checkpoints())
                    raise UnreadableSave(
                        f"{self._unreadable([(path, err)])} It was left where it is. Saves: {saves}."
                    ) from err
        return None

    def load_latest(self, warn: Callable[[str], None] = _warn) -> dict[str, Any] | None:
        """The newest save that reads, or None when there is none. Saves that do not read (cut short by a
        crash, a power cut or a full disk) are said through `warn` and left where they are; the newest one
        before them is used. UnreadableSave when no save reads."""
        complete = self.complete_checkpoints()
        bad: list[tuple[Path, BaseException]] = []
        for path in reversed(complete):
            try:
                state = self._load(path)
            except UNREADABLE as err:
                bad.append((path, err))
                continue
            if bad:
                warn(
                    f"{self._unreadable(bad)} {'It was' if len(bad) == 1 else 'They were'} left where "
                    f"{'it is' if len(bad) == 1 else 'they are'}; '{self.root.name}' goes on from the save "
                    f"before, at {int(path.name):,} agent steps."
                )
            return state
        if not bad:
            return None
        raise UnreadableSave(
            f"{self._unreadable(bad)} {'It was' if len(bad) == 1 else 'They were'} left where "
            f"{'it is' if len(bad) == 1 else 'they are'}, and '{self.root.name}' has no other save. To start "
            f"it again from the beginning, move the folder {self.checkpoints} somewhere else, then run: "
            f"{resume_command(self)}"
        )

    def _load(self, path: Path) -> dict[str, Any]:
        import torch

        # Checkpoints hold only tensors and plain values, so PyTorch's safe loader is enough.
        state: dict[str, Any] = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        return state

    def _unreadable(self, bad: list[tuple[Path, BaseException]]) -> str:
        """'The save of 'x' at 2,000 agent steps (path) cannot be read: why.' for one save or several."""
        if len(bad) == 1:
            path, err = bad[0]
            return (
                f"The save of '{self.root.name}' at {int(path.name):,} agent steps ({path / 'state.pt'}) "
                f"cannot be read: {unreadable_reason(err)}."
            )
        return (
            f"{len(bad)} saves of '{self.root.name}' in {self.checkpoints} cannot be read (at "
            f"{self._steps([p for p, _ in bad])} agent steps): {unreadable_reason(bad[0][1])}."
        )

    @staticmethod
    def _steps(paths: Sequence[Path]) -> str:
        return ", ".join(f"{int(p.name):,}" for p in sorted(paths)) or "none"
