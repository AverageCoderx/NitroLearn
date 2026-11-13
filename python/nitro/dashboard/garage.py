"""What the Garage page and the new-bot wizard ask beside `home`: each bot's newest save, reward so far and
copy progress in one cheap call; a livery picked and checked before its bot exists; and the parameters of the
networks a new bot would get. Counting parameters builds the networks, which needs PyTorch: that runs as its
own short process (this file, as a program), so the dashboard's own process never loads it."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import bots, describe, edits, layout, livery, machine, winapp
from nitro.dashboard.jobs import Jobs
from nitro.runs import (
    ATTENTION_SETTINGS,
    TRANSFORMER_SETTINGS,
    RunFolder,
    encoder_settings,
    fixed_settings,
)

POINTS = 40  # points of a row's reward line
SECONDS = 120.0  # the first load of PyTorch from a cold disk can take a while
# nitro.pretrain's names for a copy's folder and its progress file; not imported, since it loads PyTorch.
_COPY, _LIVE = "pretrained", "live.json"
_CONVERT = (".convert", "convert")  # the jobs list's entry for converting replays
_COUNTS: dict[str, dict[str, Any]] = {}  # settings text (its digest) -> the networks' parameters
_KEPT_COUNTS = 500  # counted texts remembered before the memory of them starts anew
WHOLE_BYTES = 256 * 1024  # a metrics file's rows up to this size are read whole for the reward line
_REWARDS: dict[Path, tuple[Any, list[float]]] = {}  # metrics file -> (its identity and points, the line)

# The review's plain words for the values of the settings a first save fixes.
_WORDS: dict[str, dict[Any, str]] = {
    "run.precision": {"fp32": "full (FP32)", "fp16": "half (FP16)"},
    "model.encoder": {
        "mlp": "plain",
        "attention": "attention",
        "attention_communication": "attention, cars compare notes",
        "transformer": "transformer",
    },
    "env.observation": {"nitro_v2": "detailed (nitro_v2)", "native": "basic (native)"},
    "env.opponents": {True: "an opposing team", False: "none, it plays alone"},
    "model.activation": {"relu": "ReLU", "leaky_relu": "leaky ReLU", "gelu": "GELU", "tanh": "tanh"},
}
# Always in the review: the game the bot is made for and its network's size and kind.
_REVIEW_ALWAYS = frozenset(
    {"env.team_size", "env.observation", "model.policy", "model.value", "model.shared", "model.encoder"}
    | {"run.precision"}
)
# Values that add nothing a newcomer needs to know: left out of the review.
_REVIEW_NOTHING: dict[str, Any] = {
    "env.obs_team_size": 0,
    "model.value_heads": ["main"],
    "model.value_head_layers": [],
    "model.residual": False,
}


# The wizard's card for a preset its file name says too little about: its title, its few words and its tags
# (plain labels); with "brains", the brain sizes it offers in place of the plain ones (which change only the
# policy and the value layers), each with the settings it changes and keeping the design's shape. "Full"
# changes none: the preset as written. Presets not listed keep the plain card and sizes.
PRESET_CARDS: dict[str, dict[str, Any]] = {
    "gco_1v1_starter": {
        "title": "GCO bot",
        "about": "Untested design: learns from goals scored and conceded only. Slower to start.",
        "tags": ["Experimental"],
        "brains_about": (
            "How big the bot's networks are. Both sizes keep this design's shape: a shared trunk, the policy "
            "on top of it, and a large value side with four value heads that only help it learn (the bot "
            "that plays runs the trunk and the policy). Fixed once the bot has its first save."
        ),
        "brains": [
            {
                "key": "small",
                "title": "Small",
                "about": "The starter as written: under half the design's width. Fits an 8 GB card.",
                "values": {},
            },
            {
                "key": "full",
                "title": "Full",
                "about": "The design's full width, in the same shares. For a card with 12 GB or more.",
                "values": {
                    "model.shared": [1152, 1152, 1152],
                    "model.policy": [768, 768, 768],
                    "model.value": [1280, 1280, 1280],
                    "model.value_head_layers": [1280, 1280],
                },
            },
        ],
    },
}
_STARTER = re.compile(r"^(\d)v\1_starter$")  # a plain starter for one game size, as 1v1_starter


def preset_cards(found: list[Path]) -> list[dict[str, Any]]:
    """The presets in `found` as the wizard offers them, the plain starters first and then the rest by
    name: each {"name", "path"} with what PRESET_CARDS has for it."""
    ordered = sorted(found, key=lambda path: (not _STARTER.match(path.stem), path.stem))
    return [{"name": path.stem, "path": str(path)} | PRESET_CARDS.get(path.stem, {}) for path in ordered]


class GarageApi:
    """The Garage page's and the new-bot wizard's calls; `nitro.dashboard.api.Api` inherits them."""

    _runs: Path
    _project: Path
    _jobs: Jobs

    if TYPE_CHECKING:

        def _bot_changes(
            self, preset: str, values: dict[str, Any]
        ) -> tuple[Path, str, dict[str, Any], dict[str, str], list[str]]: ...

    def garage(self) -> dict[str, Any]:
        """Beside `home`, per bot: when its newest save was written, its reward so far as a few points, and
        how far a copy of replays is; and whether replays are being converted."""
        try:
            rows = {folder.root.name: _row(folder) for folder in bots.bot_folders(self._runs)}
        except OSError as err:
            return {"error": str(err)}
        convert = self._jobs.get(*_CONVERT)
        return {"bots": rows, "converting": convert is not None and not convert.done}

    def livery_pick(self, name: str) -> dict[str, Any]:
        """The number and colours a bot called `name` gets when none are chosen, and the colour pairs to
        choose from."""
        return {"livery": livery.pick(str(name)), "palette": [list(pair) for pair in livery.PALETTE]}

    def livery_check(self, number: Any, main: Any, contrast: Any) -> dict[str, Any]:
        """A livery as `set_livery` would keep it (its readable text colour included), or why it cannot be
        taken; nothing is written."""
        try:
            return {"livery": livery.checked(number, main, contrast)}
        except livery.LiveryError as err:
            return {"error": str(err)}

    def fixed_after_first_save(self, preset: str, values: dict[str, Any]) -> dict[str, Any]:
        """The settings of a new bot from `preset` with the form's `values` that can no longer change once
        it has a save, each with its label and its value; the sizes of an encoder it does not use are left
        out."""
        try:
            file, text, changes, errors, _ = self._bot_changes(preset, values)
            if errors:
                return {"fixed": []}
            made = edits.written(text, changes)
            config = parse_toml(made, NitroConfig, file=file, base_dir=file.resolve().parent)
        except (ConfigError, OSError, ValueError, KeyError) as err:
            return {"error": str(err)}
        used = encoder_settings(fixed_settings(config), config)
        unused = {*ATTENTION_SETTINGS, *TRANSFORMER_SETTINGS} - set(used)
        fixed = [
            {
                "path": item["path"],
                "label": item["label"],
                "value": item["value"],
                "title": layout.place(item["path"]).title or item["label"],
                "words": review_words(item["path"], item["value"], machine.graphics_card()),
            }
            for item in _settings(describe.describe(config, made, locked=True))
            if item["fixed"] and item["path"] not in unused and _in_review(item, set(used))
        ]
        return {"fixed": fixed}

    def brain_sizes(self, preset: str, variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """The parameters of the networks a new bot from `preset` would get with each of `variants` (a name
        -> the form's values): {"total", "policy", "value", ...} each, or {"error": why} for values the
        form would refuse or networks that cannot be built. Each settings text is counted once."""
        texts: dict[str, str] = {}
        sizes: dict[str, dict[str, Any]] = {}
        file = Path()
        try:
            for key, values in (variants or {}).items():
                file, text, changes, errors, _ = self._bot_changes(preset, values)
                if errors:
                    sizes[key] = {"error": " ".join(errors.values())}
                else:
                    texts[key] = edits.written(text, changes)
        except (ConfigError, OSError, ValueError, KeyError) as err:
            return {"error": str(err)}
        asked = {_digest(text): text for text in texts.values() if _digest(text) not in _COUNTS}
        if asked:
            try:
                counted = count_parameters(file, asked, self._project)
            except CountError as err:
                return {"error": str(err)}
            if len(_COUNTS) > _KEPT_COUNTS:
                _COUNTS.clear()
            _COUNTS.update(counted)
        for key, text in texts.items():
            sizes[key] = _COUNTS.get(_digest(text), {"error": "these networks cannot be built"})
        return {"sizes": sizes}


def review_words(path: str, value: Any, card: bool | None = None) -> str:
    """A fixed setting's value in plain words for the wizard's review: "full (FP32)", "3 layers of 256".
    `card` False: this PC has no graphics card, so half precision is used only where the bot moves to one."""
    if path == "run.precision" and value == "fp16" and card is False:
        return "half (FP16) on a graphics card; this PC trains in full precision (FP32) on its processor"
    if path == "model.value_heads" and isinstance(value, list):
        heads = {"main": "the main one", "goal": "a goal critic", "vdag1": "two twin critics"}
        named = [heads[head] for head in value if head in heads]
        return f"{len(value)} ({', '.join(named)})" if named else str(len(value))
    words = _WORDS.get(path, {})
    if not isinstance(value, list) and value in words:
        return words[value]
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        if not value:
            return "none"
        if len(set(value)) == 1 and all(type(n) is int for n in value):
            return f"{len(value)} {'layer' if len(value) == 1 else 'layers'} of {value[0]:,}"
        return ", ".join(str(n) for n in value)
    if path == "env.team_size":
        return f"{value} ({value}v{value})"
    return "not set" if value is None or value == "" else str(value)


def _in_review(item: dict[str, Any], used: set[str]) -> bool:
    """Whether the review lists a fixed setting: the game, the network's size and kind and the sizes of the
    encoder in use always; the rest only when it differs from its default to something that means more."""
    path = item["path"]
    if path in _REVIEW_ALWAYS or (path in used and path in {*ATTENTION_SETTINGS, *TRANSFORMER_SETTINGS}):
        return True
    if path in _REVIEW_NOTHING and item["value"] == _REVIEW_NOTHING[path]:
        return False
    return bool(item["value"] != item["default"])


def _settings(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every setting of `sections` (what `describe` gives), those of the sections inside them included."""
    return [item for s in sections for item in (*s["items"], *_settings(s["sections"]))]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(folder: RunFolder) -> dict[str, Any]:
    saves = folder.complete_checkpoints()
    last_save = None
    if saves:
        try:
            last_save = (saves[-1] / "state.pt").stat().st_mtime
        except OSError:
            last_save = None
    return {"last_save": last_save, "reward": reward_points(folder), "copy": _copy(folder)}


def reward_points(folder: RunFolder, points: int = POINTS) -> list[float]:
    """The bot's reward per step so far as at most `points` values, oldest first and the newest always in.
    A short file's rows are picked evenly by number; a long one's are the rows at places spread evenly through
    the file (`bots.spread_rows`: a few small reads, where reading it whole took seconds a bot). Read again
    only when the file changed."""
    found = bots.ends(folder)
    if not found.last or "reward_per_step" not in found.header:
        return []
    key = (found.identity, points)
    known = _REWARDS.get(folder.metrics)
    if known is not None and known[0] == key:
        return list(known[1])
    if found.end - found.start <= WHOLE_BYTES:
        rows = bots.whole_rows(folder)
        spread = min(points, len(rows))
        picked = {round(i * (len(rows) - 1) / max(1, spread - 1)) for i in range(spread)}
        rows = [rows[i] for i in sorted(picked)]
    else:
        rows = bots.spread_rows(folder, points)
    column = found.header.index("reward_per_step")
    values = [value for row in rows if (value := bots.number(row[column])) is not None]
    _REWARDS[folder.metrics] = (key, values)
    return list(values)


def _copy(folder: RunFolder) -> dict[str, int] | None:
    """How far the bot's copy of replays is, as the copy writes it while it runs: replays learned and how
    many it will learn; None without one."""
    try:
        data = json.loads((folder.root / _COPY / _LIVE).read_text(encoding="utf-8"))
        return {"learned": int(data["learned"]), "limit": int(data["limit"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


class CountError(RuntimeError):
    """The process that counts parameters did not answer, in plain English."""


def count_parameters(file: Path, texts: dict[str, str], project: Path) -> dict[str, dict[str, Any]]:
    """The parameters of the networks each of `texts` (a name -> a settings file's text, read as if it were
    `file`) describes, counted by this file run as its own process: per name the counts, or {"error": why}
    for networks that cannot be built. CountError when the process gives no answer."""
    flags = 0
    if sys.platform == "win32":  # no console window flashing up
        flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined, unused-ignore]
    try:
        done = subprocess.run(
            [winapp.console_python(), "-m", "nitro.dashboard.garage"],
            cwd=project,
            input=json.dumps({"file": str(file), "texts": texts}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SECONDS,
            creationflags=flags,
        )
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        counted = json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        raise CountError("counting the networks' parameters took too long") from None
    except (OSError, IndexError, ValueError):
        raise CountError("the networks' parameters could not be counted") from None
    if not isinstance(counted, dict) or not all(isinstance(counts, dict) for counts in counted.values()):
        raise CountError("the networks' parameters could not be counted")
    return counted


def _count_here() -> None:
    """Reads {"file", "texts"} from the input and prints each text's parameter counts as one JSON line."""
    from nitro.model import parameter_counts

    asked = json.loads(sys.stdin.read())
    file = Path(asked["file"])
    out: dict[str, Any] = {}
    for key, text in asked["texts"].items():
        try:
            config = parse_toml(text, NitroConfig, file=file, base_dir=file.resolve().parent)
            out[key] = parameter_counts(config)
        except Exception as err:  # one refused variant must not lose the others' counts
            out[key] = {"error": str(err) or type(err).__name__}
    print(json.dumps(out))


if __name__ == "__main__":
    _count_here()
