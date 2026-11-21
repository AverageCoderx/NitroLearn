"""A real game scored with a bot's rewards (`nitro score-replay`): what each of the bot's rewards would have
paid each player of a recorded game, step by step, shown in the match viewer with the reward chart ("what
would my rewards pay a pro?").

The game is a converted replay (`nitro convert-replays` keeps them in the replay store; the replay's own
file finds its conversion there) or a recording (`.rlpr`, `.rlpr.zst`, such as the real games RocketSim
ships). The core puts the arena in each recorded state and plays the step with the players' recorded
controls for the events the rewards read (`nitro_core::score`); the players' value estimates and choices are
not shown, as no bot played.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nitro import _core, replays
from nitro.config import NitroConfig
from nitro.envs import core_settings, probe_config
from nitro.runs import numbered
from nitro.viewer import recording as layouts

RECORDINGS = (".rlpr", ".rlpr.zst")


class ScoreError(ValueError):
    """A game cannot be scored; the message says why in plain English."""


def _store_file(source: Path) -> Path:
    """The converted replay a `.replay` file stands for, from the store (fitted conversions first)."""
    stem = source.name.removesuffix(".replay")
    found = replays.find_stored(stem)
    if found is not None:
        return found[0]
    raise ScoreError(
        f"{source.name} has not been converted yet: convert its folder first with  nitro convert-replays "
        f'"{source.parent}"'
    )


def score(config: NitroConfig, source: Path, seconds: float = 300.0, fps: int = 30) -> dict[str, Any]:
    """The viewer's page data of the game in `source`, scored with `config`'s rewards, at most `seconds`
    of it, `fps` frames a second (rounded so every decision falls on a frame)."""
    if not source.is_file():
        raise ScoreError(f"there is no file {source}")
    tick_skip = config.env.tick_skip
    wanted = 120 / min(max(fps, 1), 120)
    every = min((d for d in range(1, tick_skip + 1) if tick_skip % d == 0), key=lambda d: abs(d - wanted))
    # The bot's own settings, as one arena: only its rewards, timing and physics are read.
    settings = core_settings(probe_config(config, 0))
    ticks = max(1, round(seconds * 120))
    name = source.name
    try:
        if name.endswith(RECORDINGS):
            out = _core.score_recording(settings, every, ticks, path=source)
        else:
            stored = _store_file(source) if name.endswith(".replay") else source
            if not stored.name.endswith(replays.STORED):
                raise ScoreError(f"{name} is not a replay, a converted replay or a recording")
            arrays, players = replays.read_stored(stored)
            out = _core.score_recording(settings, every, ticks, columns=replays.core_columns(arrays, players))
    except ValueError as err:
        if isinstance(err, ScoreError):
            raise
        raise ScoreError(f"{name} cannot be scored: {err}") from None
    cars = len(out["teams"])
    frames = np.frombuffer(out["frames"], dtype="<f4").reshape(-1, layouts.width(cars))
    if not len(out["step_frames"]):
        raise ScoreError(f"{name} holds no stretch of play long enough to score")
    labels = numbered(out["labels"])
    pay = np.frombuffer(out["pay"], dtype="<f4").reshape(len(out["step_frames"]), len(labels), cars)
    data = layouts.data(
        frames,
        fps=120 / every,
        pads=[list(pad) for pad in out["pads"]],
        decisions=[],
        decisions_per_second=120 / tick_skip,
        rewards={"labels": labels, "frames": list(out["step_frames"]), "pay": layouts.pack_floats(pay)},
    )
    data["scored"] = True  # the cars are a recorded game's players, not the bot
    return data


def totals(data: dict[str, Any]) -> dict[str, list[float]]:
    """Per reward, what it paid each player over the whole game."""
    rewards = data["rewards"]
    cars = int(data["cars"])
    pay = layouts.unpack_floats(rewards["pay"]).reshape(-1, len(rewards["labels"]), cars)
    return {label: [float(v) for v in pay[:, r].sum(0)] for r, label in enumerate(rewards["labels"])}
