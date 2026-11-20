"""`nitro gauntlet`: a report card of a save against fixed opponents.

The opponents are scripted players written for NitroLearn and run inside the simulator
(`nitro_core::scripted`): a ball-chaser, a goalie that shadows between the ball and its goal, a kickoff
rusher and an aerial-minded chaser (plus `idle`, which does nothing, for checks). They never learn, so
"beats the chaser 97% of the goals" means the same thing at every save and for every bot of the same game
size. Each match is a run of points from kickoff, each ending at a goal, the bot's own no-touch time or
POINT_SECONDS; the share of the goals comes with a Wilson 95% interval, and the same seed plays the same
games. For a player's own reading, not proof that one setting is better than another.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from nitro.config import NitroConfig
from nitro.eval import EvalError, MatchResult, Side, _player, _sizes, summarise
from nitro.evaluate import evaluate
from nitro.training import resolve_device

# Scripted opponents, weakest first as written (the measured order is the gauntlet's to show).
SCRIPTED = ("idle", "chaser", "goalie", "kickoff", "aerial")
DEFAULT_OPPONENTS = ("chaser", "goalie", "kickoff", "aerial")
POINT_SECONDS = 60.0


def gauntlet_game(config: NitroConfig, point_seconds: float = POINT_SECONDS) -> NitroConfig:
    """`config`'s game played from kickoffs, a point ending at a goal, the bot's no-touch time or
    `point_seconds`."""
    from nitro.config.state_setters import Kickoff

    seconds = config.terminals.max_episode_seconds
    terminals = dataclasses.replace(
        config.terminals,
        max_episode_seconds=min(seconds, point_seconds) if seconds > 0 else point_seconds,
        truncate_on_ball_landing=0.0,
    )
    return dataclasses.replace(config, state_setters=(Kickoff(),), terminals=terminals)


def _arenas(config: NitroConfig, games: int) -> tuple[NitroConfig, int]:
    arenas = max(1, min(config.env.num_arenas, games))
    env = dataclasses.replace(config.env, num_arenas=arenas, groups=min(config.env.groups, arenas))
    return dataclasses.replace(config, env=env), math.ceil(games / arenas)


def play_scripted(side: Side, opponent: str, games: int, seed: int) -> MatchResult:
    """`games` points (rounded up to whole arena rounds) of `side` (Blue) against the scripted `opponent`
    driving every Orange car, in `side`'s game from kickoffs."""
    if opponent not in SCRIPTED:
        raise EvalError(f"No scripted opponent called '{opponent}' (choose from {', '.join(SCRIPTED)}).")
    if not side.config.env.opponents:
        raise EvalError(
            f"'{side.name}' trains without opponents (env.opponents = false), so it cannot play a match."
        )
    config, per_arena = _arenas(gauntlet_game(side.config), games)
    device = resolve_device(config.run)
    fp16 = config.run.precision == "fp16" and device.type == "cuda"
    obs_size, num_actions = _sizes(config)
    model = _player(side, config, obs_size, num_actions, device)
    records = evaluate(model, config, seed, per_arena, device, fp16, scripted=[(1, opponent)])
    terminals = config.terminals
    return summarise(records, terminals.max_episode_seconds, terminals.no_touch_seconds)


class _Unused(torch.nn.Module):
    """A stand-in policy where scripts drive every car: its actions are ignored."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(obs.shape[0], self.num_actions), torch.zeros(obs.shape[0])


def play_scripts(config: NitroConfig, blue: str, orange: str, games: int, seed: int) -> MatchResult:
    """Two scripted players against each other in `config`'s game (checks of the scale). The scripts have
    no randomness, so from kickoffs they would play the same few games over and over; each point here starts
    from a random state instead."""
    from nitro import _core
    from nitro.config.state_setters import RandomState

    game = dataclasses.replace(gauntlet_game(config), state_setters=(RandomState(),))
    config, per_arena = _arenas(game, games)
    records = evaluate(
        cast(Any, _Unused(_core.num_actions())),
        config,
        seed,
        per_arena,
        torch.device("cpu"),
        False,
        scripted=[(0, blue), (1, orange)],
    )
    terminals = config.terminals
    return summarise(records, terminals.max_episode_seconds, terminals.no_touch_seconds)


@dataclass(frozen=True)
class CardLine:
    opponent: str
    result: MatchResult

    def entry(self) -> dict[str, Any]:
        low, high = self.result.interval()
        r = self.result
        return {
            "opponent": self.opponent,
            "points": r.points,
            "won": r.blue_goals,
            "lost": r.orange_goals,
            "draws": r.no_goal,
            "share": None if r.decided == 0 else round(r.blue_share, 4),
            "low": None if r.decided == 0 else round(low, 4),
            "high": None if r.decided == 0 else round(high, 4),
        }


def run_gauntlet(
    side: Side, opponents: tuple[str, ...] = DEFAULT_OPPONENTS, games: int = 200, seed: int = 1_000_001
) -> list[CardLine]:
    """`side` against each opponent in turn, each with its own seed from `seed` (same seed, same games)."""
    return [
        CardLine(name, play_scripted(side, name, games, seed + 7919 * number))
        for number, name in enumerate(opponents)
    ]


def append_report(path: Path, agent_steps: int, lines: list[CardLine], seed: int, games: int) -> None:
    """Adds this save's card to the run's gauntlet.json (newest last), replacing an earlier card of the same
    save, seed and games."""
    try:
        cards = json.loads(path.read_text(encoding="utf-8"))
        cards = [x for x in cards if isinstance(x, dict)] if isinstance(cards, list) else []
    except (OSError, ValueError):
        cards = []
    cards = [
        c
        for c in cards
        if not (c.get("agent_steps") == agent_steps and c.get("seed") == seed and c.get("games") == games)
    ]
    cards.append(
        {
            "when": time.strftime("%Y-%m-%d %H:%M"),
            "agent_steps": agent_steps,
            "seed": seed,
            "games": games,
            "lines": [line.entry() for line in lines],
        }
    )
    cards.sort(key=lambda c: int(c.get("agent_steps", 0)))
    partial = path.with_name(path.name + ".partial")  # never a half-written report
    partial.write_text(json.dumps(cards, indent=1), encoding="utf-8")
    partial.replace(path)
