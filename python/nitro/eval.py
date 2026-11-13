"""`nitro eval`: how much better is a run's bot than another save or another run?

Two saved policies play each other under the first run's own settings (its start states, game endings
and team size), Blue against Orange, in every arena at once, sampled as in training unless a side is given
another pick rule (likeliest, as the RLBot bot plays, or a temperature). Each episode from a
kickoff to a goal or the time limit is one *point*; the result is Blue's share of the goals with a 95%
interval, so a difference is only called clear when the luck of a few hundred points cannot explain it.
No rating system: the question is the plain one after more training, "is it better than before?", and the
saves are the run's own (`runs/<name>/checkpoints`), the self-play pool's or another run's. The same seed
gives the same games. Both ways round (`play_both_ways`) each save plays Blue once and Orange once
from the same start states, so the start's lean to one colour cancels.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nitro.config import NitroConfig
from nitro.envs import EnvPool, probe_config
from nitro.evaluate import SAMPLED, check_pick, evaluate, pick_rule
from nitro.model import ActorCritic, build_networks, observation_room, play_settings
from nitro.runs import RunFolder, actions_problem
from nitro.training import resolve_device

_Z95 = 1.959964  # two-sided 95% normal quantile, for the Wilson interval
# A point's time limit when the bot's own games have none (terminals.max_episode_seconds = 0).
DEFAULT_POINT_SECONDS = 300.0


class EvalError(Exception):
    """A plain-English reason the match cannot be played."""


@dataclass(frozen=True)
class Side:
    """One side of the match: a run's save."""

    name: str
    agent_steps: int
    state: dict[str, Any]
    config: NitroConfig

    @property
    def label(self) -> str:
        return f"{self.name} at {self.agent_steps:,} agent steps"


@dataclass(frozen=True)
class MatchResult:
    points: int  # episodes played (one per arena episode)
    blue_goals: int
    orange_goals: int
    no_goal: int  # points that ended without a goal (the time limit, or no touch for a while)
    mean_seconds: float
    time_limit: float
    no_touch: float = 0.0  # terminals.no_touch_seconds of the match's games; 0: never
    game: str = ""  # `game_code` of the match's games
    limit_added: bool = False  # the bot's games have no time limit: the match gave each point one

    def limit_line(self) -> list[str]:
        """The line that says a time limit was added for this match, when it was."""
        if not self.limit_added:
            return []
        return [
            f"  The bot's own games have no time limit, so each point here ended after {self.time_limit:g} "
            "seconds of game time without a goal (--max-seconds sets another limit)."
        ]

    def no_goal_text(self, count: int) -> str:
        """`count` points that ended without a goal, and what ends such a point."""
        touch = f" or no touch for {self.no_touch:g} s" if self.no_touch > 0 else ""
        return f"{count} ended without a goal (the {self.time_limit:g} s limit{touch})"

    @property
    def decided(self) -> int:
        return self.blue_goals + self.orange_goals

    @property
    def blue_share(self) -> float:
        return self.blue_goals / self.decided if self.decided else math.nan

    def interval(self) -> tuple[float, float]:
        """Wilson 95% interval of Blue's share of the goals (NaN when no goal was scored)."""
        return wilson(self.blue_goals, self.decided)

    def reading(self) -> str:
        low, high = self.interval()
        if self.decided == 0:
            return "Nobody scored: nothing to compare; raise --games or check that the bots can score at all."
        if low > 0.5:
            return "Blue is clearly better: even allowing for luck, it takes more than half the goals."
        if high < 0.5:
            return "Orange is clearly better: even allowing for luck, it takes more than half the goals."
        return (
            "No clear difference with this many points: the interval includes an even split. More --games "
            "narrows it."
        )

    def lines(self, blue: str, orange: str) -> list[str]:
        low, high = self.interval()
        share = "-" if self.decided == 0 else f"{self.blue_share:.0%} (95% interval {low:.0%} to {high:.0%})"
        return [
            f"{blue} (Blue) vs {orange} (Orange): {self.points} points from kickoff.",
            f"  Blue scored {self.blue_goals}, Orange {self.orange_goals}; "
            f"{self.no_goal_text(self.no_goal)}. A point lasted {self.mean_seconds:.1f} s on average.",
            f"  Blue's share of the goals: {share}.",
            *self.limit_line(),
            f"  Reading: {self.reading()}",
        ]


def pick_line(pick: tuple[float, float]) -> str:
    """How each side picks its actions."""
    return f"Picking actions: Blue {pick_rule(pick[0])}, Orange {pick_rule(pick[1])}."


def randomness_line(entropy: list[float]) -> str:
    """Each policy's mean normalised entropy over its decisions, from `evaluate`'s `entropy`."""
    return (
        "  How random each policy was (0 = always one action, 1 = any allowed action alike): "
        f"Blue {entropy[0]:.2f}, Orange {entropy[1]:.2f}."
    )


def _check_pick(pick: tuple[float, float]) -> None:
    try:
        check_pick(pick)
    except ValueError:
        raise EvalError(
            f"A temperature must be 0 (likeliest) or a positive number (got {pick[0]:g} and {pick[1]:g})."
        ) from None


def wilson(successes: int, trials: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a share: trustworthy at small counts and near 0 or 1, unlike the plain
    normal interval."""
    if trials == 0:
        return (math.nan, math.nan)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarise(records: list[Any], time_limit: float, no_touch: float = 0.0) -> MatchResult:
    """Counts each arena episode once (its car 0, a Blue car): a goal for Blue, for Orange, or none."""
    points = [r for r in records if r.car == 0]
    return MatchResult(
        points=len(points),
        blue_goals=sum(1 for r in points if r.goal > 0),
        orange_goals=sum(1 for r in points if r.goal < 0),
        no_goal=sum(1 for r in points if r.goal == 0),
        mean_seconds=sum(r.seconds for r in points) / len(points) if points else math.nan,
        time_limit=time_limit,
        no_touch=no_touch,
    )


def load_side(folder: RunFolder, at: int | None, *, oldest: bool = False) -> Side:
    """A run's save: the one made at `at` agent steps, the oldest, or the newest."""
    if not folder.info.is_file():
        raise EvalError(f"There is no run called '{folder.root.name}' in {folder.root.parent}.")
    saves = folder.complete_checkpoints()
    if not saves:
        raise EvalError(
            f"{folder.root.name} has no save yet: start training it first (the first save comes after a few "
            "minutes)."
        )
    if at is not None:
        state = folder.load_checkpoint(at)
        if state is None:
            listed = ", ".join(f"{int(p.name):,}" for p in saves)
            raise EvalError(
                f"There is no save of {folder.root.name} at {at:,} agent steps. Its saves: {listed}."
            )
    else:
        state = torch.load(
            (saves[0] if oldest else saves[-1]) / "state.pt", map_location="cpu", weights_only=True
        )
    config = folder.load_settings(state)
    return Side(folder.root.name, int(state["agent_steps"]), state, config)


def _sizes(config: NitroConfig) -> tuple[int, int]:
    """Observation length and action count of `config`'s game, from one arena."""
    tensors = EnvPool(probe_config(config), pin_memory=False).groups[0]
    return tensors.obs.shape[1], tensors.mask.shape[1]


def _model(side: Side, obs_size: int, num_actions: int, device: torch.device) -> ActorCritic:
    """`side`'s networks for a game of `num_actions` actions."""
    weights = side.state["learner"]["model"]
    if problem := actions_problem(weights, side.label):
        raise EvalError(f"{problem[0].upper()}{problem[1:]}.")
    model = build_networks(side.config, obs_size, num_actions).to(device)
    try:
        model.load_state_dict(weights)
    except RuntimeError as err:
        raise EvalError(
            f"The network of {side.label} does not fit these games ({obs_size} observation values and "
            f"{num_actions} actions): it was trained on another kind of game or with other networks. {err}"
        ) from None
    return model


class _Flagged(nn.Module):
    """An attention network playing from a plain observation: each car's block gets the "present" flag
    the network reads before it is passed on. A plain observation has a place for every car and no empty
    ones, so every flag is 1 and the network sees exactly what a padded observation of the game shows."""

    def __init__(self, inner: nn.Module, own: int, block: int, cars: int) -> None:
        super().__init__()
        self.inner = inner
        self.own, self.block, self.cars = own, block, cars

    def forward(self, obs: torch.Tensor) -> Any:
        rows = obs.shape[0]
        cars = obs[:, self.own :].reshape(rows, self.cars, self.block)
        present = torch.ones(rows, self.cars, 1, dtype=obs.dtype, device=obs.device)
        flagged = torch.cat([cars, present], dim=2).reshape(rows, -1)
        return self.inner(torch.cat([obs[:, : self.own], flagged], dim=1))


class _Unflagged(nn.Module):
    """A plain network playing in an attention bot's game, whose observation has a place for every car of
    the game (no empty ones): each car's "present" flag (always 1 there) is left out before it reads it."""

    def __init__(self, inner: nn.Module, own: int, block: int, cars: int) -> None:
        super().__init__()
        self.inner = inner
        self.own, self.block, self.cars = own, block, cars

    def forward(self, obs: torch.Tensor) -> Any:
        rows = obs.shape[0]
        cars = obs[:, self.own :].reshape(rows, self.cars, self.block + 1)[:, :, : self.block]
        return self.inner(torch.cat([obs[:, : self.own], cars.reshape(rows, -1)], dim=1))


# The game settings two bots must share to play each other, in plain words.
_SAME_GAME = {
    "observation": "what the bot sees (the observation)",
    "obs_team_size": "how many cars a side it has places for",
    "team_size": "the game size",
    "opponents": "whether it plays against opponents",
    "tick_skip": "how often it decides (the tick skip)",
    "action_delay": "the action delay",
}


def _game_setting(config: NitroConfig, name: str) -> Any:
    return observation_room(config) if name == "obs_team_size" else getattr(config.env, name)


def _shown(config: NitroConfig, name: str) -> str:
    """A game setting's value as the differences show it."""
    value = _game_setting(config, name)
    if name == "obs_team_size":
        return str(value or config.env.team_size)
    if name == "team_size":
        return f"{value}v{value}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def same_game(blue: Side, orange: Side, *, any_tick_skip: bool = False) -> list[str]:
    """The game settings the two saves disagree on, in plain words (empty when they saw the same game);
    with `any_tick_skip`, tick skips that `decide_every` can bridge are not a difference. With an attention
    network on either side the places for cars are not one: it reads any number of them (`match_sides`
    gives it the other's, and `_player` lets it read a plain bot's game)."""
    names = [n for n in _SAME_GAME if not (any_tick_skip and n == "tick_skip")]
    if blue.config.model.uses_attention or orange.config.model.uses_attention:
        names.remove("obs_team_size")
    return [
        f"{_SAME_GAME[name]}: {_shown(blue.config, name)} against {_shown(orange.config, name)}"
        for name in names
        if _game_setting(blue.config, name) != _game_setting(orange.config, name)
    ]


def decide_every(blue: Side, orange: Side) -> tuple[int, tuple[int, int]]:
    """The match's tick skip (the smaller of the two) and how many of its steps go by between each
    side's decisions: a tick-skip-8 bot against a tick-skip-4 one plays every second step and holds its action
    between, as its bot program holds it for 8 ticks. The larger tick skip must be a multiple of the
    smaller, and both must use the same action_delay (the game applies one delay to every car)."""
    ours, theirs = blue.config.env.tick_skip, orange.config.env.tick_skip
    small, large = min(ours, theirs), max(ours, theirs)
    if large % small:
        raise EvalError(
            f"These bots cannot share a game: {blue.label} decides every {ours} ticks and {orange.label} "
            f"every {theirs}, and one must be a multiple of the other."
        )
    delays = blue.config.env.action_delay, orange.config.env.action_delay
    if ours != theirs and delays[0] != delays[1]:
        raise EvalError(
            f"These bots cannot share a game: they decide at different rhythms, and {blue.label} waits "
            f"{delays[0]} ticks before acting while {orange.label} waits {delays[1]}. Bots with different "
            "tick skips can only meet with the same action delay."
        )
    return small, (ours // small, theirs // small)


def match_sides(blue: Side, orange: Side, team_size: int | None = None) -> tuple[Side, Side]:
    """Both sides' settings for the match: at `team_size` a side when given (each must be able to play it,
    `dashboard.bots.watch_sizes`), else at the size both trained at. Two attention bots read any number of
    cars, so they meet in the larger of their rooms whatever rooms they trained with; trained at different
    sizes, they meet only at a `team_size` chosen (how well a bot plays beyond its size is not measured)."""
    from nitro.dashboard.bots import watch_sizes

    size = blue.config.env.team_size if team_size is None else team_size
    any_size = blue.config.model.uses_attention and orange.config.model.uses_attention
    if any_size and team_size is None and orange.config.env.team_size != size:
        theirs = orange.config.env.team_size
        raise EvalError(
            f"The two bots trained at different sizes ({blue.label}: {size}v{size}; {orange.label}: "
            f"{theirs}v{theirs}). Both have attention networks, which play any size: choose the match's size "
            f"with --team-size (for example --team-size {size})."
        )
    sides = []
    for side in (blue, orange):
        config = side.config
        if team_size is not None:
            if size not in watch_sizes(config):
                raise EvalError(
                    f"{side.label} cannot play {size}v{size}: it reads {config.env.team_size}v"
                    f"{config.env.team_size} games only. A bot that sees room for more cars plays smaller "
                    "games too, and an attention network plays every size."
                )
            config = play_settings(config, size)
        sides.append(dataclasses.replace(side, config=config))
    rooms = [observation_room(side.config) for side in sides]
    attention = [side.config.model.uses_attention for side in sides]
    if any_size:
        room = max(rooms)
    elif attention.count(True) == 1:
        # The attention network reads the plain one's places; with none written, a place for each car of
        # the game, so the plain one can read the attention one's game too (`_player`).
        room = rooms[attention.index(False)] or sides[attention.index(True)].config.env.team_size
    else:
        room = None
    if room is not None:
        sides = [
            dataclasses.replace(
                side,
                config=dataclasses.replace(
                    side.config, env=dataclasses.replace(side.config.env, obs_team_size=room)
                ),
            )
            if side.config.model.uses_attention
            else side
            for side in sides
        ]
    return sides[0], sides[1]


def play(
    blue: Side,
    orange: Side,
    games: int,
    seed: int,
    team_size: int | None = None,
    max_seconds: float | None = None,
    pick: tuple[float, float] = SAMPLED,
    entropy: list[float] | None = None,
    game: NitroConfig | None = None,
) -> MatchResult:
    """`games` points (rounded up to whole arena rounds) under Blue's settings, Blue against Orange, at
    `team_size` a side when given (`match_sides`); with `max_seconds`, a point that lasts longer ends
    with no goal (for this match only: a run whose games have no time limit can be evaluated). `pick`
    and `entropy`: each side's temperature and randomness, as `evaluate`'s. With `game` (one side's
    settings after `match_sides`) the games are its instead of Blue's: its arenas, start states, game
    endings, rewards, folders and device; each side's networks still follow its own settings."""
    _check_pick(pick)
    blue, orange = match_sides(blue, orange, team_size)
    if differences := same_game(blue, orange, any_tick_skip=True):
        raise EvalError(
            f"{blue.label} and {orange.label} read the game differently, so they cannot play each other: "
            f"{'; '.join(differences)}."
        )
    tick_skip, every = decide_every(blue, orange)
    base = blue.config if game is None else game
    config = dataclasses.replace(base, env=dataclasses.replace(base.env, tick_skip=tick_skip))
    limit_added = False
    if max_seconds is not None:
        if not max_seconds > 0:
            raise EvalError(f"The time limit a point (--max-seconds) must be above 0, not {max_seconds:g}.")
    elif config.terminals.max_episode_seconds <= 0:
        max_seconds, limit_added = DEFAULT_POINT_SECONDS, True  # else a point might never end
    if max_seconds is not None:
        config = dataclasses.replace(
            config, terminals=dataclasses.replace(config.terminals, max_episode_seconds=max_seconds)
        )
    if not config.env.opponents:
        raise EvalError(
            f"{blue.name} trains without opponents, so there is nobody to play against. Evaluate a bot that "
            "plays matches."
        )
    device = resolve_device(config.run)
    fp16 = config.run.precision == "fp16" and device.type == "cuda"
    obs_size, num_actions = _sizes(config)
    blue_model = _player(blue, config, obs_size, num_actions, device)
    orange_model = _player(orange, config, obs_size, num_actions, device)
    arenas = min(config.env.num_arenas, max(games, 1))
    match_config = dataclasses.replace(
        config, env=dataclasses.replace(config.env, num_arenas=arenas, groups=min(config.env.groups, arenas))
    )
    episodes_per_arena = math.ceil(games / arenas)
    records = evaluate(
        blue_model,
        match_config,
        seed,
        episodes_per_arena,
        device,
        fp16,
        orange_model,
        decide_every=every,
        pick=pick,
        entropy=entropy,
    )
    terminals = config.terminals
    result = summarise(records, terminals.max_episode_seconds, terminals.no_touch_seconds)
    return dataclasses.replace(result, game=game_code(match_config), limit_added=limit_added)


def _player(
    side: Side, game: NitroConfig, obs_size: int, num_actions: int, device: torch.device
) -> nn.Module:
    """`side`'s networks for the match's games (`game`, which produce `obs_size` observation values). An
    attention network and a plain one without places for cars read each other's games (`match_sides`
    leaves a padded game a place for each car and no more): the attention network with every car marked
    present (`_Flagged`), the plain one without the marks (`_Unflagged`)."""
    from nitro import _core

    env = game.env
    flags = observation_room(game) > 0
    attention = side.config.model.uses_attention
    if attention == flags or (not attention and observation_room(side.config) > 0):
        return _model(side, obs_size, num_actions, device)
    _, own, block, teammates, opponents, _ = _core.observation_layout_of(
        env.observation, env.team_size, env.opponents, 0
    )
    cars = teammates + opponents
    if attention:
        return _Flagged(_model(side, own + cars * (block + 1), num_actions, device), own, block, cars).to(
            device
        )
    return _Unflagged(_model(side, own + cars * block, num_actions, device), own, block, cars).to(device)


def game_code(config: NitroConfig) -> str:
    """A short code of the settings that shape a match's games: with the same code and seed, the
    same start states; with another code, other games, so two shares are not like for like."""
    env = config.env
    shape = {
        "env": [env.team_size, env.num_arenas, env.groups, env.tick_skip, env.action_delay],
        "terminals": dataclasses.asdict(config.terminals),
        "state_setters": [dataclasses.asdict(setter) for setter in config.state_setters],
    }
    text = json.dumps(shape, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class BothWays:
    """Side A against side B both ways round from the same start states, from A's view: A as Blue
    first, then as Orange, so the start's lean to one colour cancels in the pooled share."""

    as_blue: MatchResult
    as_orange: MatchResult

    @property
    def goals_for(self) -> int:
        return self.as_blue.blue_goals + self.as_orange.orange_goals

    @property
    def goals_against(self) -> int:
        return self.as_blue.orange_goals + self.as_orange.blue_goals

    @property
    def decided(self) -> int:
        return self.goals_for + self.goals_against

    @property
    def timeouts(self) -> int:
        return self.as_blue.no_goal + self.as_orange.no_goal

    @property
    def points_each_way(self) -> int:
        """Points played each way: the points asked for, rounded up to whole rounds of the arenas."""
        return min(self.as_blue.points, self.as_orange.points)

    @property
    def game(self) -> str:
        return self.as_blue.game

    @property
    def share(self) -> float:
        return self.goals_for / self.decided if self.decided else math.nan

    def interval(self) -> tuple[float, float]:
        """Wilson 95% interval of A's share of the goals of both legs together."""
        return wilson(self.goals_for, self.decided)

    def reading(self) -> str:
        low, high = self.interval()
        if self.decided == 0:
            return "Nobody scored: nothing to compare; raise --games or check that the bots can score at all."
        if low > 0.5:
            return "The first is clearly better: even allowing for luck, it takes more than half the goals."
        if high < 0.5:
            return "The second is clearly better: even allowing for luck, it takes more than half the goals."
        return (
            "No clear difference with this many points: the interval includes an even split. More --games "
            "narrows it."
        )

    def lines(self, a: str, b: str) -> list[str]:
        low, high = self.interval()
        legs = self.as_blue, self.as_orange
        share = "-" if self.decided == 0 else f"{self.share:.0%} (95% interval {low:.0%} to {high:.0%})"
        blue = "-" if not legs[0].decided else f"{legs[0].blue_share:.0%}"
        orange = "-" if not legs[1].decided else f"{1 - legs[1].blue_share:.0%}"
        return [
            f"{a} vs {b}, both ways round from the same starts: {legs[0].points} points as Blue, "
            f"{legs[1].points} as Orange.",
            f"  As Blue it scored {legs[0].blue_goals} and let in {legs[0].orange_goals}; as Orange it "
            f"scored {legs[1].orange_goals} and let in {legs[1].blue_goals}; "
            f"{legs[0].no_goal_text(self.timeouts)}.",
            f"  Its share of the goals: as Blue {blue}, as Orange {orange}; both ways {share}.",
            *legs[0].limit_line(),
            f"  Reading: {self.reading()}",
        ]


def play_both_ways(
    a: Side,
    b: Side,
    games_each: int,
    seed: int,
    team_size: int | None = None,
    max_seconds: float | None = None,
) -> BothWays:
    """`games_each` points with A as Blue, then as many with B as Blue, both under A's games and on the same
    seed: the same start states with the colours swapped. Tick skips mix as in `play`."""
    game = match_sides(a, b, team_size)[0].config
    as_blue = play(a, b, games_each, seed, team_size, max_seconds, game=game)
    as_orange = play(b, a, games_each, seed, team_size, max_seconds, game=game)
    return BothWays(as_blue, as_orange)


EVAL_COLUMNS = (
    "when_utc", "agent_steps", "opponent", "opponent_steps", "points_each_way", "seed", "game", "as_blue_for",
    "as_blue_against", "as_orange_for", "as_orange_against", "timeouts", "share", "low", "high",
)  # fmt: skip


def eval_row(a: Side, b: Side, result: BothWays, seed: int) -> dict[str, str | int]:
    """A both-ways result as a row of `EVAL_COLUMNS` (A's view): the points played, and `game_code`."""
    low, high = result.interval()

    def plain(value: float) -> str:
        return "" if math.isnan(value) else f"{value:.4f}"

    return {
        "when_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent_steps": a.agent_steps,
        "opponent": b.name,
        "opponent_steps": b.agent_steps,
        "points_each_way": result.points_each_way,
        "seed": seed,
        "game": result.game,
        "as_blue_for": result.as_blue.blue_goals,
        "as_blue_against": result.as_blue.orange_goals,
        "as_orange_for": result.as_orange.orange_goals,
        "as_orange_against": result.as_orange.blue_goals,
        "timeouts": result.timeouts,
        "share": plain(result.share),
        "low": plain(low),
        "high": plain(high),
    }


def append_row(path: Path, columns: tuple[str, ...], row: dict[str, str | int]) -> None:
    """Adds `row` to the CSV file `path`, writing the header first when the file is new."""
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)
