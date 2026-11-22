"""Multi-model bots: a learned selector over several trained bots, its models (`[selector]` in the settings).
`nitro export` writes one RLBot bot of the selector and its models.

The selector sits between the trainer and the games: to the trainer it is an ordinary set of games whose
actions are which model drives each car, so collection, the rollout, GAE and PPO are unchanged. Per car it
sees the game's observation plus which model drives now (one-hot), how long it has driven, whether the
selector may choose again, every model's own value estimate of the moment and, with `goal_heads`, each model's
guess of who scores next. Each step every model samples an action for every car from its own view of the game
(its own observation, room and game size, which the core builds) and the chosen one is played; the
observation's previous action is therefore always the control applied, whoever chose it.

A choice holds for `min_hold_seconds`: until then the selector's action mask allows only the active model, so
those steps carry no policy gradient (one allowed action has log-probability 0) while the value network still
learns from every step. That is call-and-return options on the ordinary per-step PPO. A switch costs
`switch_cost` reward; a model below its `min_share` earns a bonus that grows while it stays below. With
`train_models` or `goal_heads` every step's driver is recorded for `nitro.selector_learning`.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from nitro.config.schema import ModelConfig, NitroConfig, SelectorConfig
from nitro.envs import GroupTensors
from nitro.model import ActorCritic, build_networks, config_parts, observation_layout, observation_room
from nitro.ppo import likeliest, masked_log_probs
from nitro.runs import FIXED_SETTINGS, actions_problem, all_settings, numbered, saved_settings
from nitro.transfer import TransferError, find_save

# What a model must share with the multi-model bot: it decides at the same rate.
_SAME = ("env.tick_skip",)
# How a model sees the game; one that differs gets a view of its own.
_VIEW = (
    "env.observation", "env.obs_team_size", "env.team_size", "env.opponents", "env.action_masks",
    "env.ball_prediction",
)  # fmt: skip
_RECORDED = (*_SAME, *_VIEW, "env.action_delay")
# How long a hold the selector's observation tells apart (seconds); longer reads as this.
HELD_CAP_SECONDS = 5.0
# A goal head's classes: no goal within its horizon, the car's team scores, the other team scores.
GOAL_CLASSES = 3


class SelectorError(ValueError):
    """A model cannot be used in this run; the message says why."""


@dataclass(frozen=True)
class Specialist:
    """A model the selector can hand a car to, as a multi-model run's saves keep it."""

    name: str  # its metrics label: the run's name
    label: str  # "touch at 10,000,000 agent steps"
    model: dict[str, Any]  # its [model] settings, to rebuild its networks
    weights: dict[str, torch.Tensor] = field(repr=False)
    game: dict[str, Any] = field(default_factory=dict)  # the game settings it learned with (_RECORDED)
    # Its weights when this run began, which `train_models` pulls it back toward; None while never trained.
    start: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    # Its value network's units in its own run (PopArt's statistics); None when that run kept none.
    value_norm: dict[str, torch.Tensor] | None = field(default=None, repr=False)


def extra_size(specialists: int, goal_heads: bool = False) -> int:
    """Numbers the selector's observation adds to the game's: the active one (one-hot), the time held,
    whether it may choose now, each model's value and, with goal heads, each model's two goal chances."""
    return 2 * specialists + 2 + (2 * specialists if goal_heads else 0)


def hold_steps(settings: SelectorConfig, tick_skip: int) -> int:
    """Steps a choice holds at least: `min_hold_seconds`, rounded up; 1 lets it choose every step."""
    return max(1, math.ceil(settings.min_hold_seconds * 120 / tick_skip - 1e-9))


def extras_at(config: NitroConfig, game_size: int) -> int:
    """Where the selector's own numbers sit in its observation: after the game's for a plain network, after
    the own part for an attention network, whose layout then reads them as the own car's."""
    return observation_layout(config)[0] if config.model.uses_attention else game_size


def network(config: NitroConfig, game_size: int, choices: int) -> ActorCritic:
    """The selector's networks for a game observation of `game_size` numbers and `choices` models."""
    extra = extra_size(choices, config.selector.goal_heads)
    if not config.model.uses_attention:
        return build_networks(config, game_size + extra, choices)
    own, block, teammates, opponents = observation_layout(config)
    reads_ball = config.model.encoder in ("attention_communication", "transformer")
    parts = config_parts(config) if reads_ball else None
    layout = (own + extra, block, teammates, opponents)
    return ActorCritic(game_size + extra, choices, config.model, layout, parts)


def from_model(name: str, model: ActorCritic, config: NitroConfig, label: str = "") -> Specialist:
    """A trained network as a model (the selector benchmarks hand theirs over directly)."""
    weights = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
    flat = _game_of(config)
    settings = {f.name: flat[f"model.{f.name}"] for f in dataclasses.fields(ModelConfig)}
    return Specialist(name, label or name, settings, weights, {path: flat[path] for path in _RECORDED})


def pseudo(name: str, config: NitroConfig) -> Specialist:
    """A bot as a model, from its settings alone (no weights): enough to tell whether it can drive in a
    multi-model bot (`problems_of`), as the dashboard's form checks before anything is loaded."""
    flat = _game_of(config)
    settings = {f.name: flat[f"model.{f.name}"] for f in dataclasses.fields(ModelConfig)}
    return Specialist(_label(name), name, settings, {}, {path: flat[path] for path in _RECORDED})


def _game_of(config: NitroConfig) -> dict[str, Any]:
    """`all_settings`, with the observation room the networks really read (`observation_room`): an
    attention network with env.obs_team_size 0 reads its own team size's room."""
    return {**all_settings(config), "env.obs_team_size": observation_room(config)}


def load_specialists(config: NitroConfig, runs: Path) -> tuple[Specialist, ...]:
    """The newest saves of the runs `selector.specialists` names (or the saves it names), checked against
    this run's settings; empty when the selector is off."""
    found = []
    for name in config.selector.specialists:
        specialist = load_one(name, runs)
        if problems := problems_of(specialist, config):
            raise SelectorError(f"selector.specialists: {specialist.label}: " + "; ".join(problems))
        found.append(specialist)
    return tuple(found)


def load_one(name: str, runs: Path) -> Specialist:
    """The newest save of run `name` (or the save it names) as a model; SelectorError when it is none."""
    try:
        source = find_save(name, 0, runs, "selector.specialists")
    except TransferError as err:
        raise SelectorError(str(err)) from None
    state = source.state
    if "specialists" in state:
        raise SelectorError(f"selector.specialists: {source.label} is itself a multi-model bot")
    if "learner" not in state:
        raise SelectorError(f"selector.specialists: {source.label} holds no trained networks")
    game = {path: value for path in _RECORDED if (value := _recorded(state, path)) is not None}
    model = _model_settings(state)
    if game.get("env.obs_team_size") == 0 and model.get("encoder", "mlp") != "mlp":
        game["env.obs_team_size"] = _recorded(state, "env.team_size")  # its room (observation_room)
    learner = state["learner"]
    if problem := actions_problem(learner["model"], source.label):
        raise SelectorError(f"selector.specialists: {problem}")
    return Specialist(
        _label(source.run),
        source.label,
        model,
        learner["model"],
        game,
        value_norm=learner.get("value_norm"),
    )


def problems_of(specialist: Specialist, config: NitroConfig) -> list[str]:
    """Why `specialist` cannot drive in `config`'s games, in plain words (empty: it can)."""
    mine = _game_of(config)
    problems = [
        f"it decides every {theirs} ticks and this bot every {mine[path]} (env.tick_skip must match)"
        for path in _SAME
        if (theirs := specialist.game.get(path, mine[path])) != mine[path]
    ]
    delay = specialist.game.get("env.action_delay", mine["env.action_delay"])
    if delay != mine["env.action_delay"]:
        problems.append(
            f"its actions reached the game {delay} ticks after it decided and this bot's "
            f"{mine['env.action_delay']} (env.action_delay must match)"
        )
    view = view_of(specialist, config)
    if view is not None and (problem := _core_view_problem(view, config)):
        problems.append(problem)
    return problems


def _core_view_problem(view: dict[str, Any], config: NitroConfig) -> str | None:
    from nitro import _core

    try:
        _core.observation_layout_of(
            view["observation"],
            view["team_size"],
            view["opponents"],
            view["obs_team_size"],
            len(view.get("ball_prediction", ())),
        )
    except ValueError as err:
        return f"its observation cannot be built: {err}"
    env = config.env
    if view["team_size"] > env.team_size or (view["opponents"] and not env.opponents):
        theirs = f"{view['team_size']}v{view['team_size']}" if view["opponents"] else "alone"
        ours = f"{env.team_size}v{env.team_size}" if env.opponents else "alone"
        return (
            f"it learned to see games {theirs} and this bot plays {ours}, which shows it less than it needs"
        )
    return None


def view_of(specialist: Specialist, config: NitroConfig) -> dict[str, Any] | None:
    """The view of the game `specialist` needs (the core's `views` entry), or None when it sees the game as
    the run's own observation does. A save that did not record a setting saw the run's."""
    mine = _game_of(config)
    theirs = {path: specialist.game.get(path, mine[path]) for path in _VIEW}
    if theirs == {path: mine[path] for path in _VIEW}:
        return None
    times = [float(t) for t in theirs["env.ball_prediction"]]
    return {
        "observation": theirs["env.observation"],
        "team_size": int(theirs["env.team_size"]),
        "opponents": bool(theirs["env.opponents"]),
        "obs_team_size": int(theirs["env.obs_team_size"]),
        "action_masks": theirs["env.action_masks"],
        **({"ball_prediction": times} if times else {}),  # its own ball prediction
    }


def views(specialists: Sequence[Specialist], config: NitroConfig) -> list[dict[str, Any]]:
    """The distinct views the models need, in the order they first appear (the core's `views`)."""
    found: list[dict[str, Any]] = []
    for specialist in specialists:
        view = view_of(specialist, config)
        if view is not None and view not in found:
            found.append(view)
    return found


def _recorded(state: dict[str, Any], path: str) -> Any:
    """A setting as `state` recorded it, as plain JSON; None when that save did not record it."""
    flat = state.get("all_settings", {})
    if path in flat:
        return flat[path]
    if path in FIXED_SETTINGS:
        value = saved_settings(state).get(path) if "settings" in state else None
        return [list(v) if isinstance(v, tuple) else v for v in value] if isinstance(value, tuple) else value
    return None


def _model_settings(state: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for f in dataclasses.fields(ModelConfig):
        value = _recorded(state, f"model.{f.name}")
        if value is not None:
            out[f.name] = value
    return out


def _label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "bot"


def labels(specialists: Sequence[Specialist]) -> list[str]:
    """One metrics label per model; a repeated name gets a number from its second appearance on."""
    return numbered([specialist.name for specialist in specialists])


def to_state(specialists: Sequence[Specialist]) -> list[dict[str, Any]]:
    """The models as a save keeps them (plain values and tensors only)."""
    return [
        {
            "name": s.name,
            "label": s.label,
            "model": s.model,
            "weights": s.weights,
            "game": s.game,
            **({"start": s.start} if s.start is not None else {}),
            **({"value_norm": s.value_norm} if s.value_norm is not None else {}),
        }
        for s in specialists
    ]


def from_state(saved: Sequence[dict[str, Any]]) -> tuple[Specialist, ...]:
    return tuple(Specialist(**entry) for entry in saved)


def model_config(specialist: Specialist) -> ModelConfig:
    names = {f.name for f in dataclasses.fields(ModelConfig)}
    values: dict[str, Any] = {
        k: tuple(v) if isinstance(v, list) else v for k, v in specialist.model.items() if k in names
    }
    return dataclasses.replace(ModelConfig(), **values)


def _own_config(specialist: Specialist, config: NitroConfig) -> NitroConfig:
    """`config` with the model's own [model] and the game it learned to see, to build its networks."""
    view = view_of(specialist, config) or {}
    env = dataclasses.replace(
        config.env,
        observation=view.get("observation", config.env.observation),
        team_size=view.get("team_size", config.env.team_size),
        opponents=view.get("opponents", config.env.opponents),
        obs_team_size=view.get("obs_team_size", observation_room(config)),
        ball_prediction=tuple(view.get("ball_prediction", ())) if view else config.env.ball_prediction,
    )
    return dataclasses.replace(config, model=model_config(specialist), env=env)


def networks(
    specialists: Sequence[Specialist],
    config: NitroConfig,
    obs_size: int,
    num_actions: int,
    device: torch.device,
    *,
    trainable: bool = False,
    view_sizes: Sequence[int] = (),
) -> list[ActorCritic]:
    """The models' networks on `device`, each for its own view (`view_sizes`: the core's sizes of
    `views(specialists, config)`; `obs_size` for the run's own) and `num_actions` outputs; frozen unless
    `trainable`."""
    built = []
    found = views(specialists, config)
    for specialist in specialists:
        if problems := problems_of(specialist, config):
            raise SelectorError(f"selector.specialists: {specialist.label}: " + "; ".join(problems))
        view = view_of(specialist, config)
        size = obs_size if view is None else view_sizes[found.index(view)] if view_sizes else _view_size(view)
        try:
            model = build_networks(_own_config(specialist, config), size, num_actions)
            model.load_state_dict(specialist.weights)
        except (RuntimeError, ValueError) as err:
            raise SelectorError(
                f"selector.specialists: {specialist.label} does not fit its observation ({size} numbers) "
                f"and {num_actions} actions: {err}"
            ) from None
        model.requires_grad_(trainable)
        built.append(model.eval().to(device))
    return built


def _view_size(view: dict[str, Any]) -> int:
    from nitro import _core

    size: int = _core.observation_layout_of(
        view["observation"],
        view["team_size"],
        view["opponents"],
        view["obs_team_size"],
        len(view.get("ball_prediction", ())),
    )[0]
    return size


def not_yet(config: NitroConfig, doing: str) -> str | None:
    """Why a multi-model run cannot be `doing` ("used as a teacher") yet, or None for any other run."""
    if not config.selector.specialists:
        return None
    return f"it is a multi-model bot (selector.specialists), which cannot be {doing} yet"


class Policy(Protocol):
    """A model as the selector runs it: logits and a normalised value per observation."""

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def normalised_value(self, obs: torch.Tensor) -> torch.Tensor: ...


class Games(Protocol):
    """The games a selector plays (`nitro.envs.EnvPool`, or a test's)."""

    @property
    def groups(self) -> tuple[GroupTensors, ...]: ...

    def step(self, group: int) -> None: ...

    def wait(self, group: int) -> None: ...


@dataclass
class _Cars:
    """One group's selector state, per car, on the device."""

    active: torch.Tensor  # int64: which model drives; -1 = none yet (a new episode)
    held: torch.Tensor  # int64: steps it has driven
    candidates: torch.Tensor  # [models, agents] int64: each model's action for the next step
    log_probs: torch.Tensor  # [models, agents] float32: each candidate's log-probability under its model
    values: torch.Tensor  # [agents, models] float32: each model's normalised value of the state now
    switched: torch.Tensor  # bool, on the CPU: the last step handed the car to another model
    driving: torch.Tensor  # int64: who drove the last step (-1 before the first)


@dataclass
class SelectorStats:
    driven: list[int]  # car-steps each model drove (the learner's own cars)
    switches: int


@dataclass
class Recorded:
    """One collection's steps as the models saw them ([steps, agents, ...] on the device), for training the
    models and their goal heads (`nitro.selector_learning`). `obs` and `mask` are the driving model's own,
    padded to the widest view; the values are every model's, normalised."""

    obs: torch.Tensor  # [T, N, widest] float32
    mask: torch.Tensor  # [T, N, actions] bool
    actions: torch.Tensor  # [T, N] int64
    log_probs: torch.Tensor  # [T, N] float32
    driver: torch.Tensor  # [T, N] int64
    values: torch.Tensor  # [T, N, models] float32
    rewards: torch.Tensor  # [T, N] float32: the game's, without the selector's switch cost and bonus
    terminal: torch.Tensor  # [T, N] bool
    truncated: torch.Tensor  # [T, N] bool
    truncated_values: torch.Tensor  # [T, N, models] float32: each model's value where a truncated step ended
    goals: torch.Tensor  # [T, N] int8: 1 the car's team scored at step t, -1 the other team, 0 neither
    episode: torch.Tensor  # [T, N] int64
    episode_step: torch.Tensor  # [T, N] int32
    bootstrap_values: torch.Tensor  # [N, models] float32: after the last step
    steps: list[int] = field(default_factory=list)  # per group, steps recorded so far


class GoalHeads(nn.Module):
    """Per model, a small network on its own observation that guesses who scores next within a horizon
    (no goal, the car's team, the other team), trained on this run's games where that model drove."""

    def __init__(self, sizes: Sequence[int], hidden: int = 128) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, GOAL_CLASSES))
            for size in sizes
        )  # fmt: skip

    def forward(self, model: int, obs: torch.Tensor) -> torch.Tensor:
        """Model `model`'s goal logits for its observations `obs` ([n, its size])."""
        logits: torch.Tensor = self.heads[model](obs.float())
        return logits


class ValueStats:
    """Each model's value estimates standardised over this run's states (`selector.standardise_values`): a
    running mean and spread per model, updated once per collection so a collection sees one scale."""

    def __init__(self, count: int, device: torch.device) -> None:
        self.mean = torch.zeros(count, device=device)
        self.std = torch.ones(count, device=device)
        self.started = False
        self._sums = torch.zeros(3, count, device=device)  # n, sum, sum of squares

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        if not self.started:  # the first states seen set the first scale
            self._take(values)
            self.update()
        return (values - self.mean) / self.std

    def add(self, values: torch.Tensor) -> None:
        self._take(values)

    def _take(self, values: torch.Tensor) -> None:
        self._sums[0] += values.shape[0]
        self._sums[1] += values.sum(0)
        self._sums[2] += values.pow(2).sum(0)

    def update(self, rate: float = 0.1) -> None:
        n = self._sums[0].clamp_min(1.0)
        if float(self._sums[0][0]) < 2:
            return
        mean = self._sums[1] / n
        std = (self._sums[2] / n - mean.pow(2)).clamp_min(1e-8).sqrt().clamp_min(1e-3)
        share = 1.0 if not self.started else rate
        self.mean = self.mean + share * (mean - self.mean)
        self.std = self.std + share * (std - self.std)
        self.started = True
        self._sums.zero_()

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "started": self.started}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = state["mean"].to(self.mean.device)
        self.std = state["std"].to(self.std.device)
        self.started = bool(state["started"])


class ShareBonus:
    """Steers the selector toward each model's `min_share`: per model a bonus paid every step it
    drives, in units of the run's typical reward per step, from a proportional-integral controller on the
    last collection's relative shortfall e = (target - share) / target: bonus = integral + PUSH x rate x
    typical x e, the integral growing by rate x typical x e a collection (a Lagrange multiplier for
    "share >= target"). An integral alone was too slow at a small rate and swung around the target at a large
    one (benchmarks/micro/share_bonus_toy.py). Never forces a choice."""

    PUSH = 6.0  # the proportional gain, in multiples of the rate
    CAP = 5.0  # the bonus and its integral are at most this many typical step rewards

    def __init__(self, targets: Sequence[float], rate: float, device: torch.device) -> None:
        self.targets = torch.tensor([t / 100 for t in targets], device=device)
        self.rate = rate
        self.integral = torch.zeros(len(targets), device=device)  # raw reward per step
        self.bonus = torch.zeros(len(targets), device=device)  # raw reward per step, paid
        self.share: torch.Tensor | None = None  # the last collection's

    def update(self, driven: Sequence[int], typical: float) -> None:
        total = sum(driven)
        if total == 0:
            return
        self.share = torch.tensor([n / total for n in driven], device=self.bonus.device)
        wanted = self.targets > 0
        # Relative shortfall, at most 1 either way: a share far above its target fades the bonus at the
        # pace a share of 0 raises it.
        shortfall = torch.where(wanted, (self.targets - self.share) / self.targets.clamp_min(1e-9), 0.0)
        shortfall = shortfall.clamp(-1.0, 1.0)
        typical = max(typical, 1e-6)
        cap = self.CAP * typical
        self.integral = torch.where(
            wanted, (self.integral + self.rate * typical * shortfall).clamp(0.0, cap), 0.0
        )
        push = self.PUSH * self.rate * typical * shortfall
        self.bonus = torch.where(wanted, (self.integral + push).clamp(0.0, cap), 0.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "bonus": self.bonus.cpu(),
            "integral": self.integral.cpu(),
            "share": None if self.share is None else self.share.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["bonus"].shape != self.bonus.shape:
            return  # another number of models: start afresh
        self.bonus = state["bonus"].to(self.bonus.device)
        self.integral = state["integral"].to(self.bonus.device)
        self.share = None if state["share"] is None else state["share"].to(self.bonus.device)


class SelectorEnvs:
    """The games as the selector sees them (see the module docstring). Same use as `EnvPool`: fill
    `groups[g].actions` (model numbers), `step(g)`, then `wait(g)` before reading the group again."""

    def __init__(
        self,
        envs: Games,
        models: Sequence[Policy],
        settings: SelectorConfig,
        tick_skip: int,
        device: torch.device,
        fp16: bool,
        seed: int,
        *,
        pin_memory: bool,
        views: Sequence[int | None] = (),
        extras_at: int | None = None,
        goal_heads: GoalHeads | None = None,
        record: bool = False,
        value_stats: ValueStats | None = None,
        greedy: bool = False,
    ) -> None:
        """`views[k]`: the core's view model k sees the game through (None or absent: the game's own);
        `extras_at`: where the selector's own numbers go (`extras_at()`; None: after the game's);
        `goal_heads`: each model's goal guesses for the selector to see (`settings.goal_heads`); `record`:
        keep every step for `nitro.selector_learning` (call `begin` before each collection); `value_stats`:
        the value scale to start from (with `settings.standardise_values`; a saved bot's); `greedy`: each
        model plays its likeliest action, as the exported bot does (else sampled as in training)."""
        self.greedy = greedy
        if len(models) < 2:
            raise ValueError("a multi-model bot needs at least two models")
        self.models = list(models)
        self.count = len(self.models)
        self.views = list(views) or [None] * self.count
        self.step_seconds = tick_skip / 120
        self.hold_steps = hold_steps(settings, tick_skip)
        self.switch_cost = settings.switch_cost
        self.device, self.fp16 = device, fp16
        self.goal_heads = goal_heads
        self.generator = torch.Generator(device=device)
        self.driven = torch.zeros(self.count, dtype=torch.int64, device=device)
        self.switches = torch.zeros((), dtype=torch.int64, device=device)
        self.reward_sum = torch.zeros((), device=device)  # |reward| of the learner's car-steps
        self.value_stats = (
            (value_stats or ValueStats(self.count, device)) if settings.standardise_values else None
        )
        self.bonus = (
            ShareBonus(settings.min_share, settings.share_bonus_rate, device)
            if any(settings.min_share)
            else None
        )
        # The learner's own cars (bool per agent, all groups in order; None: every car): self-play's past
        # versions drive the others, which count in no statistic.
        self.learner_rows: torch.Tensor | None = None
        game_size = envs.groups[0].obs.shape[1]
        self.extras_at = game_size if extras_at is None else extras_at
        extra = extra_size(self.count, goal_heads is not None)
        self.groups: tuple[GroupTensors, ...] = tuple(
            GroupTensors.allocate(tensors.obs.shape[0], game_size + extra, self.count, pin_memory=pin_memory)
            for tensors in envs.groups
        )
        sizes = [tensors.obs.shape[0] for tensors in envs.groups]
        self.offsets = [sum(sizes[:g]) for g in range(len(sizes))]
        self.record = record
        self.recorded: Recorded | None = None
        self._view_obs: list[list[torch.Tensor]] = []  # per group, per view: [agents, size] on the CPU
        self._view_mask: list[list[torch.Tensor]] = []
        self._model_obs: list[list[torch.Tensor]] = []  # per group, per model: what it saw now (device)
        self._model_mask: list[list[torch.Tensor]] = []
        self._goals: list[torch.Tensor] = []  # per group: int8 [agents] on the CPU
        self.cars: list[_Cars] = []
        self.attach(envs, seed)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def attach(self, envs: Games, seed: int) -> SelectorEnvs:
        """Continues with `envs` (freshly built games of the same shape, at a save): every car starts
        without an active model and the sampling restarts from `seed`."""
        if [t.obs.shape[0] for t in envs.groups] != [t.obs.shape[0] for t in self.groups]:
            raise ValueError("the new games must have the same groups as the old ones")
        self.envs = envs
        self.generator.manual_seed(seed)
        self.cars = []
        used = sorted({view for view in self.views if view is not None})
        sizes = {view: envs.view_size(view) for view in used} if used else {}  # type: ignore[attr-defined]
        self._view_obs, self._view_mask, self._goals = [], [], []
        for group, tensors in enumerate(envs.groups):
            agents = tensors.obs.shape[0]
            actions = tensors.mask.shape[1]
            self._view_obs.append(
                [torch.zeros(agents, sizes.get(v, 0)) for v in range(max(used, default=-1) + 1)]
            )
            self._view_mask.append(
                [torch.zeros(agents, actions, dtype=torch.bool) for _ in range(max(used, default=-1) + 1)]
            )
            self._goals.append(torch.zeros(agents, dtype=torch.int8))
            self.cars.append(
                _Cars(
                    active=torch.full((agents,), -1, dtype=torch.int64, device=self.device),
                    held=torch.zeros(agents, dtype=torch.int64, device=self.device),
                    candidates=torch.zeros(self.count, agents, dtype=torch.int64, device=self.device),
                    log_probs=torch.zeros(self.count, agents, device=self.device),
                    values=torch.zeros(agents, self.count, device=self.device),
                    switched=torch.zeros(agents, dtype=torch.bool),
                    driving=torch.full((agents,), -1, dtype=torch.int64, device=self.device),
                )
            )
            outer = self.groups[group]
            for name in ("reward", "terminal", "truncated", "episode", "step"):
                getattr(outer, name).copy_(getattr(tensors, name))
        self._model_obs = [[] for _ in envs.groups]
        self._model_mask = [[] for _ in envs.groups]
        for group in range(len(envs.groups)):
            self._prepare(group)
        return self

    def release(self) -> None:
        """Lets the games go (before new ones are built at a save, so only one set is in memory)."""
        del self.envs

    def _rows(self, group: int) -> torch.Tensor | None:
        """The learner's own cars of `group` (bool on the device), or None for every car."""
        if self.learner_rows is None:
            return None
        agents = self.groups[group].obs.shape[0]
        return self.learner_rows[self.offsets[group] : self.offsets[group] + agents].to(self.device)

    def begin(self, steps: int) -> None:
        """Starts recording a collection of `steps` steps per car (with `record`)."""
        if not self.record:
            return
        agents = sum(t.obs.shape[0] for t in self.groups)
        widest = max(max(obs.shape[1] for obs in models) for models in self._model_obs)
        actions = self.envs.groups[0].mask.shape[1]
        if self.recorded is None or self.recorded.obs.shape[:2] != (steps, agents):
            d, k = self.device, self.count

            def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
                return torch.zeros(shape, dtype=dtype, device=d)

            self.recorded = Recorded(
                obs=zeros(steps, agents, widest),
                mask=zeros(steps, agents, actions, dtype=torch.bool),
                actions=zeros(steps, agents, dtype=torch.int64),
                log_probs=zeros(steps, agents),
                driver=zeros(steps, agents, dtype=torch.int64),
                values=zeros(steps, agents, k),
                rewards=zeros(steps, agents),
                terminal=zeros(steps, agents, dtype=torch.bool),
                truncated=zeros(steps, agents, dtype=torch.bool),
                truncated_values=zeros(steps, agents, k),
                goals=zeros(steps, agents, dtype=torch.int8),
                episode=zeros(steps, agents, dtype=torch.int64),
                episode_step=zeros(steps, agents, dtype=torch.int32),
                bootstrap_values=zeros(agents, k),
            )
        self.recorded.steps = [0] * len(self.groups)

    def step(self, group: int) -> None:
        choice = self.groups[group].actions.to(self.device, torch.int64)
        self.envs.groups[group].actions.copy_(self.act(group, choice).to(torch.int32))
        self.envs.step(group)

    def act(self, group: int, choice: torch.Tensor) -> torch.Tensor:
        """Hands each car of `group` to the model `choice` names (int64 on the device), counts and records it,
        and returns the actions those models picked (int64 on the device), without stepping the games (`nitro
        eval` and `nitro watch` step them themselves, with `after_step` after)."""
        cars = self.cars[group]
        actions = self.choose(group, choice)
        if self.recorded is not None:
            self._record_step(group, choice, actions)
        rows = self._rows(group)
        counted = choice if rows is None else choice[rows]
        self.driven += torch.bincount(counted, minlength=self.count)
        switched = cars.switched if rows is None else cars.switched & rows.cpu()
        self.switches += switched.sum().to(self.device)
        return actions

    def choose(self, group: int, choice: torch.Tensor) -> torch.Tensor:
        """Hands each car of `group` to the model `choice` names (int64 on the device) and returns the actions
        those models picked (int64 on the device)."""
        cars = self.cars[group]
        cars.switched.copy_(((cars.active >= 0) & (choice != cars.active)).cpu())
        changed = choice != cars.active
        cars.held = torch.where(changed, torch.zeros_like(cars.held), cars.held) + 1
        cars.active = choice
        cars.driving = choice.clone()
        return cars.candidates.gather(0, choice[None]).squeeze(0)

    def _record_step(self, group: int, choice: torch.Tensor, actions: torch.Tensor) -> None:
        recorded, cars = self.recorded, self.cars[group]
        assert recorded is not None
        t = recorded.steps[group]
        rows = slice(self.offsets[group], self.offsets[group] + choice.shape[0])
        obs, mask = recorded.obs[t, rows], recorded.mask[t, rows]
        obs.zero_()
        for k in range(self.count):
            mine = choice == k
            seen = self._model_obs[group][k]
            obs[:, : seen.shape[1]] = torch.where(mine[:, None], seen, obs[:, : seen.shape[1]])
            mask.copy_(torch.where(mine[:, None], self._model_mask[group][k], mask))
        recorded.actions[t, rows] = actions
        recorded.log_probs[t, rows] = cars.log_probs.gather(0, choice[None]).squeeze(0)
        recorded.driver[t, rows] = choice
        recorded.values[t, rows] = cars.values
        inner = self.envs.groups[group]
        recorded.episode[t, rows] = inner.episode.to(self.device)
        recorded.episode_step[t, rows] = inner.step.to(self.device)

    def wait(self, group: int) -> None:
        self.envs.wait(group)
        self.after_step(group)

    def after_step(self, group: int) -> None:
        """Reads `group`'s step once the games have made it: rewards, endings, the next observation."""
        inner, outer, cars = self.envs.groups[group], self.groups[group], self.cars[group]
        outer.reward.copy_(inner.reward)
        if self.switch_cost:
            outer.reward.sub_(self.switch_cost * cars.switched.float())
        rows = self._rows(group)
        if self.bonus is not None or self.recorded is not None:
            rewards = inner.reward.to(self.device)
            self.reward_sum += (rewards.abs() if rows is None else rewards.abs()[rows]).sum()
        if self.bonus is not None:
            paid = torch.where(cars.driving >= 0, self.bonus.bonus[cars.driving.clamp_min(0)], 0.0)
            outer.reward.add_(paid.cpu())
        for name in ("terminal", "truncated", "episode", "step"):
            getattr(outer, name).copy_(getattr(inner, name))
        truncated_values = None
        if inner.truncated.any():
            local = inner.truncated.nonzero().squeeze(1)
            on_device = local.to(self.device)
            final = inner.final_obs.index_select(0, local).to(self.device)
            values = self._final_values(group, final, local)
            extended = self._extended(
                final, self._seen_values(values), cars.active[on_device], cars.held[on_device],
                self._final_goals(group, final, local),
            )  # fmt: skip
            outer.final_obs.index_copy_(0, local, extended.cpu())
            truncated_values = (on_device, values)
        if self.recorded is not None:
            self._record_result(group, inner, truncated_values)
        ended = (inner.terminal | inner.truncated).to(self.device)
        cars.active = torch.where(ended, torch.full_like(cars.active, -1), cars.active)
        cars.held = torch.where(ended, torch.zeros_like(cars.held), cars.held)
        self._prepare(group)
        if self.recorded is not None and self.recorded.steps[group] == self.recorded.obs.shape[0]:
            rows_all = slice(self.offsets[group], self.offsets[group] + cars.values.shape[0])
            self.recorded.bootstrap_values[rows_all] = cars.values

    def _record_result(
        self, group: int, inner: GroupTensors, truncated: tuple[torch.Tensor, torch.Tensor] | None
    ) -> None:
        recorded = self.recorded
        assert recorded is not None
        t = recorded.steps[group]
        rows = slice(self.offsets[group], self.offsets[group] + inner.reward.shape[0])
        recorded.rewards[t, rows] = inner.reward.to(self.device)
        recorded.terminal[t, rows] = inner.terminal.to(self.device)
        recorded.truncated[t, rows] = inner.truncated.to(self.device)
        goals = self._goals[group]
        if inner.terminal.any() and hasattr(self.envs, "goals_into"):
            self.envs.goals_into(group, goals)
        else:
            goals.zero_()
        recorded.goals[t, rows] = goals.to(self.device)
        if truncated is not None:
            local, values = truncated
            recorded.truncated_values[t, rows].index_copy_(0, local, values)
        recorded.steps[group] = t + 1

    def take_stats(self) -> SelectorStats:
        """Car-steps each model drove and the switches since the last call (the learner's own cars)."""
        stats = SelectorStats([int(n) for n in self.driven.tolist()], int(self.switches))
        self.driven.zero_()
        self.switches.zero_()
        return stats

    def end_collection(self, driven: Sequence[int]) -> None:
        """After a collection: the share bonus follows the shares `driven` gave (`take_stats`), the value
        inputs' scale moves to the states seen."""
        steps = sum(driven)
        typical = float(self.reward_sum) / steps if steps else 0.0
        self.reward_sum.zero_()
        if self.bonus is not None:
            self.bonus.update(driven, typical)
        if self.value_stats is not None:
            self.value_stats.update()

    def _views_of(self, group: int) -> None:
        """The core's views of `group` now, for the models that need one."""
        for view in sorted({v for v in self.views if v is not None}):
            obs, mask = self._view_obs[group][view], self._view_mask[group][view]
            self.envs.views_into(group, view, obs, mask)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _prepare(self, group: int) -> None:
        """The selector's observation and mask for the group's current state, and each model's next action."""
        inner, outer, cars = self.envs.groups[group], self.groups[group], self.cars[group]
        obs = inner.obs.to(self.device)
        mask = inner.mask.to(self.device)
        if any(v is not None for v in self.views):
            self._views_of(group)
        values, seen_obs, seen_mask = [], [], []
        for index, model in enumerate(self.models):
            view = self.views[index]
            own_obs = obs if view is None else self._view_obs[group][view].to(self.device)
            own_mask = mask if view is None else self._view_mask[group][view].to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                logits, value = model(own_obs)
            log_probs = masked_log_probs(logits, own_mask)
            if self.greedy:
                pick = likeliest(logits, own_mask)[:, None]
            else:
                pick = torch.multinomial(log_probs.exp(), 1, generator=self.generator)
            cars.candidates[index] = pick.squeeze(1)
            cars.log_probs[index] = log_probs.gather(1, pick).squeeze(1)
            values.append(value.float())
            seen_obs.append(own_obs)
            seen_mask.append(own_mask)
        self._model_obs[group], self._model_mask[group] = seen_obs, seen_mask
        cars.values = torch.stack(values, dim=1)
        if self.value_stats is not None:
            rows = self._rows(group)
            self.value_stats.add(cars.values if rows is None else cars.values[rows])
        goals = self._goal_chances(seen_obs) if self.goal_heads is not None else None
        outer.obs.copy_(self._extended(obs, self._seen_values(cars.values), cars.active, cars.held, goals))
        free = self._free(cars.active, cars.held)
        outer.mask.copy_(free[:, None] | self._one_hot(cars.active).bool())

    def _seen_values(self, values: torch.Tensor) -> torch.Tensor:
        return values if self.value_stats is None else self.value_stats.apply(values)

    def _goal_chances(self, seen: Sequence[torch.Tensor]) -> torch.Tensor:
        """[agents, 2 * models]: per model, its chance that the car's team scores next and that the other
        team does."""
        assert self.goal_heads is not None
        chances = [self.goal_heads(k, obs).softmax(-1)[:, 1:] for k, obs in enumerate(seen)]
        return torch.cat(chances, dim=1)

    def _final_values(self, group: int, final: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Each model's value of the states truncated episodes ended in ([rows, models])."""
        values = []
        for index, model in enumerate(self.models):
            seen = self._final_view(group, index, final, local)
            with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                values.append(model.normalised_value(seen).float())
        return torch.stack(values, dim=1)

    def _final_view(self, group: int, model: int, final: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """What model `model` saw where a truncated episode ended: the game's final observation, or for a
        model with a view of its own, its view of the same car before the step (the core keeps no final view;
        one step earlier is the nearest state it has)."""
        view = self.views[model]
        if view is None:
            return final
        return self._model_obs[group][model].index_select(0, local.to(self.device))

    def _final_goals(self, group: int, final: torch.Tensor, local: torch.Tensor) -> torch.Tensor | None:
        if self.goal_heads is None:
            return None
        seen = [self._final_view(group, k, final, local) for k in range(self.count)]
        with torch.no_grad():
            return self._goal_chances(seen)

    def _free(self, active: torch.Tensor, held: torch.Tensor) -> torch.Tensor:
        return (active < 0) | (held >= self.hold_steps)

    def _one_hot(self, active: torch.Tensor) -> torch.Tensor:
        return F.one_hot(active.clamp_min(0), self.count).float() * (active >= 0)[:, None].float()

    def _extended(
        self,
        obs: torch.Tensor,
        values: torch.Tensor,
        active: torch.Tensor,
        held: torch.Tensor,
        goals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seconds = (held.float() * self.step_seconds).clamp(max=HELD_CAP_SECONDS) / HELD_CAP_SECONDS
        free = self._free(active, held).float()
        extras = [self._one_hot(active), seconds[:, None], free[:, None], values]
        if goals is not None:
            extras.append(goals)
        at = self.extras_at
        return torch.cat([obs[:, :at], *extras, obs[:, at:]], dim=1)

    def state_dict(self) -> dict[str, Any]:
        """What a save keeps of the selector's own running state (the bonus and the value scale)."""
        return {
            **({"bonus": self.bonus.state_dict()} if self.bonus is not None else {}),
            **({"value_stats": self.value_stats.state_dict()} if self.value_stats is not None else {}),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.bonus is not None and "bonus" in state:
            self.bonus.load_state_dict(state["bonus"])
        if self.value_stats is not None and "value_stats" in state:
            self.value_stats.load_state_dict(state["value_stats"])


@dataclass
class Engine:
    """Everything a multi-model run's games need: the models (each with its view), their goal heads, and
    the games as the selector sees them."""

    games: SelectorEnvs
    models: list[ActorCritic]
    obs_sizes: list[int]  # each model's own observation size
    goal_heads: GoalHeads | None


def engine(
    config: NitroConfig,
    envs: Any,
    specialists: Sequence[Specialist],
    device: torch.device,
    fp16: bool,
    seed: int,
    *,
    record: bool = False,
    pin_memory: bool = False,
    saved: dict[str, Any] | None = None,
) -> Engine:
    """The models of `specialists` on `device` for `envs` (an `EnvPool` built with `views(specialists,
    config)`), and the selector's games over them. `saved`: a save's `selector` entry, whose goal heads and
    value scale are in place before the games' first observation is made (so a resumed run decides as the
    run that saved it would have)."""
    found = views(specialists, config)
    view_sizes = [envs.view_size(v) for v in range(len(found))]
    game_size, num_actions = envs.groups[0].obs.shape[1], envs.groups[0].mask.shape[1]
    models = networks(
        specialists,
        config,
        game_size,
        num_actions,
        device,
        trainable=config.selector.train_models,
        view_sizes=view_sizes,
    )
    indices = [None if (v := view_of(s, config)) is None else found.index(v) for s in specialists]
    obs_sizes = [game_size if i is None else view_sizes[i] for i in indices]
    heads = GoalHeads(obs_sizes).to(device).eval() if config.selector.goal_heads else None
    learning, kept = (saved or {}).get("learning", {}), (saved or {}).get("games", {})
    if heads is not None and "goal_heads" in learning:
        heads.load_state_dict(learning["goal_heads"])
    value_stats = None
    if config.selector.standardise_values and "value_stats" in kept:
        value_stats = ValueStats(len(models), device)
        value_stats.load_state_dict(kept["value_stats"])
    games = SelectorEnvs(
        envs,
        models,
        config.selector,
        config.env.tick_skip,
        device,
        fp16,
        seed,
        pin_memory=pin_memory,
        views=indices,
        extras_at=extras_at(config, game_size),
        goal_heads=heads,
        record=record,
        value_stats=value_stats,
    )
    return Engine(games, models, obs_sizes, heads)


@dataclass
class Saved:
    """A multi-model run's save, rebuilt to play (export, `nitro eval`, `nitro watch`): the selector's
    networks, the models (each with its view of the game: None for the game's own), their goal heads and
    the value scale the selector saw them in."""

    selector: ActorCritic
    labels: list[str]
    models: list[ActorCritic]
    views: list[dict[str, Any] | None]
    goal_heads: GoalHeads | None
    value_shift: list[float]
    value_scale: list[float]
    extras_at: int
    game_size: int
    settings: SelectorConfig = field(default_factory=SelectorConfig)  # its hold and inputs

    @classmethod
    def of(cls, chooser: ActorCritic, named: Sequence[tuple[str, ActorCritic]]) -> Saved:
        """A plain selector: its models see the game's own observation, no goal heads, in their own units."""
        size, count = named[0][1].obs_size, len(named)
        return cls(
            chooser, [n for n, _ in named], [m for _, m in named], [None] * count, None, [0.0] * count,
            [1.0] * count, size, size,
        )  # fmt: skip


def game_size(config: NitroConfig) -> int:
    """Numbers in the observation of `config`'s game (no games built)."""
    from nitro import _core

    env = config.env
    size: int = _core.observation_layout_of(
        env.observation, env.team_size, env.opponents, observation_room(config), len(env.ball_prediction)
    )[0]
    return size


def saved(state: dict[str, Any], config: NitroConfig, num_actions: int, device: torch.device) -> Saved:
    """The multi-model bot a save holds, for games of `config` with `num_actions` actions."""
    kept = from_state(state.get("specialists", []))
    if len(kept) < 2:
        raise SelectorError(f"the save keeps {len(kept)} model(s); a multi-model bot needs 2 or more")
    size = game_size(config)
    found = views(kept, config)
    view_sizes = [_view_size(v) for v in found]
    models = networks(kept, config, size, num_actions, device, view_sizes=view_sizes)
    choices = len(kept)
    chooser = network(config, size, choices)
    try:
        chooser.load_state_dict(state["learner"]["model"])
    except RuntimeError as err:
        raise SelectorError(f"the selector's networks do not fit {choices} models: {err}") from None
    chooser.eval().to(device)
    each = [view_of(s, config) for s in kept]
    heads = None
    learning = state.get("selector", {}).get("learning", {})
    if config.selector.goal_heads:
        sizes = [size if v is None else view_sizes[found.index(v)] for v in each]
        heads = GoalHeads(sizes)
        if "goal_heads" in learning:
            heads.load_state_dict(learning["goal_heads"])
        heads.eval().to(device)
    shift, scale = [0.0] * choices, [1.0] * choices
    # The value scale the save's selector learned with: a save keeps one only while standardise_values was on,
    # whatever the settings file says now.
    stats = state.get("selector", {}).get("games", {}).get("value_stats")
    standardised = stats is not None and bool(stats["started"])
    if standardised:
        shift, scale = [float(v) for v in stats["mean"]], [float(v) for v in stats["std"]]
    # The hold the save's selector learned with (the exported bot keeps it too), whatever the file says now.
    hold = state.get("all_settings", {}).get("selector.min_hold_seconds", config.selector.min_hold_seconds)
    settings = dataclasses.replace(  # a bonus belongs to training
        config.selector, min_share=(), standardise_values=standardised, min_hold_seconds=float(hold)
    )
    return Saved(
        chooser, labels(kept), models, each, heads, shift, scale, extras_at(config, size), size, settings
    )


def distinct_views(bundles: Sequence[Saved]) -> list[dict[str, Any]]:
    """The views every bundle's models need, each once, for one pool (`EnvPool(views=...)`)."""
    found: list[dict[str, Any]] = []
    for bundle in bundles:
        for view in bundle.views:
            if view is not None and view not in found:
                found.append(view)
    return found


@dataclass
class Decision:
    """What a multi-model bot decided for every car of a group: the actions played, the model that drives
    each car, the selector's value of the moment and each played action's probability and the randomness of
    its model's choice (0 sure, 1 uniform), as `nitro watch` shows them."""

    actions: torch.Tensor
    drivers: torch.Tensor
    values: torch.Tensor
    sure: torch.Tensor
    randomness: torch.Tensor


class Player:
    """A saved multi-model bot playing games without learning (`nitro eval`, `nitro watch`), sampled as in
    training or, `greedy`, the most likely choice of each (as the exported bot plays by default)."""

    def __init__(
        self,
        bundle: Saved,
        config: NitroConfig,
        envs: Any,
        pool_views: Sequence[dict[str, Any]],
        device: torch.device,
        fp16: bool,
        seed: int,
        greedy: bool = False,
    ) -> None:
        self.bundle, self.device, self.fp16, self.greedy = bundle, device, fp16, greedy
        indices = [None if v is None else list(pool_views).index(v) for v in bundle.views]
        stats = ValueStats(len(bundle.models), device)  # the scale the selector last trained with
        stats.mean = torch.tensor(bundle.value_shift, device=device)
        stats.std = torch.tensor(bundle.value_scale, device=device)
        stats.started = True
        self.games = SelectorEnvs(
            envs, bundle.models, bundle.settings, config.env.tick_skip, device, fp16, seed, pin_memory=False,
            views=indices, extras_at=bundle.extras_at, goal_heads=bundle.goal_heads, value_stats=stats,
            greedy=greedy,
        )  # fmt: skip
        self.generator = torch.Generator(device=device).manual_seed(seed ^ 0x9E3779B9)

    @torch.no_grad()
    def decide(self, group: int) -> Decision:
        """The selector's choice for every car of `group` now and the actions its models play (the games are
        stepped by the caller, then `after_step`)."""
        tensors = self.games.groups[group]
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            logits, value = self.bundle.selector(tensors.obs.to(self.device))
        log_probs = masked_log_probs(logits, tensors.mask.to(self.device))
        if self.greedy:
            choice = likeliest(logits, tensors.mask.to(self.device))
        else:
            choice = torch.multinomial(log_probs.exp(), 1, generator=self.generator).squeeze(1)
        cars = self.games.cars[group]
        actions = self.games.act(group, choice)
        sure = cars.log_probs.gather(0, choice[None]).squeeze(0).exp()
        return Decision(actions, choice, value.float(), sure, self._randomness(group, choice))

    def _randomness(self, group: int, choice: torch.Tensor) -> torch.Tensor:
        """The driving model's normalised entropy at this decision (0 sure, 1 uniform over its allowed)."""
        out = torch.zeros(choice.shape[0], device=self.device)
        for index, model in enumerate(self.bundle.models):
            mine = choice == index
            if not bool(mine.any()):
                continue
            obs, mask = self.games._model_obs[group][index][mine], self.games._model_mask[group][index][mine]
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                logits, _ = model(obs)
            log_probs = masked_log_probs(logits, mask)
            probs = log_probs.exp()
            entropy = -(probs * log_probs.clamp(min=-1e9)).nan_to_num().sum(1)
            out[mine] = entropy / mask.sum(1).clamp(min=2).float().log()
        return out

    def after_step(self, group: int) -> None:
        self.games.after_step(group)
