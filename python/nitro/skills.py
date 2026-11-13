"""`nitro skills`: what can a saved bot do? Its policy plays each practice task's situations and is scored by
the task's own measure, never the reward.

The situations are held out: they come from the tasks' state setters with the fixed evaluation seeds, which
training never sees. The bot plays with its own game settings (observation, team size, rhythm), so a 1v1 bot
meets one idle opponent parked somewhere on the field, as the task files park extra cars. Each task's bar is
what a fresh bot trained on that one skill for its budget reaches, so a full-game bot above the bar has the
skill; below it, not yet. Results are printed and appended to `runs/<name>/skills.csv`, one row per task and
save, so the skills' progress over a run can be read off later. Played sampled as in training by default;
another pick rule for the bot (0 = likeliest, as the RLBot bot) writes to a file of its own
(`skills_file`), so the sampled history stays one series.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nitro import _core
from nitro.config import NitroConfig
from nitro.eval import Side, _model, _sizes
from nitro.evaluate import evaluate
from nitro.tasks import EVAL_SEEDS, TASKS, Task
from nitro.training import resolve_device

# Game settings the bot was trained with, which the task situations must keep so its observation fits.
_BOT_ENV = (
    "team_size", "opponents", "tick_skip", "action_delay", "observation", "obs_team_size", "groups",
    "threads",
)  # fmt: skip
# Tasks a saved bot can be scored on alone; `kickoff` needs an opponent policy and is left out.
DEFAULT_TASKS = ("ball_touch", "recovery", "boost_pickup", "shot", "save", "aerial_touch")
CSV_FIELDS = ("when", "agent_steps", "task", "metric", "value", "unit", "bar", "above_bar", "games", "seed")


class Idle(nn.Module):
    """An opponent that does nothing: it always picks the neutral action (no throttle, steering, jump, boost
    or handbrake), so the bot's situation is the task's and nothing else."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        logits = torch.full((num_actions,), -30.0)
        logits[neutral_action()] = 30.0
        self.logits: torch.Tensor
        self.register_buffer("logits", logits)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits.expand(obs.shape[0], -1), torch.zeros(obs.shape[0], device=obs.device)


def neutral_action() -> int:
    """The action table's index whose controls are all zero."""
    for index in range(_core.num_actions()):
        if not any(_core.action_controls(index)):
            return index
    raise RuntimeError("the action table has no neutral action")


def battery_config(task: Task, bot: NitroConfig) -> NitroConfig:
    """The task's situation (start states, endings, rewards, arena count) played with the bot's own game
    settings and networks."""
    base = task.settings()
    env = dataclasses.replace(base.env, **{name: getattr(bot.env, name) for name in _BOT_ENV})
    run = dataclasses.replace(base.run, precision=bot.run.precision, device=bot.run.device)
    return dataclasses.replace(base, paths=bot.paths, run=run, env=env, model=bot.model)


@dataclass(frozen=True)
class SkillResult:
    task: str
    metric: str
    value: float
    unit: str
    bar: float
    higher_is_better: bool
    games: int
    others: dict[str, float]

    @property
    def above_bar(self) -> bool:
        return self.value >= self.bar if self.higher_is_better else self.value <= self.bar

    def line(self) -> str:
        # A time is better when lower: under the bar is the good side.
        if self.higher_is_better:
            verdict = "above the bar: has the skill" if self.above_bar else "below the bar: not yet"
        else:
            verdict = "under the bar: has the skill" if self.above_bar else "over the bar: not yet"
        extras = ", ".join(f"{name} {value:.3g}" for name, value in self.others.items())
        return (
            f"{self.task:<13} {self.metric} {self.value:.3g}{self.unit} (bar {self.bar:g}{self.unit}; "
            f"{verdict}){'; ' + extras if extras else ''}"
        )


def skills_file(temperature: float) -> str:
    """The results file for the bot's pick rule: sampled rows stay in skills.csv."""
    if temperature == 1:
        return "skills.csv"
    return "skills_likeliest.csv" if temperature == 0 else f"skills_temperature_{temperature:g}.csv"


def play_task(side: Side, task: Task, games: int, seed: int, temperature: float = 1.0) -> SkillResult:
    config = battery_config(task, side.config)
    arenas = min(config.env.num_arenas, max(games, 1))
    config = dataclasses.replace(
        config, env=dataclasses.replace(config.env, num_arenas=arenas, groups=min(config.env.groups, arenas))
    )
    device = resolve_device(config.run)
    fp16 = config.run.precision == "fp16" and device.type == "cuda"
    obs_size, num_actions = _sizes(config)
    model = _model(side, obs_size, num_actions, device)
    # The save drives its first car only; every other car (the opponents and, in 2v2 or 3v3, the
    # teammates) is idle, so nobody else can touch the ball first and end the episode.
    idle = Idle(num_actions).to(device)
    records = evaluate(
        model,
        config,
        seed,
        math.ceil(games / arenas),
        device,
        fp16,
        idle,
        bot_cars=(0,),
        pick=(temperature, 1.0),
    )
    mine = [r for r in records if r.car == 0]  # the bot's car; the idle cars don't count
    measured = task.measure(mine)
    return SkillResult(
        task=task.name,
        metric=task.metric,
        value=measured[task.metric],
        unit=task.unit,
        bar=task.bar,
        higher_is_better=task.higher_is_better,
        games=len(mine),
        others={name: value for name, value in measured.items() if name != task.metric},
    )


def run_battery(
    side: Side,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    games: int = 512,
    seed: int = EVAL_SEEDS[0],
    temperature: float = 1.0,
) -> list[SkillResult]:
    """`temperature`: the bot car's pick rule, 1 = sampled as in training, 0 = likeliest."""
    return [play_task(side, TASKS[name], games, seed, temperature) for name in tasks]


def append_csv(path: Path, agent_steps: int, results: list[SkillResult], seed: int) -> None:
    """One row per task; the file grows with every save scored, so a run's skills can be charted later."""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        when = time.strftime("%Y-%m-%d %H:%M")
        for r in results:
            row: dict[str, Any] = {
                "when": when,
                "agent_steps": agent_steps,
                "task": r.task,
                "metric": r.metric,
                "value": round(r.value, 5),
                "unit": r.unit,
                "bar": r.bar,
                "above_bar": r.above_bar,
                "games": r.games,
                "seed": seed,
            }
            writer.writerow(row)
