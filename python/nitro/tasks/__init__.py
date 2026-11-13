"""Narrow practice tasks: one simple situation each, judged by a task metric.

A run trains a fresh policy on the task for a fixed number of agent steps and evaluates it before and
after, playing the sampled policy without learning on the same start states for every run
(`nitro.evaluate`). The reward never decides anything. The thresholds were fixed before the first
results. Used by `nitro ab` and by the scripts in benchmarks/tasks.

Metrics:
- ball_touch: mean time to touch, counting a miss as 10 s; touch rate within 10 s.
- recovery: mean time to land on the wheels, counting a miss as 6 s; landing rate.
- boost_pickup: mean boost collected per 20 s episode (100 = one full tank).
- kickoff (against the untrained starting policy of the same seed): mean time to first touch, counting a
  miss as 6 s; share of kickoffs touched first; ball position toward the opponent's goal 2 s after the
  first touch, or when the episode ended if that came sooner (uu).
- shot: share of episodes ending in a goal; mean time to score, counting a miss as 8 s.
- save: share of episodes without a goal conceded; share in which the car touched the ball.
- aerial_touch: mean time to touch a ball in the air, counting a miss as 6 s; touch rate.
- touch_mix: half ball-touch, half aerial-touch episodes: mean time to touch over all of them, a miss
  counting as the task's 10 s limit; the two kinds' times; touch rate.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nitro import _core
from nitro.config import NitroConfig, load_config
from nitro.evaluate import evaluate
from nitro.health import learning_problems
from nitro.learner import LearnStats
from nitro.training import build

# The same two evaluation seeds for every run, so every policy meets the same start states. The second
# one re-measures the trained policy: the gap between the two shows how stable a measurement is.
EVAL_SEEDS = (1_000_001, 1_000_002)
# How the bot learns, which a task takes from the settings it is given; everything else is the task's.
_LEARNING_ENV = (
    "tick_skip", "action_delay", "observation", "obs_team_size", "groups", "threads", "pipelined_inference",
    "concurrent_passes", "card_starts_games",
)  # fmt: skip

# The task's own players and arenas, never taken from the settings (every other [env] field is _LEARNING_ENV).
_TASK_ENV = ("team_size", "team_sizes", "team_size_shares", "shuffle_slots", "opponents", "num_arenas")

Records = list[_core.EpisodeRecord]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def _touch_within(limit: float) -> Callable[[Records], dict[str, float]]:
    """Time to touch, counting a miss as `limit` seconds, and the share of episodes with a touch."""

    def measure(records: Records) -> dict[str, float]:
        touched = [r.first_touch for r in records if not math.isnan(r.first_touch)]
        return {
            "time_to_touch": _mean(
                [r.first_touch if not math.isnan(r.first_touch) else limit for r in records]
            ),
            "touch_rate": len(touched) / len(records),
        }

    return measure


_ball_touch = _touch_within(10.0)
_aerial_touch = _touch_within(6.0)
# touch_mix: one limit for both kinds (the task's max_episode_seconds); the kind of an episode is the index
# of the setter that started it.
_TOUCH_MIX_LIMIT = 10.0
_TOUCH_MIX_KINDS = {0: "ground_time", 1: "aerial_time"}


def _touch_mix(records: Records) -> dict[str, float]:
    def time(record: _core.EpisodeRecord) -> float:
        return record.first_touch if not math.isnan(record.first_touch) else _TOUCH_MIX_LIMIT

    measured = {
        "time_to_touch": _mean([time(r) for r in records]),
        "touch_rate": _mean([float(not math.isnan(r.first_touch)) for r in records]),
    }
    for index, name in _TOUCH_MIX_KINDS.items():
        measured[name] = _mean([time(r) for r in records if r.setter == index])
    return measured


def _recovery(records: Records) -> dict[str, float]:
    limit = 6.0
    landed = [r for r in records if r.terminal]
    return {
        "time_to_land": _mean([r.landed if r.terminal else limit for r in records]),
        "landing_rate": len(landed) / len(records),
    }


def _boost_pickup(records: Records) -> dict[str, float]:
    return {"boost_collected": _mean([r.boost_collected for r in records])}


def _shot(records: Records) -> dict[str, float]:
    limit = 8.0
    return {
        "goal_rate": _mean([float(r.goal > 0) for r in records]),
        "time_to_goal": _mean([r.seconds if r.goal > 0 else limit for r in records]),
    }


def _save(records: Records) -> dict[str, float]:
    return {
        "save_rate": _mean([float(r.goal >= 0) for r in records]),
        "touch_rate": _mean([float(not math.isnan(r.first_touch)) for r in records]),
    }


def _kickoff(records: Records) -> dict[str, float]:
    """Blue (the policy under test) against Orange (its untrained starting policy), episode by episode."""
    limit = 6.0
    cars: dict[tuple[int, int], Records] = {}
    for record in records:
        cars.setdefault((record.env, record.car), []).append(record)
    pairs = [
        (blue, orange)
        for (env, car), blues in cars.items()
        if car == 0
        for blue, orange in zip(blues, cars[(env, 1)], strict=True)
    ]
    first = [
        not math.isnan(b.first_touch) and (math.isnan(o.first_touch) or b.first_touch < o.first_touch)
        for b, o in pairs
    ]
    return {
        "time_to_first_touch": _mean(
            [b.first_touch if not math.isnan(b.first_touch) else limit for b, _ in pairs]
        ),
        "first_touch_rate": sum(first) / len(pairs),
        "ball_y_after_touch": _mean(
            [b.ball_y_after_touch for b, _ in pairs if not math.isnan(b.ball_y_after_touch)]
        ),
    }


@dataclass(frozen=True)
class Task:
    name: str
    metrics: tuple[str, ...]  # what `measure` returns; the first one decides
    measure: Callable[[Records], dict[str, float]]
    unit: str
    higher_is_better: bool
    smallest_difference: float  # the smallest difference in the deciding metric that matters
    bar: float  # a working learner's trained mean reaches this (at least, or at most when lower is better)
    steps: int  # agent steps of training per run
    episodes: int  # evaluation episodes per arena
    against_untrained: bool = False  # evaluated against the untrained policy instead of alone

    @property
    def metric(self) -> str:
        return self.metrics[0]

    def settings(self) -> NitroConfig:
        return load_config(Path(__file__).with_name(f"{self.name}.toml"))

    def better(self, a: float, b: float) -> float:
        """How much better `a` is than `b` in the deciding metric (positive: better)."""
        return a - b if self.higher_is_better else b - a


TASKS = {
    task.name: task
    for task in (
        Task(
            name="ball_touch",
            metrics=("time_to_touch", "touch_rate"),
            measure=_ball_touch,
            unit="s",
            higher_is_better=False,
            smallest_difference=0.1,
            bar=3.0,
            steps=10_000_000,
            episodes=1,
        ),
        Task(
            name="recovery",
            metrics=("time_to_land", "landing_rate"),
            measure=_recovery,
            unit="s",
            higher_is_better=False,
            smallest_difference=0.1,
            bar=1.6,
            steps=10_000_000,
            episodes=1,
        ),
        Task(
            name="boost_pickup",
            metrics=("boost_collected",),
            measure=_boost_pickup,
            unit="",
            higher_is_better=True,
            smallest_difference=12.0,
            bar=40.0,
            steps=10_000_000,
            episodes=1,
        ),
        # Sized by a pilot run: the goal rate levels off near 0.52-0.55 from
        # 7.5M steps on (0.04 untrained), so 10M steps and a bar of 0.45.
        Task(
            name="shot",
            metrics=("goal_rate", "time_to_goal"),
            measure=_shot,
            unit="",
            higher_is_better=True,
            smallest_difference=0.05,
            bar=0.45,
            steps=10_000_000,
            episodes=1,
        ),
        # Sized by a second pilot run: at 2000-3200 uu/s the save rate climbs from 0.17
        # to 0.84 at 10M and levels off near 0.88 from 15M on (no longer saturated): 15M steps, bar 0.8.
        Task(
            name="save",
            metrics=("save_rate", "touch_rate"),
            measure=_save,
            unit="",
            higher_is_better=True,
            smallest_difference=0.05,
            bar=0.8,
            steps=15_000_000,
            episodes=1,
        ),
        # Sized by a pilot run: the touch rate peaks at 0.15 at 10M steps (time 5.34 s, 5.94
        # untrained) and then falls back to 0.06 by 25M, so 10M steps and a bar of 5.6 s.
        Task(
            name="aerial_touch",
            metrics=("time_to_touch", "touch_rate"),
            measure=_aerial_touch,
            unit="s",
            higher_is_better=False,
            smallest_difference=0.1,
            bar=5.6,
            steps=10_000_000,
            episodes=1,
        ),
        # Sized by a pilot run: 9.8 s untrained, 5.3 s at 10M, about 4.2 s from 17.5M on
        # (ground touches level off near 2.1 s by 7.5M; aerial ones keep improving to 6.2 s), so 20M steps
        # and a bar of 4.5 s.
        Task(
            name="touch_mix",
            metrics=("time_to_touch", "ground_time", "aerial_time", "touch_rate"),
            measure=_touch_mix,
            unit="s",
            higher_is_better=False,
            smallest_difference=0.1,
            bar=4.5,
            steps=20_000_000,
            episodes=1,
        ),
        Task(
            name="kickoff",
            metrics=("time_to_first_touch", "first_touch_rate", "ball_y_after_touch"),
            measure=_kickoff,
            unit="s",
            higher_is_better=False,
            smallest_difference=0.2,
            bar=4.5,
            steps=10_000_000,
            episodes=2,
            against_untrained=True,
        ),
    )
}


def practice_config(task: Task, settings: NitroConfig, seed: int, with_rewards: bool = False) -> NitroConfig:
    """The task's situation (players, rewards, start states, endings, arena count) with everything about
    how the bot learns taken from `settings`: networks, learner, precision, observation and actions.
    With `with_rewards`, the settings' `[[rewards]]` replace the task's too: a comparison of reward designs,
    still judged by the task's measure."""
    base = task.settings()
    env = dataclasses.replace(base.env, **{name: getattr(settings.env, name) for name in _LEARNING_ENV})
    # From [run] only what changes the maths: precision and determinism. The task keeps its own device,
    # saving and problem handling, so both sides are treated alike.
    run = dataclasses.replace(
        base.run, seed=seed, precision=settings.run.precision, deterministic=settings.run.deterministic
    )
    return dataclasses.replace(
        base,
        paths=settings.paths,
        run=run,
        env=env,
        model=settings.model,
        ppo=settings.ppo,
        rewards=settings.rewards if with_rewards else base.rewards,
    )


def practise(
    task: Task,
    config: NitroConfig,
    steps: int,
    checkpoints: Sequence[int] = (),
    also_likeliest: bool = False,
) -> list[dict[str, Any]]:
    """Trains a fresh policy on `task` for `steps` agent steps; one row per evaluation.

    Evaluates the untrained policy, the policy at each of `checkpoints` (agent steps) and the trained one,
    the last with both evaluation seeds. With `also_likeliest`, the trained policy is evaluated once more
    picking its likeliest action, as the RLBot bot plays (stage "trained_likeliest", the first seed);
    an untrained opponent still samples. A row's `agent_steps` are those learned from when it was
    evaluated: for a checkpoint, after the first iteration at or past it.
    """
    setup = build(config)
    device = next(setup.model.parameters()).device
    fp16 = config.run.precision == "fp16" and device.type == "cuda"
    opponent = copy.deepcopy(setup.model) if task.against_untrained else None

    def evaluation(
        stage: str, at: int, eval_seed: int, pick: tuple[float, float] = (1.0, 1.0)
    ) -> dict[str, Any]:
        records = evaluate(setup.model, config, eval_seed, task.episodes, device, fp16, opponent, pick=pick)
        measured = {name: round(value, 5) for name, value in task.measure(records).items()}
        return {
            "stage": stage,
            "agent_steps": at,
            "eval_seed": eval_seed,
            "episodes": len(records),
        } | measured

    stats: list[LearnStats] = []
    rows = [evaluation("untrained", 0, EVAL_SEEDS[0])]
    done, start = 0, time.perf_counter()
    pending = sorted(c for c in checkpoints if c < steps)
    while done < steps:
        report = setup.trainer.iteration()
        stats.append(report.learn)
        done += report.agent_steps
        while pending and done >= pending[0]:
            pending.pop(0)
            rows.append(evaluation("checkpoint", done, EVAL_SEEDS[0]))
    minutes = (time.perf_counter() - start) / 60
    problems = learning_problems(stats)
    for eval_seed in EVAL_SEEDS:
        rows.append(evaluation("trained", done, eval_seed))
    if also_likeliest:
        pick = (0.0, 1.0 if opponent is not None else 0.0)
        rows.append(evaluation("trained_likeliest", done, EVAL_SEEDS[0], pick))
    for row in rows:
        row |= {"train_minutes": round(minutes, 2), "learned": not problems, "problems": "; ".join(problems)}
    return rows
