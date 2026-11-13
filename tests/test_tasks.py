"""Practice tasks: lone-car environments, their measurements and the evaluation that reads them."""

from __future__ import annotations

import copy
import dataclasses
import math
import os
from pathlib import Path

import pytest
import torch

from nitro.config.schema import NitroConfig, PPOConfig
from nitro.envs import EnvPool
from nitro.evaluate import evaluate, team_logits
from nitro.tasks import TASKS
from nitro.training import build

ROOT = Path(__file__).parents[1]
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
pytestmark = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")


def small(task: str, arenas: int = 8) -> NitroConfig:
    config = TASKS[task].settings()
    return dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES),
        run=dataclasses.replace(config.run, device="cpu"),
        env=dataclasses.replace(config.env, num_arenas=arenas, threads=2),
        ppo=PPOConfig(steps_per_iteration=arenas * 16, minibatches=2),
    )


def test_a_lone_car_sees_no_opponents() -> None:
    envs = EnvPool(small("ball_touch"), pin_memory=False)
    assert sum(group.obs.shape[0] for group in envs.groups) == 8, "one car per arena"
    assert envs.obs_layout.opponents == 0 and envs.obs_layout.size == 110, "nitro_v2 without other cars"


def outcome(record: object) -> tuple[object, ...]:
    fields = ("seconds", "first_touch", "boost_collected", "ball_y_after_touch", "terminal")
    values = (getattr(record, f) for f in fields)
    return tuple("nan" if isinstance(v, float) and math.isnan(v) else v for v in values)


def test_evaluation_counts_every_arena_equally_and_repeats_exactly() -> None:
    config = small("ball_touch")
    model = build(config).model
    cpu = torch.device("cpu")
    first = evaluate(model, config, seed=3, episodes_per_arena=2, device=cpu, fp16=False)
    again = evaluate(model, config, seed=3, episodes_per_arena=2, device=cpu, fp16=False)
    assert [(r.env, r.car) for r in first] == [(env, 0) for env in range(8) for _ in range(2)]
    assert [outcome(r) for r in first] == [outcome(r) for r in again], "same seed, same episodes"
    for record in first:
        assert record.seconds <= 10.0 + 1e-6
        # A touch ends a ball-touch episode, so it is touched exactly when it ended early.
        assert record.terminal == (not math.isnan(record.first_touch))


class Constant(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.full((obs.shape[0], 3), self.value), None


def test_the_opponent_drives_exactly_the_orange_cars() -> None:
    obs = torch.zeros(8, 5)  # 2 arenas of 2v2: Blue, Orange, Blue, Orange per arena
    logits = team_logits(Constant(1.0), Constant(2.0), obs, cars=4)
    assert logits[:, 0].tolist() == [1.0, 2.0, 1.0, 2.0] * 2
    assert torch.equal(team_logits(Constant(1.0), None, obs, cars=4), torch.ones(8, 3))
    # The skills battery: the bot drives car 0 of each arena only, teammates and opponents are idle.
    alone = team_logits(Constant(1.0), Constant(2.0), obs, cars=4, bot_cars=(0,))
    assert alone[:, 0].tolist() == [1.0, 2.0, 2.0, 2.0] * 2


def test_an_identical_opponent_changes_nothing() -> None:
    config = small("kickoff")
    model = build(config).model
    cpu = torch.device("cpu")
    alone = evaluate(model, config, seed=5, episodes_per_arena=1, device=cpu, fp16=False)
    paired = evaluate(model, config, 5, 1, cpu, False, opponent=copy.deepcopy(model))
    assert [(r.env, r.car) for r in alone] == [(env, car) for env in range(8) for car in (0, 1)]
    assert [outcome(r) for r in alone] == [outcome(r) for r in paired]


@pytest.mark.parametrize(
    "task",
    ["ball_touch", "recovery", "boost_pickup", "kickoff", "shot", "save", "aerial_touch", "touch_mix"],
)
def test_each_practice_task_trains_and_is_measured(task: str) -> None:
    config = small(task)
    setup = build(config)
    report = setup.trainer.iteration()
    assert report.learn.policy_change > 0
    records = evaluate(
        setup.model, config, seed=0, episodes_per_arena=1, device=torch.device("cpu"), fp16=False
    )
    assert len(records) == 8 * (2 if task == "kickoff" else 1)
    limit = config.terminals.max_episode_seconds
    for record in records:
        assert 0 < record.seconds <= limit + 1e-6
        if task == "recovery" and record.terminal:
            assert record.landed == record.seconds, "a recovery episode ends when the car lands"
        if task == "boost_pickup":
            assert not record.terminal and record.seconds == pytest.approx(limit, abs=0.07)
            assert record.boost_collected >= 0
        if task == "shot":
            assert record.goal >= 0 and record.terminal == (record.goal == 1), "only a goal ends a shot"
        if task == "save":
            assert record.goal <= 0 and record.terminal == (record.goal == -1), "only a goal ends a save"
        if task == "aerial_touch":
            assert record.terminal == (not math.isnan(record.first_touch)), (
                "a touch ends it, a landing cuts it"
            )
