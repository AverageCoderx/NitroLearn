"""`nitro skills`: the idle opponent, the task situations played with the bot's own game settings, the CSV;
and a real tiny battery when the collision meshes are present."""

from __future__ import annotations

import csv
import dataclasses
import os
from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.config import default_config
from nitro.eval import load_side
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.runs import RunFolder, fixed_settings
from nitro.skills import (
    DEFAULT_TASKS,
    Idle,
    SkillResult,
    append_csv,
    battery_config,
    neutral_action,
    run_battery,
    save_results,
)
from nitro.tasks import TASKS
from nitro.weakness import read_report

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


def test_the_idle_opponent_always_takes_the_do_nothing_action() -> None:
    index = neutral_action()
    assert _core.action_controls(index) == [0.0] * 8
    assert index == 8, "throttle 0, steer 0, no boost, no handbrake in the ground block"
    idle = Idle(_core.num_actions())
    obs = torch.zeros(5, 7)
    logits, values = idle(obs)
    assert logits.shape == (5, _core.num_actions()) and torch.equal(values, torch.zeros(5))
    mask = torch.ones(5, _core.num_actions(), dtype=torch.bool)
    probs = masked_log_probs(logits, mask).exp()
    assert probs[:, index].min() > 0.999


def test_task_situations_keep_the_bots_game_and_the_tasks_rules() -> None:
    bot = default_config(Path.cwd())
    bot = dataclasses.replace(
        bot,
        env=dataclasses.replace(
            bot.env,
            observation="nitro_v2",
            tick_skip=4,
            action_delay=2,
            groups=3,
            threads=5,
        ),
        model=dataclasses.replace(bot.model, policy=(64,), value=(64,)),
    )
    config = battery_config(TASKS["ball_touch"], bot)
    assert config.env.opponents is True and config.env.team_size == 1, (
        "the bot's game, so its observation fits"
    )
    assert (config.env.observation, config.env.tick_skip, config.env.action_delay) == ("nitro_v2", 4, 2)
    assert (config.env.groups, config.env.threads) == (3, 5)
    assert config.model.policy == (64,) and config.paths == bot.paths
    task = TASKS["ball_touch"].settings()
    assert config.terminals == task.terminals and config.rewards == task.rewards
    assert config.state_setters == task.state_setters and config.env.num_arenas == task.env.num_arenas
    assert "kickoff" not in DEFAULT_TASKS and set(DEFAULT_TASKS) <= set(TASKS)


def test_results_are_read_against_the_bar_and_appended(tmp_path: Path) -> None:
    good = SkillResult("ball_touch", "time_to_touch", 2.0, "s", 3.0, False, 512, {"touch_rate": 0.9})
    weak = SkillResult("shot", "goal_rate", 0.2, "", 0.45, True, 512, {})
    assert good.above_bar and not weak.above_bar
    assert "has the skill" in good.line() and "touch_rate 0.9" in good.line()
    assert "not yet" in weak.line()
    # A time is better lower: under the bar has it, over the bar not yet; a score keeps above and below.
    slow = SkillResult("ball_touch", "time_to_touch", 4.0, "s", 3.0, False, 512, {})
    strong = SkillResult("shot", "goal_rate", 0.6, "", 0.45, True, 512, {})
    assert "(bar 3s; under the bar: has the skill)" in good.line()
    assert "(bar 3s; over the bar: not yet)" in slow.line()
    assert "(bar 0.45; above the bar: has the skill)" in strong.line()
    assert "(bar 0.45; below the bar: not yet)" in weak.line()
    path = tmp_path / "skills.csv"
    append_csv(path, 5_000_000, [good, weak], seed=1_000_001)
    append_csv(path, 10_000_000, [good], seed=1_000_001)
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(r["agent_steps"], r["task"], r["above_bar"]) for r in rows] == [
        ("5000000", "ball_touch", "True"),
        ("5000000", "shot", "False"),
        ("10000000", "ball_touch", "True"),
    ]


def save(folder: RunFolder, agent_steps: int) -> None:
    config = folder.load_settings()
    torch.manual_seed(1)
    model = ActorCritic(148, 90, config.model)
    state = {
        "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": agent_steps,
        "iterations": 1,
        "settings": fixed_settings(config),
    }
    folder.save_checkpoint(state, agent_steps, keep=5, milestone_steps=0)


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_saved_bot_is_scored_on_two_skills(tmp_path: Path) -> None:
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nthreads = 2\n[model]\npolicy = [8]\nvalue = [8]\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    save(folder, 500)
    side = load_side(folder, None)
    results = run_battery(side, ("ball_touch", "recovery", "fifty_fifty"), games=4, seed=3)
    assert [r.task for r in results] == ["ball_touch", "recovery", "fifty_fifty"]
    for result in results:
        assert result.games == 4, "one situation per arena, the bot's car only"
        assert result.value >= 0 and isinstance(result.above_bar, bool)
        assert result.weak is not None and result.weak["situations"] == 4
    again = run_battery(side, ("fifty_fifty",), games=4, seed=3)
    assert again[0].value == results[2].value, "same seed, same score"
    save_results(folder.root, 500, results, 3, 4)
    assert len(read_report(folder.root / "weakness.json")[0]["tasks"]) == 3
    lone = dataclasses.replace(
        side,
        config=dataclasses.replace(side.config, env=dataclasses.replace(side.config.env, opponents=False)),
    )
    skipped: list[str] = []
    assert run_battery(lone, ("kickoff_duel",), games=4, seed=3, skipped=skipped) == []
    assert skipped == [
        "kickoff_duel: left out, it needs an opponent and this bot plays without one (env.opponents = false)"
    ]


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_training_scores_its_saves_on_a_schedule_and_survives_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from nitro import skills
    from nitro.training import TrainingRun

    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n'
        '[run]\ndevice = "cpu"\nskills_every = 2\nskills_games = 8\n'
        "[env]\nnum_arenas = 2\nthreads = 2\n[model]\npolicy = [8]\nvalue = [8]\n"
        "[ppo]\nsteps_per_iteration = 256\nminibatches = 2\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    run = TrainingRun(folder.load_settings(), folder)
    run.iteration()
    caplog.set_level("INFO", logger="nitro")
    run.checkpoint()
    assert not (folder.root / "skills.csv").exists(), "every second save"
    run.iteration()
    run.checkpoint()
    with (folder.root / "skills.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [r["task"] for r in rows] == list(DEFAULT_TASKS) and {r["games"] for r in rows} == {"8"}
    assert len(read_report(folder.root / "weakness.json")) == 1
    assert sum("Skills at" in r.message for r in caplog.records) == 1

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no room for the cars")

    monkeypatch.setattr(skills, "run_battery", broken)
    run.iteration()
    run.checkpoint()
    run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    assert any("training carries on" in r.message and "no room" in r.message for r in caplog.records)
