"""ppo.goal_ends: with 'continue' a goal is a cut that bootstraps from the next game's first
state; and the start_value metric. GAE by hand, cuts unchanged, a goal continuing for every car in the game,
'terminal' (the default) learning as before to the bit, the refusals, and the metric by hand."""

from __future__ import annotations

import dataclasses
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro.config import ConfigError, load_config
from nitro.config.loader import parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig
from nitro.learner import Learner, goals_continuing
from nitro.model import ActorCritic
from nitro.ppo import gae
from nitro.rollout import Rollout
from nitro.runs import RunFolder
from nitro.tasks import TASKS, practice_config
from nitro.training import TrainingRun
from test_optimizer_options import ACTIONS, AGENTS, OBS, STEPS, rollout_for

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
GAMMA, LAM = 0.9, 0.8


def every(t: torch.Tensor) -> torch.Tensor:
    return t


def learner_for(goal_ends: str = "terminal", seed: int = 5, **ppo: Any) -> Learner:
    torch.manual_seed(seed)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(16,), value=(16,), shared=(8,)))
    config = PPOConfig(
        **({"goal_ends": goal_ends, "gae_lambda": LAM, "epochs": 1, "minibatches": 2} | ppo)  # type: ignore[arg-type]
    )
    return Learner(model, config, gamma=GAMMA, device=CPU, fp16=False)


def hand_rollout() -> Rollout:
    """4 steps of 2 agents, no episode end: tests mark their goals and cuts."""
    r = Rollout.allocate(4, 2, OBS, ACTIONS, CPU)
    r.rewards.copy_(torch.tensor([[0.5, -1.0], [2.0, 0.25], [-0.5, 1.5], [1.0, -2.0]]))
    r.values.copy_(torch.tensor([[0.3, 0.7], [1.1, -0.4], [0.9, 0.2], [-0.6, 1.3]]))
    r.bootstrap_values.copy_(torch.tensor([0.4, -0.8]))
    return r


def answers(goal_ends: str, r: Rollout) -> tuple[torch.Tensor, ...]:
    """The learner's advantages and value targets for every agent, the rewards taken as scaled."""
    learner = learner_for(goal_ends, popart=False)
    advantages, targets, _ = learner._advantages(r, r.rewards, every, learner._goal_rows(r))
    return advantages, targets


def test_a_goal_bootstraps_from_the_next_games_first_state_and_stops_the_recursion_by_hand() -> None:
    r = hand_rollout()
    r.terminal[1, 0] = True  # agent 0 scores at step 1; steps 2-3 are the next game
    advantages, targets = answers("continue", r)
    v, rw = r.values[:, 0].tolist(), r.rewards[:, 0].tolist()
    a3 = rw[3] + GAMMA * 0.4 - v[3]
    a2 = rw[2] + GAMMA * v[3] - v[2] + GAMMA * LAM * a3
    a1 = rw[1] + GAMMA * v[2] - v[1]  # A1 = delta1 = r1 + gamma V(s2) - V(s1): it stops there
    a0 = rw[0] + GAMMA * v[1] - v[0] + GAMMA * LAM * a1
    expected = torch.tensor([a0, a1, a2, a3])
    torch.testing.assert_close(advantages[:, 0], expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(targets[:, 0], expected + r.values[:, 0], rtol=0, atol=1e-6)
    terminal_advantages, terminal_targets = answers("terminal", r)
    assert torch.equal(advantages[:, 1], terminal_advantages[:, 1]), "agent 1 has no goal: unchanged"
    assert torch.equal(targets[:, 1], terminal_targets[:, 1])
    assert terminal_advantages[1, 0] == pytest.approx(rw[1] - v[1]), "'terminal': nothing after the goal"


def test_a_goal_on_the_last_row_takes_the_bootstrap_value() -> None:
    r = hand_rollout()
    r.terminal[3, 1] = True
    advantages, _ = answers("continue", r)
    v, rw = r.values[:, 1].tolist(), r.rewards[:, 1].tolist()
    a3 = rw[3] + GAMMA * -0.8 - v[3]
    a2 = rw[2] + GAMMA * v[3] - v[2] + GAMMA * LAM * a3
    torch.testing.assert_close(advantages[2:, 1], torch.tensor([a2, a3]), rtol=0, atol=1e-6)


def test_a_cut_is_unchanged() -> None:
    """Agent 1 is cut at step 1 and scores at step 3: the cut and the steps before it are as before, only the
    goal's episode changes. Agent 0 has neither: unchanged."""
    r = hand_rollout()
    r.truncated[1, 1], r.truncated_values[1, 1] = True, 5.0
    r.terminal[3, 1] = True
    continued, continued_targets = answers("continue", r)
    terminal, terminal_targets = answers("terminal", r)
    assert torch.equal(continued[:, 0], terminal[:, 0]) and torch.equal(
        continued_targets[:, 0], terminal_targets[:, 0]
    )
    assert torch.equal(continued[:2, 1], terminal[:2, 1])
    assert continued[1, 1] == pytest.approx(0.25 + GAMMA * 5.0 - -0.4), "the cut's own value, as always"
    assert not torch.equal(continued[2:, 1], terminal[2:, 1])
    learner = learner_for("continue")
    goals = learner._goal_rows(r)
    assert goals is not None and goals.tolist() == [[False, False]] * 3 + [[False, True]]
    shifted = goals_continuing(r, goals)
    assert torch.equal(shifted.truncated_values[1, 1], torch.tensor(5.0)), (
        "a cut keeps its final state's value"
    )
    assert torch.equal(r.terminal[3], torch.tensor([False, True])), "the rollout itself is not changed"


def test_a_goal_continues_for_every_car_in_the_game() -> None:
    """One goal at step 4 ends the game for every car: each one's goal row continues, bootstrapping from the
    next game's first state."""
    r = rollout_for(learner_for().model)  # a goal at step 4 for every agent
    seen: list[tuple[Any, ...]] = []
    learner = learner_for("continue")
    plain = learner._advantages

    def spying(*args: Any) -> Any:
        seen.append((args[1], args[3], *plain(*args)))
        return seen[-1][2:]

    learner._advantages = spying  # type: ignore[method-assign]
    _update(learner, r)
    ((rewards, goals, advantages, _, values),) = seen
    assert torch.equal(goals, r.terminal), "every car's goal row"
    for agent in range(AGENTS):
        expected = rewards[4, agent] + GAMMA * values[5, agent] - values[4, agent]
        assert float(advantages[4, agent]) == pytest.approx(float(expected), rel=0, abs=1e-6), (
            "a goal bootstraps from the next game's first state"
        )


def test_terminal_is_todays_gae_to_the_bit() -> None:
    r = rollout_for(learner_for().model)
    r.truncated[1, :2], r.truncated_values[1, :2] = True, torch.tensor([0.5, -1.5])
    learner = learner_for(popart=False)
    assert PPOConfig().goal_ends == "terminal" and learner._goal_rows(r) is None
    advantages, targets, values = learner._advantages(r, r.rewards, every, learner._goal_rows(r))
    expected = gae(r.rewards, r.values, r.next_values(), r.terminal, r.truncated, GAMMA, LAM)
    assert torch.equal(advantages, expected[0]) and torch.equal(targets, expected[1])
    assert torch.equal(values, r.values)


def _update(learner: Learner, r: Rollout, **kwargs: Any) -> Any:
    torch.manual_seed(1)
    return learner.learn(r, **kwargs)


def _same_weights(a: Learner, b: Learner) -> bool:
    return all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True))


def test_terminal_learns_as_the_code_before_it_to_the_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default against a learner with code taken out (no goal rows, no metric): the same weights,
    optimizer, scaler and statistics after one update. Checked against an earlier commit too."""
    today, before = learner_for(), learner_for()
    monkeypatch.setattr(before, "_goal_rows", lambda rollout: None)
    monkeypatch.setattr(before, "_start_values", lambda *args: torch.zeros(2, dtype=torch.float64))
    r = rollout_for(today.model)
    got, old = _update(today, r), _update(before, r)
    assert _same_weights(today, before)
    for p, q in zip(today.model.parameters(), before.model.parameters(), strict=True):
        for key, value in today.optimizer.state[p].items():
            assert torch.equal(value, before.optimizer.state[q][key]), key
    assert dataclasses.replace(got, start_value=math.nan) == dataclasses.replace(old, start_value=math.nan)
    assert math.isnan(old.start_value) and math.isfinite(got.start_value)


def test_continue_without_a_goal_learns_as_terminal_to_the_bit() -> None:
    r = rollout_for(learner_for().model)
    r.terminal.zero_()
    r.truncated[2, 1:], r.truncated_values[2, 1:] = True, torch.tensor([0.5, -1.5, 2.0])
    terminal, continued = learner_for("terminal"), learner_for("continue")
    assert _update(terminal, r) == _update(continued, r)
    assert _same_weights(terminal, continued)
    # And with its goals (step 4 for every agent) the switch acts.
    scored = rollout_for(learner_for().model)
    terminal, continued = learner_for("terminal"), learner_for("continue")
    _update(terminal, scored)
    _update(continued, scored)
    assert not _same_weights(terminal, continued)


@pytest.mark.parametrize(
    ("other", "ppo", "problem"),
    [
        ("[terminals]\nend_on_touch = true\n", "", r"terminals\.end_on_touch: these end games"),
        ("[terminals]\nend_on_landing = true\n", "", r"terminals\.end_on_landing: these end games"),
        (
            '[model]\nvalue_heads = ["main", "goal"]\n',
            "graphed_steps = false\ngoal_critic_weight = 0.25\n",
            r"ppo\.goal_critic_weight: their targets keep goals as ends",
        ),
        (
            '[model]\nvalue_heads = ["main", "vdag1", "vdag2"]\n',
            "graphed_steps = false\nvdag_beta = 0.3\n",
            r"ppo\.vdag_beta: their targets keep goals as ends",
        ),
    ],
)
def test_what_continue_cannot_be_used_with_is_refused(
    tmp_path: Path, other: str, ppo: str, problem: str
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(other + "[ppo]\n" + ppo, encoding="utf-8")
    load_config(path)  # fine with goals as ends
    path.write_text(other + '[ppo]\ngoal_ends = "continue"\n' + ppo, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert re.search(r"ppo\.goal_ends: 'continue' cannot be used with " + problem, str(caught.value)), str(
        caught.value
    )


def test_continue_is_allowed_in_a_plain_run_and_on_the_shot_and_save_tasks() -> None:
    config = NitroConfig(ppo=PPOConfig(goal_ends="continue"))
    assert config.problems() == []
    for name in ("shot", "save"):
        assert practice_config(TASKS[name], config, seed=0).problems() == [], name


def _fresh_start_rollout(model: ActorCritic) -> Rollout:
    """rollout_for's (a goal at step 4 for all) with each episode's steps counted; agent 1's game is cut at
    step 2 and a new one starts at step 3."""
    r = rollout_for(model)
    r.episode_step.copy_(torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.int32)[:, None].expand(STEPS, AGENTS))
    r.truncated[2, 1] = True
    r.episode_step[3:5, 1] = torch.tensor([0, 1], dtype=torch.int32)
    return r


def test_start_value_is_the_mean_value_of_fresh_starts_learned_from_in_raw_units() -> None:
    """Fresh starts of agents 0, 1 (the one after its cut too) and 2 count; agent 3 is a past version's (not
    driven)."""
    learner = learner_for()
    r = _fresh_start_rollout(learner.model)
    rows = torch.tensor([True, True, True, False])
    values = learner.value_norm.denormalize(r.values).double()
    counted = [(0, 0), (5, 0), (0, 1), (3, 1), (5, 1), (0, 2), (5, 2)]
    hand = sum(float(values[t, n]) for t, n in counted) / len(counted)
    stats = _update(learner, r, rows=rows)
    assert stats.start_value == pytest.approx(hand * stats.reward_scale, rel=1e-12)
    no_starts = rollout_for(learner.model)
    no_starts.episode_step.fill_(3)
    assert math.isnan(_update(learner, no_starts).start_value), "none in the rollout: no number"


def test_the_metric_leaves_learning_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    measured, unmeasured = learner_for(), learner_for()
    monkeypatch.setattr(unmeasured, "_start_values", lambda *args: torch.zeros(2, dtype=torch.float64))
    r = _fresh_start_rollout(measured.model)
    rows = torch.tensor([True, True, True, False])
    with_metric = _update(measured, r, rows=rows)
    without = _update(unmeasured, r, rows=rows)
    assert _same_weights(measured, unmeasured)
    assert dataclasses.replace(with_metric, start_value=0.0) == dataclasses.replace(without, start_value=0.0)


SETTINGS = """
[run]
seed = 3
device = "cpu"
[env]
num_arenas = 4
threads = 2
tick_skip = 8               # each iteration plays past a game's 1 s, so every update has fresh starts
[terminals]
no_touch_seconds = 1
[model]
policy = [16]
value = [16]
shared = []
[ppo]
steps_per_iteration = 160
minibatches = 2
graphed_steps = false
goal_ends = "{ends}"
[[rewards]]
name = "goal"
weight = 10
[[rewards]]
name = "touch_ball"
weight = 1
[[state_setters]]
name = "kickoff"
"""


def _run(tmp_path: Path, name: str, ends: str, goal: bool = True) -> TrainingRun:
    text = SETTINGS.format(ends=ends)
    if not goal:
        text = text.replace('name = "goal"\nweight = 10', 'name = "air"\nweight = 1')
    file = tmp_path / f"{name}.toml"
    file.write_text(text, encoding="utf-8")
    config = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    return TrainingRun(config, RunFolder.create(tmp_path / "runs", name, file))


@NEEDS_MESHES
@pytest.mark.parametrize("ends", ["terminal", "continue"])
def test_a_run_writes_start_value_and_its_share_of_the_goals_weight(tmp_path: Path, ends: str) -> None:
    run = _run(tmp_path, ends, ends)
    rows: list[dict[str, Any]] = []
    learn = run.trainer.learner.learn
    stats: list[Any] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        stats.append(learn(*args, **kwargs))
        return stats[-1]

    run.trainer.learner.learn = recording  # type: ignore[method-assign]
    for _ in range(3):
        rows.append(run.iteration())
    run.metrics.close()
    assert {"start_value", "start_value_share"} <= set(run.metrics.columns)
    for row, learned in zip(rows, stats, strict=True):
        assert row["start_value"] == learned.start_value and math.isfinite(row["start_value"])
        assert row["start_value_share"] == pytest.approx(row["start_value"] / 10, rel=1e-12)
        assert math.isfinite(row["value_loss"])
    without = _run(tmp_path, f"{ends}-no-goal", ends, goal=False)
    assert "start_value" in without.metrics.columns and "start_value_share" not in without.metrics.columns
    without.metrics.close()
