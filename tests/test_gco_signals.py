"""The GCO bot's learning signals: the goal critic, the twin optimistic critics (V-dagger)
and their shaping, the advantage filter and the entropy controller; hand-computed small cases, exact
resumes, the settings refused, and nothing changed with them off. ppo.head_loss_scale: 'raw' is the
step's maths before it (spelled out, `old_losses`, with value_share) to the bit; 'popart' gives every
weight the gradient of the hand sum main + (goal + twins) / PopArt's variance (with value_share below 1 the
critics' parts on the value side's rows), the trunks the gradients of a PopArt on each head with the main
value's spread, raw goal_loss and vdag_loss, a weighted frozen warm-up and an exact resume."""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro.config import ConfigError, load_config
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig, PPOConfig
from nitro.learner import HeadValues, Learner, LearnStats, head_values_on, signal_columns
from nitro.model import ActorCritic, _output
from nitro.ppo import (
    PopArt,
    expectile_loss,
    gae,
    kept_advantages,
    masked_log_probs,
    normalized_entropy,
    policy_loss,
    potential_shaping,
    scaled_shaping,
    twin_target,
    weighted_mean,
)
from nitro.rollout import Rollout
from nitro.runs import RunFolder
from nitro.trainer import Trainer
from nitro.training import TrainingRun
from test_optimizer_options import ACTIONS, AGENTS, OBS, STEPS, rollout_for
from test_residual_heads import HEADS
from test_trainer import CountingEnvs

ROOT = Path(__file__).resolve().parents[1]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
CPU = torch.device("cpu")
MESHES = ROOT / "collision_meshes"
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
EVERY = {
    "goal_critic_weight": 0.25,
    "vdag_beta": 0.3,
    "advantage_keep": 0.5,
    "entropy_target": 0.5,
}


def learner_with(fp16: bool = False, seed: int = 0, **ppo: Any) -> Learner:
    torch.manual_seed(seed)
    model = ActorCritic(OBS, ACTIONS, HEADS)
    config = PPOConfig(**({"epochs": 2, "minibatches": 2, "graphed_steps": False} | ppo))
    learner = Learner(model, config, gamma=0.9, device=CPU, fp16=fp16)
    learner.goal_gamma = 0.99
    return learner


def rollout_and_heads(learner: Learner, seed: int) -> tuple[Rollout, HeadValues]:
    """`rollout_for`'s rollout (a goal at step 4) with agents 0 and 1 cut at step 1, its final states and
    the states after the last step valued as the trainer values them."""
    model = learner.model
    r = rollout_for(model, seed)
    generator = torch.Generator().manual_seed(100 + seed)
    last = torch.randn(AGENTS, OBS, generator=generator)
    cut_obs = torch.randn(2, OBS, generator=generator)
    cut_steps, cut_agents = torch.tensor([1, 1]), torch.tensor([0, 1])
    r.truncated[1, :2] = True
    with torch.no_grad():
        r.truncated_values[1, :2] = model.normalised_value(cut_obs)
        r.bootstrap_values.copy_(model.normalised_value(last))
    heads = head_values_on(
        model, learner.extra_heads, learner.fp16, r, last, (cut_steps, cut_agents, cut_obs)
    )
    return r, heads


def test_the_expectile_loss_by_hand() -> None:
    u = torch.tensor([2.0, -2.0, 0.5, 0.0])
    # tau 0.75: a target above the prediction costs 0.75 u^2, below 0.25 u^2.
    assert torch.equal(expectile_loss(u, 0.75), torch.tensor([3.0, 1.0, 0.1875, 0.0]))
    assert torch.equal(expectile_loss(u, 0.5), 0.5 * u**2), "0.5 is half the squared error"


def test_the_twin_target_takes_the_lower_twin_and_stops_at_every_episode_end() -> None:
    rewards = torch.tensor([1.0, 0.0, 2.0, -1.0])
    one = torch.tensor([3.0, 1.0, 5.0, 7.0])
    two = torch.tensor([2.0, 4.0, 1.0, 9.0])
    terminal = torch.tensor([False, False, True, False])
    truncated = torch.tensor([False, True, False, False])
    y = twin_target(rewards, one, two, terminal, truncated, 0.5)
    # 1 + 0.5 min(3, 2); a cut: the reward alone; a goal: the reward alone; -1 + 0.5 min(7, 9).
    assert torch.equal(y, torch.tensor([2.0, 0.0, 2.0, 2.5]))


def test_the_potential_shaping_by_hand() -> None:
    potential = torch.tensor([1.0, 2.0, 3.0])
    reached = torch.tensor([4.0, 5.0, 6.0])
    terminal = torch.tensor([False, False, True])
    truncated = torch.tensor([False, True, False])
    g = potential_shaping(potential, reached, terminal, truncated, 0.5)
    assert torch.equal(g, torch.tensor([0.5 * 4 - 1, -2.0, -3.0]))


def test_the_shaping_is_centred_scaled_to_beta_and_clamped_at_three_sigma() -> None:
    shaping = torch.tensor([0.0] * 19 + [20.0])
    shaped = scaled_shaping(shaping, torch.tensor(2.0), beta=0.3)
    # Mean 1: centred -1 (x19) and 19; spread sqrt((19 + 361) / 19) = sqrt(20). In units of it -0.2236 and
    # 4.249, the second clamped to 3; times 0.3 x 2.
    unit = -1 / math.sqrt(20)
    expected = torch.tensor([0.6 * unit] * 19 + [0.6 * 3.0])
    torch.testing.assert_close(shaped.added, expected, rtol=0, atol=1e-6)
    assert float(shaped.clamped) == pytest.approx(1 / 20)
    # Nearly flat: divided by at least 0.05 x the advantages' spread, so not blown up to full size.
    flat = scaled_shaping(torch.tensor([0.001, -0.001] * 5), torch.tensor(1.0), beta=0.3)
    torch.testing.assert_close(flat.added, torch.tensor([0.3 * 0.001 / 0.05, -0.3 * 0.001 / 0.05] * 5))


def test_gae_at_the_goal_critics_discount_by_hand() -> None:
    g, lam = 0.9994, 0.95
    rewards = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.tensor([[0.1], [0.2], [0.3]])
    reached = torch.tensor([[0.2], [0.3], [9.0]])  # a goal last: what it reached is worth nothing
    terminal = torch.tensor([[False], [False], [True]])
    advantages, targets = gae(rewards, values, reached, terminal, torch.zeros_like(terminal), g, lam)
    a2 = 1.0 - 0.3
    a1 = (g * 0.3 - 0.2) + g * lam * a2
    a0 = (g * 0.2 - 0.1) + g * lam * a1
    torch.testing.assert_close(advantages.squeeze(1), torch.tensor([a0, a1, a2]), rtol=0, atol=1e-6)
    torch.testing.assert_close(targets, advantages + values)


def test_the_goal_critic_learns_its_own_return_and_adds_a_quarter_of_the_spread() -> None:
    learner = learner_with(goal_critic_weight=0.25)
    assert learner.extra_heads == ("goal",)
    r, heads = rollout_and_heads(learner, 0)
    rewards = learner.reward_scaler(r.rewards, r.terminal | r.truncated)
    advantages, _, values = learner._advantages(r, rewards, lambda t: t)
    signals = learner._signals(r, rewards, lambda t: t, advantages, values, heads)
    assert signals is not None
    goal_a, goal_t = gae(
        rewards, heads.now["goal"], heads.reached["goal"], r.terminal, r.truncated, 0.99, 0.95
    )
    torch.testing.assert_close(signals.targets["goal"], goal_t)
    z = (goal_a - goal_a.mean()) / (goal_a.std() + 1e-8)
    torch.testing.assert_close(signals.extra, 0.25 * advantages.std() * z)
    assert float(signals.extra.std() / advantages.std()) == pytest.approx(0.25, rel=1e-4)
    # The cut steps' final states, not the next step's, are what the goal critic looks ahead to.
    with torch.no_grad():
        following = learner.model.head_values(r.obs[2])["goal"]
    assert not torch.equal(heads.reached["goal"][1, :2], following[:2])
    torch.testing.assert_close(heads.reached["goal"][1, 2:], following[2:])  # a batch's rounding


def test_the_twins_target_has_no_cut_value_in_it_the_designs_truncation_correction() -> None:
    """The design rebuilds each reward from the advantages, r = A - g l (1-d) A' - g (1-d) V' + V, and must
    take g V_trunc off a cut row (else the bot is paid to reach a cut); the twins' target here starts from the
    rollout's own reward, which that corrected rebuild gives back."""
    learner = learner_with(vdag_beta=0.3)
    r, heads = rollout_and_heads(learner, 1)
    rewards = learner.reward_scaler(r.rewards, r.terminal | r.truncated)
    advantages, _, values = learner._advantages(r, rewards, lambda t: t)
    signals = learner._signals(r, rewards, lambda t: t, advantages, values, heads)
    assert signals is not None
    gamma, lam = 0.9, 0.95
    ended = (r.terminal | r.truncated).float()
    following = torch.cat([advantages[1:], torch.zeros_like(advantages[:1])])
    reached = learner.value_norm.denormalize(r.next_values())
    rebuilt = advantages - gamma * lam * (1 - ended) * following - gamma * (1 - ended) * reached + values
    cut_value = learner.value_norm.denormalize(r.truncated_values)
    corrected = rebuilt - gamma * r.truncated.float() * cut_value
    torch.testing.assert_close(corrected, rewards, rtol=0, atol=1e-5)
    assert float((rebuilt - corrected)[1, :2].abs().min()) > 1e-3, "the correction matters on the cut rows"
    lower = torch.minimum(heads.reached["vdag1"], heads.reached["vdag2"])
    expected = corrected + gamma * (1 - ended) * lower
    torch.testing.assert_close(signals.targets["vdag"], expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(signals.targets["vdag"][1, :2], rewards[1, :2])


def test_the_shaping_takes_the_lower_twin_as_its_potential() -> None:
    learner = learner_with(vdag_beta=0.3)
    r, heads = rollout_and_heads(learner, 2)
    rewards = learner.reward_scaler(r.rewards, r.terminal | r.truncated)
    advantages, _, values = learner._advantages(r, rewards, lambda t: t)
    signals = learner._signals(r, rewards, lambda t: t, advantages, values, heads)
    assert signals is not None
    potential = torch.minimum(heads.now["vdag1"], heads.now["vdag2"])
    reached = torch.minimum(heads.reached["vdag1"], heads.reached["vdag2"])
    g = potential_shaping(potential, reached, r.terminal, r.truncated, 0.9)
    expected = scaled_shaping(g.reshape(-1), advantages.std(), 0.3).added.view_as(g)
    torch.testing.assert_close(signals.extra, expected)
    stats = {name: float(value) for name, value in signals.stats.items()}
    assert stats["vdag_dose"] == pytest.approx(float(expected.std() / advantages.std()), rel=1e-5)
    assert stats["vdag_headroom"] == pytest.approx(float((potential - values).clamp_min(0).mean()))


def test_the_filter_keeps_the_largest_advantages_of_both_signs() -> None:
    a = torch.tensor([0.1, -3.0, 2.0, -0.5, 1.0, 0.2])
    kept, share = kept_advantages(a, 0.5)
    assert torch.equal(kept, torch.tensor([0.0, -6.0, 4.0, 0.0, 2.0, 0.0])), "3 of 6, times 6 / 3"
    assert float(share) == pytest.approx(6.0 / 6.8)
    # The clipped loss on everything with the filtered advantages: the gradient of the loss on the kept rows.
    new = torch.randn(6, requires_grad=True)
    old = torch.randn(6)
    policy_loss(new, old, kept, 0.2).loss.backward()
    filtered = new.grad.clone()
    new.grad = None
    rows = torch.tensor([1, 2, 4])
    policy_loss(new[rows], old[rows], a[rows], 0.2).loss.backward()
    torch.testing.assert_close(filtered, new.grad)
    assert torch.equal(kept_advantages(a, 0.01)[0].ne(0), torch.tensor([False, True] + [False] * 4))


def test_the_filter_reaches_the_policy_only_and_reports_what_it_kept() -> None:
    learner = learner_with(advantage_keep=0.5)
    stats = learner.learn(rollout_for(learner.model, 3))
    assert stats.signals["advantage_kept"] == 0.5 and 0.5 < stats.signals["advantage_kept_size"] < 1
    # One learning step with and without it: the value side (every sample) moves alike, the policy not.
    a, b = learner_with(advantage_keep=0.5, seed=4), learner_with(seed=4)
    r = rollout_for(a.model, 3)
    generator = torch.Generator().manual_seed(8)
    advantages, targets = torch.randn(2, STEPS * AGENTS, generator=generator)
    inputs = (
        r.obs.reshape(-1, OBS),
        r.mask.reshape(-1, ACTIONS),
        r.actions.reshape(-1),
        r.log_probs.reshape(-1),
    )
    for one in (a, b):
        one._step(*inputs, advantages, targets, False)
    for name in ("value", "policy"):
        same = all(torch.equal(p, q) for p, q in zip(a.groups[name], b.groups[name], strict=True))
        assert same == (name == "value"), name


@pytest.mark.parametrize("target", [0.99, 0.05])
def test_the_entropy_controller_moves_the_bonus_toward_its_target(target: float) -> None:
    learner = learner_with(entropy_target=target, entropy=0.035)
    before = learner.entropy_coef
    stats = learner.learn(rollout_for(learner.model, 0))
    assert stats.signals["entropy_coef"] == before == 0.035, "the bonus this update used"
    assert learner.entropy_coef == pytest.approx(before * math.exp(target - stats.entropy))
    assert (learner.entropy_coef > before) == (stats.entropy < target)
    fast = learner_with(entropy_target=target, entropy_target_rate=100.0, entropy=0.035)
    fast.learn(rollout_for(fast.model, 0))
    assert fast.entropy_coef == pytest.approx(0.35 if target > 0.9 else 0.00035), "kept within its range"


def test_the_bonus_the_controller_sets_is_the_one_the_next_update_uses() -> None:
    a, b = learner_with(entropy_target=0.99, seed=6), learner_with(seed=6, entropy=0.035)
    torch.manual_seed(2)
    a.learn(rollout_for(a.model, 0))
    torch.manual_seed(2)
    b.learn(rollout_for(b.model, 0))
    assert a.entropy_coef > 0.035
    b.entropy_coef = a.entropy_coef  # a fixed bonus at the controlled value learns the same
    for one in (a, b):
        torch.manual_seed(3)
        one.learn(rollout_for(one.model, 1))
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(p, q)


def test_the_twins_start_afresh_on_schedule_with_a_fresh_optimizer_state() -> None:
    learner = learner_with(vdag_beta=0.3, vdag_reinit_steps=1000, optimizer="muon")
    r, heads = rollout_and_heads(learner, 0)
    learner.learn(r, heads=heads)
    twins = [p for name in ("vdag1", "vdag2") for p in learner.model.value_heads[name].parameters()]
    assert all(p in learner.optimizer.state for p in twins)
    assert not learner.restart_twins(500), "the first call only starts the count"
    assert not learner.restart_twins(1499)
    before = {k: v.clone() for k, v in learner.model.state_dict().items()}
    assert learner.restart_twins(1500) and learner.twins_started == 1500
    after = learner.model.state_dict()
    moved = {k for k in after if not torch.equal(before[k], after[k])}
    assert moved and all(k.startswith(("value_heads.vdag1", "value_heads.vdag2")) for k in moved)
    assert not any(p in learner.optimizer.state for p in twins)
    r, heads = rollout_and_heads(learner, 1)
    assert learner.learn(r, heads=heads).losses_finite, "learns on from the fresh start"
    assert all(p in learner.optimizer.state for p in twins)


@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_a_resumed_learner_with_every_signal_learns_exactly_as_one_that_never_stopped(fp16: bool) -> None:
    resumes_exactly(fp16)


def resumes_exactly(fp16: bool, **extra: Any) -> list[dict[str, float]]:
    """A learner with every signal saved after one update and resumed into one with other starting weights:
    the next update the same to the bit. Returns both learners' signals of that update."""
    signals = EVERY | {"vdag_reinit_steps": 100.0, "optimizer": "muon"} | extra
    straight = learner_with(fp16, **signals)
    straight.restart_twins(0)
    r, heads = rollout_and_heads(straight, 0)
    straight.learn(r, heads=heads)
    saved = copy.deepcopy(straight.state_dict())
    assert saved["entropy_coef"] != 0.035 and saved["twins_started"] == 0

    def segment(learner: Learner) -> dict[str, float]:
        torch.manual_seed(50)
        assert learner.restart_twins(150), "the twins start afresh after the save"
        r, heads = rollout_and_heads(learner, 1)
        stats = learner.learn(r, heads=heads)
        assert set(stats.signals) == set(signal_columns(learner.config))
        assert all(math.isfinite(v) for v in stats.signals.values())
        return stats.signals

    found = [segment(straight)]
    resumed = learner_with(fp16, seed=99, **signals)  # other starting weights: everything from the save
    resumed.load_state_dict(saved)
    found.append(segment(resumed))
    for a, b in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    for p, q in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        for key, value in straight.optimizer.state.get(p, {}).items():
            assert torch.equal(value, resumed.optimizer.state[q][key]), key
    assert straight.entropy_coef == resumed.entropy_coef and resumed.twins_started == 150
    assert found[0] == found[1]
    return found


def test_with_the_signals_off_nothing_changes() -> None:
    """Checked against the code before this change too (the same bits for three learners and a run)."""
    config = PPOConfig()
    assert config.signals_on() == [] and signal_columns(config) == []
    learner = learner_with()
    assert learner.extra_heads == () and learner.entropy_coef == config.entropy
    assert not {"entropy_coef", "twins_started"} & set(learner.state_dict())
    assert not learner.restart_twins(10**12)
    pair = [learner_with(seed=7), learner_with(seed=7)]
    r, heads = rollout_and_heads(pair[0], 0)
    torch.manual_seed(1)
    plain = pair[0].learn(r)
    torch.manual_seed(1)
    given = pair[1].learn(r, heads=heads)  # handed the heads' values anyway: unused
    assert plain.signals == given.signals == {}
    for p, q in zip(pair[0].model.parameters(), pair[1].model.parameters(), strict=True):
        assert torch.equal(p, q)
    with pytest.raises(ValueError, match="heads' values"):
        learner_with(vdag_beta=0.3).learn(rollout_for(learner.model, 0))


def test_the_trainer_values_the_heads_before_the_update_at_the_cut_episodes_final_states() -> None:
    envs = CountingEnvs((2, 3))
    ppo = PPOConfig(epochs=1, minibatches=1, goal_critic_weight=0.25, vdag_beta=0.3, graphed_steps=False)
    torch.manual_seed(0)
    model = ActorCritic(2, 3, HEADS)
    learner = Learner(model, ppo, gamma=0.9, device=CPU, fp16=False)
    learner.goal_gamma = 0.99
    trainer = Trainer(envs, model, learner, 5 * 6, CPU, fp16=False, cuda_graphs=False)
    seen: list[dict[str, Any]] = []
    learn = trainer.learner.learn

    def recording(rollout: Rollout, *args: Any, **kwargs: Any) -> Any:
        heads = kwargs["heads"]
        with torch.no_grad():
            cut_obs = torch.tensor([[1000.0 + 2, 0.0]])  # step 2 is cut (2 % 3 == 2), agent 0's final state
            expected = trainer.model.head_values(cut_obs)
        seen.append({name: (heads.reached[name][2, 0], expected[name][0]) for name in heads.reached})
        return learn(rollout, *args, **kwargs)

    trainer.learner.learn = recording  # type: ignore[method-assign]
    report = trainer.iteration()
    assert seen and set(seen[0]) == {"goal", "vdag1", "vdag2"}
    for got, expected in seen[0].values():
        torch.testing.assert_close(got, expected)
    assert report.learn.losses_finite and {"goal_loss", "vdag_loss"} <= set(report.learn.signals)


SETTINGS = """
[run]
seed = 7
device = "cpu"
deterministic = true
[env]
num_arenas = 8
threads = 4
[terminals]
no_touch_seconds = 2
[model]
shared = [16]
policy = [16, 16]
value = [16]
value_head_layers = [16]
value_heads = ["main", "goal", "vdag1", "vdag2"]
residual = true
[ppo]
steps_per_iteration = 320
minibatches = 2
graphed_steps = false
optimizer = "muon"
goal_critic_weight = 0.25
vdag_beta = 0.3
vdag_reinit_steps = 600
advantage_keep = 0.5
entropy_target = 0.5
[[state_setters]]
name = "kickoff"
"""


def _run(
    tmp_path: Path, name: str, state: dict[str, Any] | None = None, settings: str = SETTINGS
) -> TrainingRun:
    file = tmp_path / "settings.toml"
    file.write_text(settings, encoding="utf-8")
    config = parse_toml(settings, NitroConfig, file=file, base_dir=tmp_path)
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    folder = RunFolder(tmp_path / "runs" / name) if state else RunFolder.create(tmp_path / "runs", name, file)
    return TrainingRun(config, folder, state)


@pytest.fixture
def kept_determinism() -> Any:
    """run.deterministic turns PyTorch's deterministic algorithms on for the process; set back after."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


@NEEDS_MESHES
@pytest.mark.usefixtures("kept_determinism")
def test_a_run_with_every_signal_logs_them_and_resumes_exactly(tmp_path: Path) -> None:
    straight = _run(tmp_path, "straight")
    straight.iteration()
    straight.checkpoint()
    rows = [straight.iteration() for _ in range(2)]
    straight.metrics.close()
    for column in signal_columns(straight.config.ppo):
        assert all(math.isfinite(row[column]) for row in rows), column
    stopped = _run(tmp_path, "stopped")
    stopped.iteration()
    stopped.checkpoint(continue_after=False)
    stopped.metrics.close()
    folder = RunFolder(tmp_path / "runs" / "stopped")
    resumed = _run(tmp_path, "stopped", folder.load_latest())
    resumed_rows = [resumed.iteration() for _ in range(2)]
    resumed.metrics.close()
    assert resumed.trainer.learner.twins_started == straight.trainer.learner.twins_started == 640
    for name, weights in straight.model.state_dict().items():
        assert torch.equal(weights, resumed.model.state_dict()[name]), f"{name} differs after resuming"
    assert [r["entropy_coef"] for r in rows] == [r["entropy_coef"] for r in resumed_rows]


def test_the_gco_starter_turns_every_signal_on() -> None:
    config = load_config(GCO_STARTER)
    ppo = config.ppo
    assert ppo.signals_on() == list(EVERY)
    assert (ppo.goal_critic_weight, ppo.vdag_beta, ppo.vdag_expectile, ppo.advantage_keep) == (
        0.25,
        0.3,
        0.75,
        0.5,
    )
    assert ppo.goal_critic_half_life == 77.0 and ppo.vdag_reinit_steps == 500_000_000
    gamma = 0.5 ** (8 / (120 * ppo.goal_critic_half_life))
    assert round(gamma, 4) == 0.9994
    assert ppo.entropy_target_min == 1.0, "the bonus never below the design's 0.035"
    assert ppo.head_loss_scale == "raw", "the extra heads' losses as they were"
    text = GCO_STARTER.read_text(encoding="utf-8").replace("[run]\n", "[run]\ncards = 2\n", 1)
    two = parse_toml(text, NitroConfig, file=GCO_STARTER, base_dir=GCO_STARTER.parent)
    assert two.run.cards == 2 and two.ppo.signals_on() == list(EVERY), "the starter on two cards"


@pytest.mark.parametrize("target", [0.99, 0.05])
def test_the_entropy_floor_keeps_the_bonus_at_least_its_share_of_ppo_entropy(target: float) -> None:
    """ppo.entropy_target_min: at 1 the bonus never falls below ppo.entropy (above the target it
    stays there), and still rises below the target; a saved bonus under a raised floor is lifted on load."""
    fast = learner_with(
        entropy_target=target, entropy_target_rate=100.0, entropy=0.035, entropy_target_min=1.0
    )
    fast.learn(rollout_for(fast.model, 0))
    assert fast.entropy_coef == pytest.approx(0.35 if target > 0.9 else 0.035)
    low = learner_with(entropy_target=0.05, entropy_target_rate=100.0, entropy=0.035)
    low.learn(rollout_for(low.model, 0))
    assert low.entropy_coef == pytest.approx(0.00035), "the default floor: ppo.entropy / 100, as before"
    raised = learner_with(entropy_target=0.05, entropy=0.035, entropy_target_min=1.0)
    raised.load_state_dict(low.state_dict())
    assert raised.entropy_coef == 0.035, "the floor holds from the first update after a resume"
    kept = learner_with(entropy_target=0.05, entropy=0.035)
    kept.load_state_dict(low.state_dict())
    assert kept.entropy_coef == low.entropy_coef, "within the range: resumed exactly"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[ppo]\ngraphed_steps = false\ngoal_critic_weight = 0.25\n", 'needs "goal" in model.value_heads'),
        (
            '[model]\nvalue_heads = ["main", "vdag1"]\n[ppo]\ngraphed_steps = false\nvdag_beta = 0.3\n',
            'ppo.vdag_beta: needs "vdag1" and "vdag2"',
        ),
        ("[ppo]\ngraphed_steps = false\nentropy_target = 0.5\nentropy = 0\n", "needs ppo.entropy above 0"),
        ("[ppo]\ngraphed_steps = false\nentropy_target = 0.5\nentropy_target_min = 0\n", "must be above 0"),
        (
            "[ppo]\ngraphed_steps = false\nentropy_target = 0.5\nentropy_target_min = 10.5\n",
            "entropy_target_min",
        ),
        (
            '[model]\nvalue_heads = ["main", "goal"]\n[ppo]\ngraphed_steps = false\n'
            'goal_critic_weight = 0.25\npopart = false\nhead_loss_scale = "popart"\n',
            "ppo.head_loss_scale: 'popart' needs ppo.popart = true",
        ),
        (
            '[ppo]\ngraphed_steps = false\nhead_loss_scale = "popart"\n',
            "ppo.head_loss_scale: 'popart' does nothing without the goal critic or the twins",
        ),
    ],
)
def test_settings_that_cannot_work_are_refused(tmp_path: Path, text: str, problem: str) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert re.search(problem, str(caught.value)), str(caught.value)


# --- ppo.head_loss_scale ---------------------------------------------------------------------------


def old_losses(
    learn: Learner,
    obs: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    targets: torch.Tensor,
    value_only: bool,
    samples: torch.Tensor | None = None,
    head_targets: dict[str, torch.Tensor] | None = None,
    entropy_coef: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`Learner._losses` as it was in an earlier version (with value_share), spelled
    out: what 'raw' must give to the bit."""
    model = learn.model
    outputs: dict[str, torch.Tensor] = {}
    value_weights = samples
    with torch.autocast(learn.device.type, dtype=torch.float16, enabled=learn.fp16):
        if not value_only and learn.config.value_share < 1:
            k = max(1, math.ceil(learn.config.value_share * obs.shape[0]))
            features = model.shared(obs)
            logits = model.logits_from(features)
            if head_targets is not None:
                outputs = model.head_values_from(features[:k])
                predicted = outputs["main"]
                head_targets = {name: t[:k] for name, t in head_targets.items()}
            else:
                predicted = model.value_from(features[:k])
            targets = targets[:k]
            value_weights = learn._value_rows(samples, k)
        elif head_targets is not None:
            if value_only and learn.policy_frozen and len(model.shared):
                with torch.no_grad():
                    features = model.shared(obs)
            else:
                features = model.shared(obs)
            outputs = model.head_values_from(features)
            predicted = outputs["main"]
            if not value_only:
                logits = model.logits_from(features)
        elif value_only:
            predicted = model.normalised_value(obs, hold_trunk=learn.policy_frozen)
        else:
            logits, predicted = model(obs)
    value_loss = weighted_mean((predicted - targets).pow(2), value_weights)
    head_losses = learn._head_losses(outputs, head_targets, value_weights)
    if value_only:
        loss = value_loss + sum(head_losses, torch.zeros((), device=learn.device))
        zero = torch.zeros((), device=learn.device)
        measured = torch.stack([zero, value_loss.detach(), zero, zero, zero])
    else:
        config = learn.config
        log_probs = masked_log_probs(logits, mask)
        new_log_prob = log_probs.gather(1, actions[:, None]).squeeze(1)
        entropy = weighted_mean(normalized_entropy(log_probs, mask, config.entropy_normalization), samples)
        policy = policy_loss(new_log_prob, old_log_probs, advantages, config.clip_range, samples)
        bonus = learn.entropy_coef if entropy_coef is None else entropy_coef
        loss = policy.loss - bonus * entropy + value_loss
        for head_loss in head_losses:
            loss = loss + head_loss
        measured = torch.stack(
            [
                policy.loss.detach(),
                value_loss.detach(),
                entropy.detach(),
                policy.approx_kl,
                policy.clip_fraction,
            ]
        )
    if head_targets is not None:
        names = list(head_targets)
        reported = [h.detach() / (2 if n == "vdag" else 1) for n, h in zip(names, head_losses, strict=True)]
        measured = torch.cat([measured, torch.stack(reported)])
    return loss, measured


def three_updates(learn: Learner) -> list[LearnStats]:
    stats = []
    for number in range(3):
        torch.manual_seed(20 + number)
        r, heads = rollout_and_heads(learn, number)
        torch.manual_seed(30 + number)  # the same minibatches for every learner
        stats.append(learn.learn(r, heads=heads))
    return stats


def assert_same(a: Learner, b: Learner, a_stats: list[LearnStats], b_stats: list[LearnStats]) -> None:
    """Weights, the optimizer's state and every measurement, to the bit (repr: NaN equals NaN)."""
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(p, q)
    first, second = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    assert first.keys() == second.keys()
    for key, state in first.items():
        assert state.keys() == second[key].keys()
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, second[key][name]), (key, name)
            else:
                assert value == second[key][name], (key, name)
    assert repr(a_stats) == repr(b_stats)
    assert a.entropy_coef == b.entropy_coef


RAW_CASES = {  # fp16, a frozen warm-up, ppo.value_share
    "fp32": (False, False, 1.0),
    "fp16": (True, False, 1.0),
    "frozen warm-up": (False, True, 1.0),
    "value_share 0.5": (False, False, 0.5),
    "value_share 0.5 fp16": (True, False, 0.5),
}


@pytest.mark.parametrize("case", list(RAW_CASES))
def test_raw_head_losses_are_the_steps_maths_before_d258_to_the_bit(case: str) -> None:
    """The setting left out and an explicit 'raw' give, over three updates with the extra heads, Muon and
    every signal (and a frozen warm-up's value-only steps, or value_share below 1), the step's maths
    in an earlier version (`old_losses`) to the bit, and never read the weight."""
    fp16, frozen, share = RAW_CASES[case]
    signals = EVERY | {"optimizer": "muon", "value_share": share}
    runs = []
    for scale, old in ((None, True), (None, False), ("raw", False)):
        chosen = {} if scale is None else {"head_loss_scale": scale}
        learn = learner_with(fp16, seed=5, **signals, **chosen)
        learn.policy_frozen = frozen
        if old:
            learn._losses = functools.partial(old_losses, learn)  # type: ignore[method-assign]
        else:

            def never() -> torch.Tensor:
                raise AssertionError("'raw' reads no weight")

            learn._head_loss_weight = never  # type: ignore[method-assign]
        runs.append((learn, three_updates(learn)))
    (reference, expected), *others = runs
    assert all(math.isfinite(stats.signals["goal_loss"]) for stats in expected)
    assert "head_loss_weight" not in expected[0].signals
    for learn, stats in others:
        assert_same(reference, learn, expected, stats)


VARIANCE = 21.2  # PopArt's variance on a long GCO run


def popart_learner(scale: str = "popart", seed: int = 0, **ppo: Any) -> Learner:
    learn = learner_with(seed=seed, goal_critic_weight=0.25, vdag_beta=0.3, head_loss_scale=scale, **ppo)
    assert isinstance(learn.value_norm, PopArt)
    learn.value_norm.mean.fill_(0.7)
    learn.value_norm.mean_sq.fill_(0.7**2 + VARIANCE)
    return learn


def popart_batch(learn: Learner, rows: int = 40) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """A step's inputs (old likelihoods near the policy's, so few samples are clipped) and the extra heads'
    targets in reward units (a few times the main value's normalised ones)."""
    generator = torch.Generator().manual_seed(11)
    obs = torch.randn(rows, OBS, generator=generator)
    mask = torch.rand(rows, ACTIONS, generator=generator) < 0.8
    mask[:, 0] = True
    actions = (torch.rand(rows, ACTIONS, generator=generator) * mask).argmax(1)
    with torch.no_grad():
        log_probs = masked_log_probs(learn.model.logits(obs), mask)
    old = log_probs.gather(1, actions[:, None]).squeeze(1) + 0.05 * torch.randn(rows, generator=generator)
    advantages, targets = torch.randn(rows, generator=generator), torch.randn(rows, generator=generator)
    heads = {name: 4 * torch.randn(rows, generator=generator) for name in ("goal", "vdag")}
    return (obs, mask, actions, old, advantages, targets), heads


def hand_parts(
    learn: Learner,
    model: ActorCritic,
    batch: tuple[torch.Tensor, ...],
    heads: dict[str, torch.Tensor],
    frozen: bool = False,
    rows: int | None = None,
) -> dict[str, torch.Tensor]:
    """Each loss of a step on its own, written out: the policy's (with the entropy bonus; not when `frozen`,
    a frozen warm-up's value-only step, whose shared trunk is held), the main value's, the goal critic's and
    the two twins' together (on the first `rows` only, if given: value side)."""
    obs, mask, actions, old, advantages, targets = batch
    config = learn.config
    if frozen:
        with torch.no_grad():
            features = model.shared(obs)
    else:
        features = model.shared(obs)
    cut = slice(None) if rows is None else slice(rows)
    outputs = model.head_values_from(features[cut])
    heads = {name: t[cut] for name, t in heads.items()}
    twins = [
        expectile_loss(heads["vdag"] - outputs[n], config.vdag_expectile).mean() for n in ("vdag1", "vdag2")
    ]
    parts = {
        "main": (outputs["main"] - targets[cut]).pow(2).mean(),
        "goal": (heads["goal"] - outputs["goal"]).pow(2).mean(),
        "twins": twins[0] + twins[1],
    }
    if not frozen:
        log_probs = masked_log_probs(model.logits_from(features), mask)
        new = log_probs.gather(1, actions[:, None]).squeeze(1)
        entropy = normalized_entropy(log_probs, mask, config.entropy_normalization).mean()
        policy = policy_loss(new, old, advantages, config.clip_range).loss
        parts["policy"] = policy - learn.entropy_coef * entropy
    return parts


def gradients(model: ActorCritic, loss: torch.Tensor) -> dict[str, torch.Tensor | None]:
    names, params = zip(*model.named_parameters(), strict=True)
    found = torch.autograd.grad(loss, params, allow_unused=True)
    return dict(zip(names, found, strict=True))


def variance_of(learn: Learner) -> torch.Tensor:
    norm = learn.value_norm
    assert isinstance(norm, PopArt)
    return norm.mean_sq - norm.mean**2


def test_popart_gives_every_weight_the_gradient_of_main_plus_the_heads_over_poparts_variance() -> None:
    learn = popart_learner()
    assert float(variance_of(learn)) == pytest.approx(VARIANCE)
    batch, heads = popart_batch(learn)
    loss, _ = learn._losses(*batch, False, head_targets=heads)
    got = gradients(learn.model, loss)
    parts = hand_parts(learn, learn.model, batch, heads)
    by_hand = parts["policy"] + parts["main"] + (parts["goal"] + parts["twins"]) / variance_of(learn)
    expected = gradients(learn.model, by_hand)
    trunks = [f"shared.{n}" for n, _ in learn.model.shared.named_parameters()]
    trunks += [f"value.{n}" for n, _ in learn.model.value.named_parameters()]
    assert trunks and all(got[name] is not None for name in trunks), "every trunk weight learns"
    for name, g in got.items():
        e = expected[name]
        assert (g is None) == (e is None), name
        if g is not None and e is not None:
            torch.testing.assert_close(g, e, rtol=1e-6, atol=1e-7 * float(e.abs().max()), msg=name)
    raw = popart_learner("raw")
    raw_loss, _ = raw._losses(*batch, False, head_targets=heads)
    plain = gradients(raw.model, raw_loss)
    for name in trunks:  # the raw losses' trunk gradients are another sum
        assert not torch.allclose(plain[name], got[name], rtol=1e-3), name  # type: ignore[arg-type]


def test_for_the_trunks_popart_head_losses_are_a_popart_per_head() -> None:
    """Multi-task PopArt (Hessel et al. 2019) built by hand: each extra head predicts its targets less its
    own mean, over the main value's spread (its output layer and its targets so divided). Every weight but
    those output layers then gets the gradient 'popart' gives it, to FP32's rounding; the heads' means play
    no part."""
    learn = popart_learner()
    batch, heads = popart_batch(learn)
    loss, _ = learn._losses(*batch, False, head_targets=heads)
    got = gradients(learn.model, loss)
    norm = learn.value_norm
    assert isinstance(norm, PopArt)
    sigma = norm.std
    centres = {"goal": 1.3, "vdag": -0.4}  # the twins learn one target, so one PopArt's statistics
    per_head = copy.deepcopy(learn.model)
    with torch.no_grad():
        for name in ("goal", "vdag1", "vdag2"):
            layer = _output(per_head.value_heads[name])
            layer.weight.div_(sigma)
            layer.bias.sub_(centres["goal" if name == "goal" else "vdag"]).div_(sigma)
    normalised = {name: (t - centres[name]) / sigma for name, t in heads.items()}
    parts = hand_parts(learn, per_head, batch, normalised)
    expected = gradients(per_head, parts["policy"] + parts["main"] + parts["goal"] + parts["twins"])
    outputs = {
        f"value_heads.{name}.{len(learn.model.value_heads[name]) - 1}.{kind}"
        for name in ("goal", "vdag1", "vdag2")
        for kind in ("weight", "bias")
    }
    assert outputs < set(got)
    compared = 0
    for name, g in got.items():
        e = expected[name]
        if name in outputs or g is None or e is None:
            continue
        torch.testing.assert_close(g, e, rtol=1e-5, atol=1e-6 * float(e.abs().max()), msg=name)
        compared += name.startswith(("shared.", "value."))
    assert compared == sum(1 for _ in learn.model.shared.parameters()) + sum(
        1 for _ in learn.model.value.parameters()
    )


def test_with_value_share_popart_weighs_the_heads_on_the_value_sides_rows_only() -> None:
    """A joint step's critics learn from its first rows (value_share 0.5: 20 of 40), the extra
    heads' parts weighed by 1 / PopArt's variance; every weight's gradient is that hand sum's, the reported
    losses are raw and over those rows, and taking every row instead would be another gradient."""
    learn, raw = popart_learner(value_share=0.5), popart_learner("raw", value_share=0.5)
    batch, heads = popart_batch(learn)
    loss, measured = learn._losses(*batch, False, head_targets=heads)
    got = gradients(learn.model, loss)
    parts = hand_parts(learn, learn.model, batch, heads, rows=20)
    by_hand = parts["policy"] + parts["main"] + (parts["goal"] + parts["twins"]) / variance_of(learn)
    expected = gradients(learn.model, by_hand)
    for name, g in got.items():
        e = expected[name]
        assert (g is None) == (e is None), name
        if g is not None and e is not None:
            torch.testing.assert_close(g, e, rtol=1e-6, atol=1e-7 * float(e.abs().max()), msg=name)
    _, raw_measured = raw._losses(*batch, False, head_targets=heads)
    assert torch.equal(measured, raw_measured)
    names = list(heads)
    torch.testing.assert_close(measured[5 + names.index("goal")], parts["goal"].detach())
    torch.testing.assert_close(measured[5 + names.index("vdag")], parts["twins"].detach() / 2)
    every = hand_parts(learn, learn.model, batch, heads)
    on_every_row = every["policy"] + every["main"] + (every["goal"] + every["twins"]) / variance_of(learn)
    plain = gradients(learn.model, on_every_row)
    for name, _ in learn.model.value.named_parameters():
        assert not torch.allclose(plain[f"value.{name}"], got[f"value.{name}"], rtol=1e-3), name  # type: ignore[arg-type]


def test_goal_loss_and_vdag_loss_stay_raw_and_the_weight_is_logged() -> None:
    raw, popart = popart_learner("raw"), popart_learner("popart")
    batch, heads = popart_batch(raw)
    for value_only in (False, True):
        raw_loss, raw_measured = raw._losses(*batch, value_only, head_targets=heads)
        popart_loss, popart_measured = popart._losses(*batch, value_only, head_targets=heads)
        assert torch.equal(raw_measured, popart_measured), value_only
        assert float(popart_loss.detach()) < float(raw_loss.detach()), value_only
    # One step per update: its columns are the losses before that step, the same in both modes.
    found, learners = [], []
    for scale in ("raw", "popart"):
        learn = learner_with(
            seed=3, goal_critic_weight=0.25, vdag_beta=0.3, head_loss_scale=scale, epochs=1, minibatches=1
        )
        torch.manual_seed(4)
        r, h = rollout_and_heads(learn, 0)
        torch.manual_seed(5)
        found.append(learn.learn(r, heads=h).signals)
        learners.append(learn)
    columns = ("goal_loss", "vdag_loss", "goal_explained_variance", "vdag_dose", "vdag_headroom")
    assert {c: found[0][c] for c in columns} == {c: found[1][c] for c in columns}
    assert "head_loss_weight" not in found[0]
    assert found[1]["head_loss_weight"] == pytest.approx(1 / float(variance_of(learners[1])), rel=1e-6)
    assert set(found[1]) == set(signal_columns(learners[1].config))
    moved = [not torch.equal(p, q) for p, q in zip(*(m.model.parameters() for m in learners), strict=True)]
    assert any(moved), "the step itself differs"


def test_a_frozen_warm_ups_value_only_steps_weigh_the_heads_too() -> None:
    learn = popart_learner()
    learn.policy_frozen = True
    batch, heads = popart_batch(learn)
    loss, _ = learn._losses(*batch, True, head_targets=heads)
    got = gradients(learn.model, loss)
    parts = hand_parts(learn, learn.model, batch, heads, frozen=True)
    expected = gradients(learn.model, parts["main"] + (parts["goal"] + parts["twins"]) / variance_of(learn))
    for name, g in got.items():
        e = expected[name]
        assert (g is None) == (e is None), name
        assert (g is None) == name.startswith(("shared.", "policy.")), name  # the shared trunk is held
        if g is not None and e is not None:
            torch.testing.assert_close(g, e, rtol=1e-6, atol=1e-7 * float(e.abs().max()), msg=name)
    # A whole frozen warm-up update: the value side moves otherwise than 'raw', the policy's side not at all.
    learners = [
        learner_with(seed=6, goal_critic_weight=0.25, vdag_beta=0.3, head_loss_scale=s)
        for s in ("raw", "popart")
    ]
    before = {n: p.clone() for n, p in learners[0].model.named_parameters()}
    for one in learners:
        one.policy_frozen = True
        torch.manual_seed(7)
        r, h = rollout_and_heads(one, 0)
        torch.manual_seed(8)
        one.learn(r, heads=h)
    after = [dict(one.model.named_parameters()) for one in learners]
    for name, start in before.items():
        if name.startswith(("shared.", "policy.")):
            assert torch.equal(after[0][name], start) and torch.equal(after[1][name], start), name
    assert not torch.equal(after[0]["value.0.weight"], after[1]["value.0.weight"])


def test_a_resumed_learner_with_popart_head_losses_learns_exactly_as_one_that_never_stopped() -> None:
    found = resumes_exactly(False, head_loss_scale="popart")
    assert 0 < found[0]["head_loss_weight"] < math.inf


def test_popart_head_losses_need_popart_and_go_with_the_gco_starter() -> None:
    with pytest.raises(ValueError, match=r"needs ppo\.popart = true"):
        learner_with(goal_critic_weight=0.25, popart=False, head_loss_scale="popart")
    text = GCO_STARTER.read_text(encoding="utf-8")
    text = text.replace("[ppo]\n", '[ppo]\nhead_loss_scale = "popart"\n', 1)
    config = parse_toml(text, NitroConfig, file=GCO_STARTER, base_dir=GCO_STARTER.parent)
    assert config.ppo.head_loss_scale == "popart"
    assert signal_columns(config.ppo) == [
        "goal_explained_variance",
        "goal_loss",
        "vdag_loss",
        "vdag_dose",
        "vdag_headroom",
        "vdag_clamped",
        "advantage_kept",
        "advantage_kept_size",
        "entropy_coef",
        "head_loss_weight",
    ]


@NEEDS_MESHES
@pytest.mark.usefixtures("kept_determinism")
def test_a_run_with_popart_head_losses_logs_the_weight(tmp_path: Path) -> None:
    settings = SETTINGS.replace("[ppo]\n", '[ppo]\nhead_loss_scale = "popart"\n', 1)
    run = _run(tmp_path, "popart", settings=settings)
    rows = [run.iteration() for _ in range(2)]
    run.metrics.close()
    norm = run.trainer.learner.value_norm
    assert isinstance(norm, PopArt)
    assert all(math.isfinite(row["head_loss_weight"]) and row["head_loss_weight"] > 0 for row in rows)
    assert rows[-1]["head_loss_weight"] == pytest.approx(1 / float(norm.std**2), rel=1e-6)
