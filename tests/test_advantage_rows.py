"""ppo.advantage_keep_rest: with ppo.advantage_keep, the learning steps run on the kept
samples plus a weighted random share of the others. Each step's policy term is the mean over its kept samples
(advantage_keep's step exactly at minibatches 1), the value, heads' and entropy means unbiased; at 1 (the
default) the new code is never entered; resumes exact; settings refused."""

from __future__ import annotations

import copy
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
from nitro.learner import Learner, _Rows, signal_columns
from nitro.metrics import CHARTS
from nitro.model import ActorCritic
from nitro.ppo import kept_advantages, kept_rows, masked_log_probs, normalized_entropy
from nitro.rollout import Rollout
from nitro.runs import RunFolder
from nitro.training import TrainingRun
from test_gco_signals import EVERY, SETTINGS, rollout_and_heads
from test_residual_heads import HEADS

CPU = torch.device("cpu")
OBS, ACTIONS = 4, 3
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
# Separate networks: without an entropy bonus the policy's gradient is the policy term's alone.
SEPARATE = ModelConfig(policy=(16,), value=(16,), shared=())


def learner_with(model: ModelConfig = HEADS, seed: int = 0, fp16: bool = False, **ppo: Any) -> Learner:
    torch.manual_seed(seed)
    settings = {"epochs": 1, "minibatches": 1, "graphed_steps": False, "advantage_keep": 0.5} | ppo
    learner = Learner(
        ActorCritic(OBS, ACTIONS, model), PPOConfig(**settings), gamma=0.9, device=CPU, fp16=fp16
    )
    learner.goal_gamma = 0.99
    return learner


def rollout_of(model: ActorCritic, seed: int, steps: int = 10, agents: int = 10) -> Rollout:
    """100 samples by default: advantage_keep 0.5 keeps 50, rest 0.2 draws 10 of the other 50 (weight 5)."""
    generator = torch.Generator().manual_seed(seed)
    r = Rollout.allocate(steps, agents, OBS, ACTIONS, CPU)
    r.obs.copy_(torch.randn(steps, agents, OBS, generator=generator))
    r.mask.copy_(torch.rand(steps, agents, ACTIONS, generator=generator) > 0.3)
    r.mask[..., 2] = True
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    taken = torch.multinomial(log_probs.exp().reshape(-1, ACTIONS), 1, generator=generator)
    r.actions.copy_(taken.view(steps, agents))
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    r.rewards.copy_(torch.randn(steps, agents, generator=generator))
    r.terminal[4, :] = True
    return r


def step_inputs(learner: Learner, seed: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """A step's flat observations, masks, actions and log-probabilities, random advantages and targets."""
    r = rollout_of(learner.model, seed)
    inputs = (
        r.obs.reshape(-1, OBS),
        r.mask.reshape(-1, ACTIONS),
        r.actions.reshape(-1),
        r.log_probs.reshape(-1),
    )
    advantages, targets = torch.randn(
        2, r.obs.shape[0] * r.obs.shape[1], generator=torch.Generator().manual_seed(9)
    )
    return inputs, advantages, targets


def prepared_step(learner: Learner, seed: int) -> tuple[torch.Tensor, Any]:
    """One step as `learn` runs it with the setting on: the update's rows, their weights and advantages."""
    inputs, advantages, targets = step_inputs(learner, seed)
    rows = learner._advantage_rows(advantages)
    b = rows.picked
    step_advantages, weights = rows.step(b)
    measured, _ = learner._step(
        *(t[b] for t in inputs), step_advantages, targets[b], False, samples=weights, prepared=True
    )
    return measured, rows


def gradients(learner: Learner, group: str | None = None) -> torch.Tensor:
    params = learner.groups[group] if group else list(learner.model.parameters())
    return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for p in params])


# (a) The default: the new code never entered.


@pytest.mark.parametrize("ppo", [{}, EVERY | {"optimizer": "muon"}], ids=["defaults", "every_signal"])
def test_at_1_the_new_code_is_never_entered(monkeypatch: pytest.MonkeyPatch, ppo: dict[str, Any]) -> None:
    """The default keeps today's code path (and so its bits; the exactness and resume tests of advantage_keep,
    tests/test_gco_signals.py and tests/test_multigpu.py, run it unchanged)."""
    assert PPOConfig().advantage_keep_rest == 1.0
    assert "advantage_rows" not in signal_columns(PPOConfig(advantage_keep=0.5))

    def entered(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the new code ran")

    monkeypatch.setattr(Learner, "_advantage_rows", entered)
    real_step = Learner._step

    def step(self: Learner, *args: Any, **kwargs: Any) -> Any:
        assert not kwargs.get("prepared") and "samples" not in kwargs
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(Learner, "_step", step)
    learner = learner_with(minibatches=2, **({"advantage_keep": 1.0} | ppo))
    r, heads = rollout_and_heads(learner, 0)
    stats = learner.learn(r, heads=heads)
    assert stats.losses_finite and "advantage_rows" not in stats.signals


# (b) The policy term: advantage_keep's filtered step.


def test_the_policy_term_is_todays_filtered_step_on_the_same_rows() -> None:
    """One step over every sample (minibatches 1): today's step standardises and filters its advantages, the
    new one runs on the kept rows and the drawn ones (advantage 0) with their weights. The policy loss and the
    policy's gradient (separate networks, no entropy bonus) are the same to 1e-6."""
    plain = learner_with(SEPARATE, entropy=0.0, max_grad_norm=1e9)
    inputs, advantages, targets = step_inputs(plain, 3)
    today, _ = plain._step(*inputs, advantages, targets, False)
    new = learner_with(SEPARATE, entropy=0.0, max_grad_norm=1e9, advantage_keep_rest=0.2)
    measured, rows = prepared_step(new, 3)
    filtered, _ = kept_advantages((advantages - advantages.mean()) / (advantages.std() + 1e-8), 0.5)
    assert torch.equal(rows.chosen, filtered != 0), "the same kept rows"
    scaled, _ = rows.step(torch.arange(100))  # one step over every sample: the kept ones x n / k
    torch.testing.assert_close(scaled, filtered, rtol=1e-6, atol=1e-6)
    assert float(measured[0]) == pytest.approx(float(today[0]), rel=1e-6, abs=1e-6)
    g_today, g_new = gradients(plain, "policy"), gradients(new, "policy")
    assert float((g_new - g_today).norm() / g_today.norm()) < 1e-6


def test_through_an_update_the_policy_moves_as_with_todays_filter() -> None:
    """`learn` with minibatches 1: the policy network (its own, no entropy bonus) ends where today's filter
    takes it, though the value network learned on fewer samples (at a policy rate of 1.5e-4, where the two
    agree to 1e-9)."""
    plain = learner_with(SEPARATE, entropy=0.0, policy_lr=1.5e-4)
    new = learner_with(SEPARATE, entropy=0.0, advantage_keep_rest=0.2, policy_lr=1.5e-4)
    rollout = rollout_of(plain.model, 4)
    plain.learn(rollout)
    stats = new.learn(rollout)
    assert stats.signals["advantage_rows"] == 0.6
    for p, q in zip(plain.groups["policy"], new.groups["policy"], strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=1e-9)
    assert not all(torch.equal(p, q) for p, q in zip(plain.groups["value"], new.groups["value"], strict=True))


# (c) The weights and the value side.


def test_the_weights_add_up_to_n_and_the_value_loss_is_their_weighted_mean() -> None:
    learner = learner_with(advantage_keep_rest=0.2)
    inputs, _, targets = step_inputs(learner, 5)
    with torch.no_grad():
        logits, predicted = learner.model(inputs[0])
    measured, rows = prepared_step(learner, 5)
    w = rows.weights
    assert rows.picked.numel() == 60 and rows.share == 0.6
    assert (int((w == 1).sum()), int((w == 5).sum()), int((w == 0).sum())) == (50, 10, 40)
    assert torch.equal(rows.chosen, w == 1) and torch.equal(rows.advantages != 0, rows.chosen)
    assert float(w.sum()) == 100.0, "1 x 50 + 5 x 10"
    assert torch.equal(w[rows.picked] > 0, torch.ones(60, dtype=torch.bool))
    expected = ((predicted - targets).pow(2) * w).sum() / w.sum()
    assert float(measured[1]) == pytest.approx(float(expected), rel=1e-6)
    entropy = normalized_entropy(masked_log_probs(logits, inputs[1]), inputs[1])
    assert float(measured[2]) == pytest.approx(float((entropy * w).sum() / w.sum()), rel=1e-6)
    odd = learner_with(advantage_keep_rest=0.3, advantage_keep=0.4)
    many = torch.randn(97, generator=torch.Generator().manual_seed(1))
    k = math.ceil(0.4 * 97)
    rows = odd._advantage_rows(many)
    r = round(0.3 * (97 - k))
    assert rows.picked.numel() == k + r and float(rows.weights.sum()) == pytest.approx(97, rel=1e-6)


def test_every_sample_through_the_new_path_is_the_full_step() -> None:
    """The test hook: a rest that draws every other sample (weight 1) runs the plain step on every sample, its
    measurements and every gradient the same to rounding."""
    plain = learner_with(max_grad_norm=1e9, minibatches=1)
    inputs, advantages, targets = step_inputs(plain, 6)
    today, _ = plain._step(*inputs, advantages, targets, False)
    every = learner_with(max_grad_norm=1e9, advantage_keep_rest=0.999)
    measured, rows = prepared_step(every, 6)
    assert rows.picked.numel() == 100 and torch.equal(rows.weights, torch.ones(100))
    torch.testing.assert_close(measured, today, rtol=1e-5, atol=1e-7)
    g_today, g_new = gradients(plain), gradients(every)
    assert float((g_new - g_today).norm() / g_today.norm()) < 1e-5


def test_each_step_holds_its_share_of_the_rows_in_a_new_order_each_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """420 samples: 210 kept, 42 of the other 210 drawn (weight 5), 252 rows; 5 minibatches of 50 (2 left
    over, as before). Every step mixes kept and drawn rows, each pass in its own order, each row at most once
    a pass; each step's policy term is the mean over its kept samples, whatever its share of drawn ones."""
    seen: list[tuple[_Rows, torch.Tensor]] = []
    real = _Rows.step

    def step(self: _Rows, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append((self, batch.clone()))
        return real(self, batch)

    monkeypatch.setattr(_Rows, "step", step)
    sizes: list[int] = []
    real_step = Learner._step

    def counted(self: Learner, obs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        sizes.append(obs.shape[0])
        return real_step(self, obs, *args, **kwargs)

    monkeypatch.setattr(Learner, "_step", counted)
    learner = learner_with(advantage_keep_rest=0.2, minibatches=5, epochs=2)
    stats = learner.learn(rollout_of(learner.model, 7, steps=20, agents=21))
    assert stats.signals["advantage_rows"] == 0.6 and sizes == [50] * 10
    rows = seen[0][0]
    assert all(r is rows for r, _ in seen), "the rows are chosen once an update"
    picked = set(rows.picked.tolist())
    kept_counts = []
    for number in range(2):
        batches = [b for _, b in seen[5 * number : 5 * number + 5]]
        every = torch.cat(batches).tolist()
        assert len(every) == len(set(every)) == 250 and set(every) <= picked
        for b in batches:
            kept = rows.chosen[b]
            kept_counts.append(int(kept.sum()))
            assert 0 < kept_counts[-1] < 50, "kept and drawn rows mixed"
            a, w = real(rows, b)
            assert float((a * w).sum() / w.sum()) == pytest.approx(
                float(rows.advantages[b][kept].mean()), rel=1e-5, abs=1e-6
            )
    assert len(set(kept_counts)) > 1, "steps of other mixes: the scale is each step's own"
    first, second = (torch.cat([b for _, b in seen[i : i + 5]]) for i in (0, 5))
    assert not torch.equal(first, second), "each pass in its own order"


def test_a_value_warm_up_runs_on_every_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen policy's warm-up has value-only passes: they keep every sample, as ppo.value_share's,
    and the column says so."""

    def entered(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the new code ran")

    monkeypatch.setattr(Learner, "_advantage_rows", entered)
    sizes: list[int] = []
    real_step = Learner._step

    def counted(self: Learner, obs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        assert not kwargs.get("prepared") and "samples" not in kwargs
        sizes.append(obs.shape[0])
        return real_step(self, obs, *args, **kwargs)

    monkeypatch.setattr(Learner, "_step", counted)
    learner = learner_with(advantage_keep_rest=0.2, minibatches=2)
    learner.policy_frozen = True
    stats = learner.learn(rollout_of(learner.model, 8))
    assert sizes == [50, 50] and stats.signals["advantage_rows"] == 1.0


# (d) The choice, the draw and resuming.


def test_the_kept_rows_are_the_top_k_with_ties_by_place() -> None:
    size = torch.tensor([1.0, 3.0, 2.0, 3.0, 3.0, 0.5])
    assert kept_rows(size, 2).tolist() == [False, True, False, True, False, False]
    assert kept_rows(size, 4).tolist() == [False, True, True, True, True, False]
    learner = learner_with(advantage_keep=0.25, advantage_keep_rest=0.5)
    # Mean 0, so |A| stays 2 four times after the standardisation; k = 3 of them, by place.
    tied = torch.tensor([0.0, 2.0, -2.0, 2.0, 0.5, -2.0, 1.0, -0.5, -1.0, 0.0])
    rows = learner._advantage_rows(tied)
    assert rows.chosen.tolist() == [False, True, True, True] + [False] * 6
    standardised = (tied - tied.mean()) / (tied.std() + 1e-8)
    assert torch.equal(rows.chosen, kept_rows(standardised.abs(), 3))


def test_the_drawn_rows_repeat_with_the_same_random_numbers() -> None:
    learner = learner_with(advantage_keep_rest=0.2)
    advantages = torch.randn(100, generator=torch.Generator().manual_seed(2))

    def drawn(seed: int) -> list[int]:
        torch.manual_seed(seed)
        rows = learner._advantage_rows(advantages)
        kept = set(rows.picked[:50].tolist())
        others = rows.picked[50:].tolist()
        assert kept == set(rows.chosen.nonzero().squeeze(1).tolist())
        assert len(set(others)) == 10 and not kept & set(others), "10 others, each once"
        return others

    assert drawn(1) == drawn(1) and drawn(1) != drawn(2)


@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_a_resumed_learner_learns_exactly_as_one_that_never_stopped(fp16: bool) -> None:
    signals = EVERY | {"optimizer": "muon", "advantage_keep_rest": 0.2, "minibatches": 2, "epochs": 2}
    straight = learner_with(fp16=fp16, **signals)
    r, heads = rollout_and_heads(straight, 0)
    straight.learn(r, heads=heads)
    saved = copy.deepcopy(straight.state_dict())

    def segment(learner: Learner) -> dict[str, float]:
        torch.manual_seed(50)
        r, heads = rollout_and_heads(learner, 1)
        return learner.learn(r, heads=heads).signals

    first = segment(straight)
    resumed = learner_with(seed=99, fp16=fp16, **signals)  # other starting weights: everything from the save
    resumed.load_state_dict(saved)
    assert segment(resumed) == first and set(first) == set(signal_columns(resumed.config))
    for a, b in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert straight.entropy_coef == resumed.entropy_coef


# (e) The settings refused.


ON = "[ppo]\ngraphed_steps = false\nadvantage_keep = 0.5\nadvantage_keep_rest = 0.2\n"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[ppo]\ngraphed_steps = false\nadvantage_keep_rest = 0.2\n", "needs advantage_keep between 0 and 1"),
        (ON.replace("rest = 0.2", "rest = 0"), "must be above 0"),
        (ON.replace("graphed_steps = false\n", ""), "set graphed_steps = false"),
        (ON + "[run]\ncards = 2\n", "several cards"),
        (ON + "normalize_advantages = false\n", "needs normalize_advantages"),
        (ON + "value_share = 0.5\n", "value_share below 1"),
    ],
)
def test_settings_that_cannot_work_yet_are_refused(tmp_path: Path, text: str, problem: str) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert re.search(f"advantage_keep_rest: .*{re.escape(problem)}", str(caught.value)), str(caught.value)


@pytest.mark.parametrize(
    ("used", "named"),
    [
        ("cards", "several cards"),
        ("graphed", "graphed steps"),
        ("value_share", "ppo.value_share below 1"),
    ],
)
def test_the_learner_refuses_what_the_settings_refuse(
    monkeypatch: pytest.MonkeyPatch, used: str, named: str
) -> None:
    """Each before anything else runs: a stand-in object is enough."""
    learner = learner_with(advantage_keep_rest=0.2, **({"value_share": 0.5} if used == "value_share" else {}))
    if used in ("cards", "graphed"):
        monkeypatch.setattr(learner, used, object())
    with pytest.raises(
        ValueError, match=f"advantage_keep_rest below 1 does not work with {re.escape(named)}"
    ):
        learner.learn(rollout_of(learner.model, 0))


def test_the_learner_refuses_too_few_rows() -> None:
    tiny = learner_with(advantage_keep_rest=0.001, advantage_keep=0.01, minibatches=8)  # 1 kept, 1 drawn
    with pytest.raises(ValueError, match="leaves 2 samples for 8 minibatches"):
        tiny.learn(rollout_of(tiny.model, 0))


# (f) The metrics column.


def test_the_column_appears_only_when_on() -> None:
    assert "advantage_rows" in CHARTS
    on = PPOConfig(advantage_keep=0.5, advantage_keep_rest=0.2, graphed_steps=False)
    assert signal_columns(on) == ["advantage_kept", "advantage_kept_size", "advantage_rows"]
    assert signal_columns(PPOConfig(advantage_keep=0.5)) == ["advantage_kept", "advantage_kept_size"]
    learner = learner_with(advantage_keep_rest=0.2, minibatches=2, epochs=2)
    stats = learner.learn(rollout_of(learner.model, 1))
    assert stats.signals["advantage_rows"] == 0.6 and stats.signals["advantage_kept"] == 0.5
    assert 0.5 < stats.signals["advantage_kept_size"] < 1


def _run(tmp_path: Path, name: str, state: dict[str, Any] | None = None) -> TrainingRun:
    text = SETTINGS.replace("advantage_keep = 0.5\n", "advantage_keep = 0.5\nadvantage_keep_rest = 0.2\n")
    file = tmp_path / "settings.toml"
    file.write_text(text, encoding="utf-8")
    config = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    folder = RunFolder(tmp_path / "runs" / name) if state else RunFolder.create(tmp_path / "runs", name, file)
    return TrainingRun(config, folder, state)


@pytest.fixture
def kept_determinism() -> Any:
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


@NEEDS_MESHES
@pytest.mark.usefixtures("kept_determinism")
def test_a_run_logs_the_column_and_resumes_exactly(tmp_path: Path) -> None:
    straight = _run(tmp_path, "straight")
    straight.iteration()
    straight.checkpoint()
    rows = [straight.iteration() for _ in range(2)]
    straight.metrics.close()
    assert all(0 < row["advantage_rows"] < 1 for row in rows)
    stopped = _run(tmp_path, "stopped")
    stopped.iteration()
    stopped.checkpoint(continue_after=False)
    stopped.metrics.close()
    resumed = _run(tmp_path, "stopped", RunFolder(tmp_path / "runs" / "stopped").load_latest())
    resumed_rows = [resumed.iteration() for _ in range(2)]
    resumed.metrics.close()
    for name, weights in straight.model.state_dict().items():
        assert torch.equal(weights, resumed.model.state_dict()[name]), f"{name} differs after resuming"
    assert [r["advantage_rows"] for r in rows] == [r["advantage_rows"] for r in resumed_rows]
    assert [r["entropy_coef"] for r in rows] == [r["entropy_coef"] for r in resumed_rows]
