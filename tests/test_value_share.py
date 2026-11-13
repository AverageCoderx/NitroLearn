"""`ppo.value_share`: a joint step's value side (the value network, or the critic trunk and every
value head) learns from the first share of the step's rows, or of each microbatch part; the policy from all.

On the processor: at 1 learning is the step's maths as in an earlier version (spelled out here, `old_losses`)
to the bit
(eager, parts, a frozen warm-up, Muon, the GCO heads, two cards on gloo), and the
share's own branch taking every row gives the same bits (the GCO trunks to rounding); a frozen warm-up (value
passes only) at 0.5 is 1's to the bit; at 0.5 with separate networks the policy learns exactly as at 1 and the
value network's gradient is the hand-computed loss on the first ceil(m / 2) rows; parts as big as the step
change nothing and smaller ones each take their own share; a padded (graphed) step, its share counted on the
card, is the unpadded step's maths; on two cards each card's value losses are its own share's and the update
repeats exactly. On a graphics card (skipped without one): graphed updates against eager ones at 0.5, to the
bit unpadded, to rounding padded."""

from __future__ import annotations

import copy
import faulthandler
import functools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig
from nitro.learner import HeadValues, Learner, LearnStats
from nitro.model import ActorCritic
from nitro.multigpu import Cards, join, leave
from nitro.ppo import masked_log_probs, normalized_entropy, policy_loss, weighted_mean
from test_graphed_gco import gco_learner, gco_rollout
from test_graphed_steps import (
    CUDA,
    NEEDS_GPU,
    batch,
    learner,
    pad,
    rollout,
)
from test_learner_microbatch import ACTIONS, OBS, rollout_for
from test_multigpu import GCO, GCO_HEADS, GCO_ROWS, GLOO, agents_of, gco_heads
from test_multigpu import rollout_for as card_rollout_for

CPU = torch.device("cpu")
EVERY_ROW = 0.9999  # ceil(0.9999 x m) = m for every step here: the share's own branch on every row
_STUCK_SECONDS = 240


def plain_learner(shared: tuple[int, ...], share: float | None = None, **ppo: Any) -> Learner:
    """`share` None: the setting left out."""
    torch.manual_seed(1)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(shared=shared, policy=(16,), value=(16,)))
    chosen = {} if share is None else {"value_share": share}
    config = PPOConfig(epochs=2, minibatches=2, **chosen, **ppo)
    return Learner(model, config, gamma=0.9, device=CPU, fp16=False)


def old_losses(learn: Learner, *args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """`Learner._losses` as it was in an earlier version (399a4820), spelled out: the reference the default
    must give
    to the bit. Only what these tests use (no tensor bonus)."""
    obs, mask, actions, old, advantages, targets, value_only, samples, heads, bonus = args
    assert bonus is None
    model = learn.model
    outputs: dict[str, torch.Tensor] = {}
    with torch.autocast(learn.device.type, dtype=torch.float16, enabled=learn.fp16):
        if heads is not None:
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
    value_loss = weighted_mean((predicted - targets).pow(2), samples)
    head_losses = learn._head_losses(outputs, heads, samples)
    zero = torch.zeros((), device=learn.device)
    if value_only:
        loss = value_loss + sum(head_losses, zero)
        measured = torch.stack([zero, value_loss.detach(), zero, zero, zero])
    else:
        log_probs = masked_log_probs(logits, mask)
        new = log_probs.gather(1, actions[:, None]).squeeze(1)
        config = learn.config
        entropy = weighted_mean(normalized_entropy(log_probs, mask, config.entropy_normalization), samples)
        policy = policy_loss(new, old, advantages, config.clip_range, samples)
        loss = policy.loss - learn.entropy_coef * entropy + value_loss
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
    if heads is not None:
        reported = [h.detach() / (2 if n == "vdag" else 1) for n, h in zip(heads, head_losses, strict=True)]
        measured = torch.cat([measured, torch.stack(reported)])
    return loss, measured


def with_old_maths(learn: Learner) -> Learner:
    learn._losses = functools.partial(old_losses, learn)  # type: ignore[method-assign]
    return learn


def plain_updates(learn: Learner, updates: int = 3) -> list[LearnStats]:
    stats = []
    for number in range(updates):
        torch.manual_seed(2 + number)  # the same actions and minibatches for every learner
        stats.append(learn.learn(rollout_for(learn.model)))
    return stats


def gco_updates(learn: Learner, updates: int = 3) -> list[LearnStats]:
    stats = []
    for number in range(updates):
        r, heads = gco_rollout(learn, number)
        torch.manual_seed(100 + number)
        stats.append(learn.learn(r, heads=heads))
    return stats


Run = Callable[[float | None, bool], tuple[Learner, list[LearnStats]]]


def _plain(shared: tuple[int, ...], frozen: bool = False, **ppo: Any) -> Run:
    def run(share: float | None, old: bool = False) -> tuple[Learner, list[LearnStats]]:
        learn = plain_learner(shared, share, **ppo)
        learn.policy_frozen = frozen
        return learn, plain_updates(with_old_maths(learn) if old else learn)

    return run


def _gco(batched: bool) -> Run:
    def run(share: float | None, old: bool = False) -> tuple[Learner, list[LearnStats]]:
        learn = gco_learner(CPU, False, batched=batched, **({} if share is None else {"value_share": share}))
        return learn, gco_updates(with_old_maths(learn) if old else learn)

    return run


# GCO: Muon, the four heads, the goal critic, the twins, advantage_keep, the entropy controller, a value pass.
CASES = {
    "separate networks": _plain(()),
    "shared trunk": _plain((16,)),
    "parts": _plain((16,), microbatch=5),
    "frozen warm-up": _plain((16,), frozen=True),
    "muon": _plain((16,), optimizer="muon"),
    "gco": _gco(False),
    "gco, muon batched": _gco(True),
}


def same(first: list[torch.Tensor], second: list[torch.Tensor]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


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


@pytest.mark.parametrize("case", list(CASES))
def test_at_1_learning_is_the_old_steps_maths_to_the_bit(case: str) -> None:
    """The setting left out (1) gives learn() with the step's maths as in an earlier version (`old_losses`),
    to the bit;
    so does an explicit 1."""
    absent, absent_stats = CASES[case](None, False)
    assert absent.config.value_share == 1.0
    old, old_stats = CASES[case](None, True)
    assert_same(absent, old, absent_stats, old_stats)
    one, one_stats = CASES[case](1.0, False)
    assert_same(absent, one, absent_stats, one_stats)
    assert any(s.value_loss > 0 for s in absent_stats)


def test_a_frozen_warm_up_at_half_is_1s_to_the_bit() -> None:
    """Every pass of a frozen policy's warm-up trains the value side alone, so every row: the share does
    nothing there (a critic check must use joint steps)."""
    one, one_stats = CASES["frozen warm-up"](None, False)
    half, half_stats = CASES["frozen warm-up"](0.5, False)
    assert_same(one, half, one_stats, half_stats)


@pytest.mark.parametrize("case", [c for c in CASES if c != "frozen warm-up"])
def test_the_shares_branch_on_every_row_is_the_old_steps_maths(case: str) -> None:
    """The share's own branch taking every row gives the old step's bits, so its maths differs only in the
    rows it takes; with the GCO bot's residual trunks to rounding (the policy's and the critic's gradients
    into the shared trunk may add up in another order). A frozen warm-up's passes never take it."""
    absent, absent_stats = CASES[case](None, False)
    every, every_stats = CASES[case](EVERY_ROW, False)
    if not case.startswith("gco"):
        assert_same(absent, every, absent_stats, every_stats)
        return
    for p, q in zip(absent.model.parameters(), every.model.parameters(), strict=True):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-6)
    for a, b in zip(absent_stats, every_stats, strict=True):
        assert b.value_loss == pytest.approx(a.value_loss, rel=1e-4)
        assert b.policy_loss == pytest.approx(a.policy_loss, rel=1e-4, abs=1e-6)


def step_inputs(rows: int, seed: int = 1) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(rows, OBS, generator=generator)
    mask = torch.rand(rows, ACTIONS, generator=generator) < 0.8
    mask[:, 0] = True
    actions = (torch.rand(rows, ACTIONS, generator=generator) * mask).argmax(1)
    old = -torch.rand(rows, generator=generator) - 1.0
    advantages = torch.randn(rows, generator=generator)
    targets = torch.randn(rows, generator=generator)
    return obs, mask, actions, old, advantages, targets


def test_at_half_the_policy_gradient_is_1s_and_the_value_gradient_the_first_rows_loss() -> None:
    """Separate networks, one step of 25 rows: the policy's gradients and measurements are share 1's to the
    bit; the value network's gradient and value loss are those of the plain mean over the first 13 rows."""
    full, half = plain_learner((), 1.0), plain_learner((), 0.5)
    inputs = step_inputs(25)
    measured = []
    for learn in (full, half):
        loss, found = learn._losses(*inputs, False)  # type: ignore[arg-type]
        loss.backward()
        measured.append(found.detach())
    for p, q in zip(full.model.policy.parameters(), half.model.policy.parameters(), strict=True):
        assert p.grad is not None and q.grad is not None
        assert torch.equal(p.grad, q.grad)
    policy_entries = torch.tensor([0, 2, 3, 4])
    assert torch.equal(measured[0][policy_entries], measured[1][policy_entries])
    value = list(half.model.value.parameters())
    grads = [p.grad.clone() for p in value if p.grad is not None]
    half.model.zero_grad()
    obs, targets = inputs[0], inputs[5]
    hand = (half.model.value_from(obs[:13]) - targets[:13]).pow(2).mean()
    hand.backward()
    assert len(grads) == len(value)
    for g, p in zip(grads, value, strict=True):
        assert torch.equal(g, p.grad)
    assert torch.equal(measured[1][1], hand.detach())
    assert not torch.equal(measured[0][1], measured[1][1])


def test_at_half_with_separate_networks_the_update_moves_the_policy_as_at_1() -> None:
    """Advantages are worked out before the update, so the policy's update is share 1's to the bit; only the
    value network's moves differently."""
    found = []
    for share in (1.0, 0.5):
        learn = plain_learner((), share)
        torch.manual_seed(2)
        r = rollout_for(learn.model)
        torch.manual_seed(3)
        found.append((learn, learn.learn(r)))
    (full, full_stats), (half, half_stats) = found
    for p, q in zip(full.model.policy.parameters(), half.model.policy.parameters(), strict=True):
        assert torch.equal(p, q)
    assert not same(list(full.model.value.parameters()), list(half.model.value.parameters()))
    for name in ("policy_loss", "entropy", "approx_kl", "clip_fraction", "policy_change"):
        assert getattr(full_stats, name) == getattr(half_stats, name), name
    assert full_stats.grad_norms["policy"] == half_stats.grad_norms["policy"]
    assert full_stats.explained_variance == half_stats.explained_variance  # measured before the update
    assert full_stats.value_loss != half_stats.value_loss


def test_at_half_parts_as_big_as_the_step_change_nothing() -> None:
    whole = plain_learner((16,), 0.5)
    parts = plain_learner((16,), 0.5, microbatch=24)  # 48 samples, 24 a step
    assert_same(whole, parts, plain_updates(whole), plain_updates(parts))


def test_at_half_each_part_takes_the_first_half_of_its_own_rows() -> None:
    """Parts of 5, 5, 5, 5 and 4 rows: the step's value loss and value gradient are those of each part's
    first 3, 3, 3, 3 and 2 rows' mean, weighted by the part's share of the step (to rounding: the shared
    trunk here runs on those rows alone)."""
    learn = plain_learner((16,), 0.5, microbatch=5)
    reference = copy.deepcopy(learn.model)
    inputs = step_inputs(24)
    parts = learn._step_parts(*inputs, False)  # type: ignore[arg-type]
    measured, _count, _sums = next(parts)
    grads = [p.grad.clone() for p in learn.model.value.parameters() if p.grad is not None]
    parts.close()
    obs, targets = inputs[0], inputs[5]
    hand = torch.zeros(())
    for begin in range(0, 24, 5):
        size = min(5, 24 - begin)
        k = math.ceil(size / 2)
        rows = slice(begin, begin + k)
        hand = hand + size / 24 * (reference.normalised_value(obs[rows]) - targets[rows]).pow(2).mean()
    hand.backward()
    torch.testing.assert_close(measured[1], hand.detach(), rtol=1e-6, atol=1e-7)
    value = list(reference.value.parameters())
    assert len(grads) == len(value)
    for g, p in zip(grads, value, strict=True):
        torch.testing.assert_close(g, p.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("share", [0.5, 0.3, 1 / 3, 0.37, 0.05, EVERY_ROW])
def test_a_padded_steps_share_is_counted_on_the_card_as_on_the_host(share: float) -> None:
    """The weights a padded (graphed) step gives its first k rows: 1 on the first ceil(share x n) of its n
    real samples (at least 1), 0 on the rest and on the padding, k fixed by the padded size."""
    learn = plain_learner((), share)
    for n in range(1, 200):
        for extra in (0, 7):
            samples = torch.cat([torch.ones(n), torch.zeros(extra)])
            k = max(1, math.ceil(share * (n + extra)))
            found = learn._value_rows(samples, k)
            assert found is not None and found.shape == (k,)
            taken = max(1, math.ceil(share * n))
            assert torch.equal(found, (torch.arange(k) < taken).float()), (n, extra)
    assert learn._value_rows(None, 5) is None


@pytest.mark.parametrize("case", ["plain", "parts", "gco", "gco parts"])
def test_a_padded_step_at_half_is_the_plain_steps_maths(case: str) -> None:
    """The padded-step check at 0.5: 40 rows padded with 13 weight-0 copies of the first (the graphed
    steps' buffers of 53) give the unpadded step's weights and measurements to FP32's rounding, with parts
    (16: the padded step's third part half padding, its fourth all) and with the GCO heads and signals."""
    gco = case.startswith("gco")
    micro = (7 if gco else 16) if case.endswith("parts") else 0
    extra = 13

    def make() -> Learner:
        if gco:
            return gco_learner(CPU, False, value_share=0.5, microbatch=micro)
        return learner(CPU, value_share=0.5, microbatch=micro)

    plain, padded = make(), make()
    obs, mask, actions, old, advantages, targets = batch(torch.float32)
    rows = obs.shape[0]
    generator = torch.Generator().manual_seed(4)
    heads = (
        {"vdag": torch.randn(rows, generator=generator), "goal": torch.randn(rows, generator=generator)}
        if gco
        else None
    )
    samples = torch.cat([torch.ones(rows), torch.zeros(extra)])
    bonus = torch.tensor(plain.entropy_coef)
    for value_only in (False, False, True, False):
        a, _ = plain._step(obs, mask, actions, old, advantages, targets, value_only, head_targets=heads)
        b, _ = padded._step(
            pad(obs, extra),
            pad(mask, extra),
            pad(actions, extra),
            pad(old, extra),
            pad(advantages, extra),
            pad(targets, extra),
            value_only,
            samples=samples,
            head_targets={name: pad(t, extra) for name, t in heads.items()} if heads is not None else None,
            entropy_coef=bonus if gco else None,
        )
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
    for p, q in zip(plain.model.parameters(), padded.model.parameters(), strict=True):
        torch.testing.assert_close(q, p, rtol=1e-5, atol=1e-6)


# --- Two cards (gloo) ------------------------------------------------------------------------------------


def card_learned(cards: Cards, share: float | None, old: bool = False) -> dict[str, Any]:
    """Two updates of a GCO learner with a shared trunk on this card's half of the batch (card 0 learns from
    3 of its 4 agents: uneven steps), each joint step's value and goal losses checked against the hand-made
    means over this card's first ceil(share x m) rows as it runs. `old`: the step's maths in an earlier
    version."""
    torch.manual_seed(1)
    model = ActorCritic(
        OBS,
        ACTIONS,
        ModelConfig(shared=(16,), policy=(16,), value=(16,), value_head_layers=(16,), value_heads=GCO_HEADS),
    )
    chosen = {} if share is None else {"value_share": share}
    config = PPOConfig(epochs=2, minibatches=1, graphed_steps=False, **GCO, **chosen)
    learn = Learner(model, config, gamma=0.9, device=cards.device, fp16=False)
    learn.goal_gamma = 0.99
    learn.cards = cards
    gaps: list[float] = []
    steps: list[int] = []
    inner = functools.partial(old_losses, learn) if old else learn._losses

    def checked(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        loss, measured = inner(*args)
        obs, targets, value_only, head_targets = args[0], args[5], args[6], args[8]
        if not value_only:
            k = max(1, math.ceil(learn.config.value_share * obs.shape[0]))
            with torch.no_grad():
                values = learn.model.head_values(obs[:k])
            main = (values["main"] - targets[:k]).pow(2).mean()
            goal = (head_targets["goal"][:k] - values["goal"]).pow(2).mean()
            at = 5 + list(head_targets).index("goal")
            gaps.append(float((measured[1] - main).abs() / main) + float((measured[at] - goal).abs() / goal))
            steps.append(k)
        return loss, measured

    learn._losses = checked  # type: ignore[method-assign]
    stats = []
    for number in range(2):
        torch.manual_seed(50 + number)
        r = card_rollout_for(model)
        heads = gco_heads(model, r)
        half = 8 // cards.size
        mine = slice(cards.rank * half, (cards.rank + 1) * half)
        r, rows = agents_of(r, mine), GCO_ROWS[mine]
        heads = HeadValues(
            {k: v[:, mine] for k, v in heads.now.items()}, {k: v[:, mine] for k, v in heads.reached.items()}
        )
        stats.append(repr(learn.learn(r, rows=rows, heads=heads)))
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "stats": stats,
        "gaps": gaps,
        "steps": steps,
    }


SHARES = {"absent": None, "old maths": None, "every row": EVERY_ROW, "half": 0.5, "half again": 0.5}


def _card(rank: int, size: int, store: str, out: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, CPU)
    found = {name: card_learned(cards, share, name == "old maths") for name, share in SHARES.items()}
    torch.save(found, f"{out}.{rank}")
    leave(clean=True)


@GLOO
def test_two_cards_at_1_as_before_and_at_half_each_on_its_own_share(tmp_path: Path) -> None:
    out = tmp_path / "card"
    mp.spawn(_card, args=(2, str(tmp_path / "store"), str(out)), nprocs=2, join=True)
    cards = [torch.load(f"{out}.{rank}") for rank in (0, 1)]
    for found in cards:
        for name in ("old maths", "every row"):  # to the bit, as the setting left out
            assert found[name]["stats"] == found["absent"]["stats"], name
            assert same(found[name]["params"], found["absent"]["params"]), name
        again = found["half again"]
        assert again["stats"] == found["half"]["stats"], "the update repeats exactly"
        assert same(again["params"], found["half"]["params"])
        assert all(p.isfinite().all() for p in found["half"]["params"])
        assert found["half"]["stats"] != found["absent"]["stats"]
        for name in SHARES:
            assert max(found[name]["gaps"]) < 1e-5, name
    for name in SHARES:  # every card holds the same networks
        assert same(cards[0][name]["params"], cards[1][name]["params"]), name
    # 24 and 32 rows a step (card 0 drives 3 of its 4 agents): half of each card's own.
    assert set(cards[0]["half"]["steps"]) == {12} and set(cards[1]["half"]["steps"]) == {16}
    assert set(cards[0]["absent"]["steps"]) == {24} and set(cards[1]["absent"]["steps"]) == {32}


# --- Settings --------------------------------------------------------------------------------------------


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path("."))


def test_the_setting_is_a_share_from_5_percent_to_all() -> None:
    assert PPOConfig().value_share == 1.0
    assert parse("[ppo]\nvalue_share = 0.05\n").ppo.value_share == 0.05
    assert parse("[ppo]\nvalue_share = 1.0\n").ppo.value_share == 1.0
    refused = {"0": "must be at least 0.05, but is 0", "1.5": "must be at most 1, but is 1.5"}
    for value, message in refused.items():
        with pytest.raises(ConfigError) as info:
            parse(f"[ppo]\nvalue_share = {value}\n")
        issues = {issue.path: issue.message for issue in info.value.issues}
        assert issues == {"ppo.value_share": message}


# --- On a graphics card ----------------------------------------------------------------------------------


@NEEDS_GPU
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_graphed_updates_at_half_give_the_eager_ones(fp16: bool) -> None:
    """The padded-step check at 0.5: the graphed updates are the eager ones to the bit."""
    eager, graphed = (
        learner(CUDA, fp16=fp16, microbatch=0, graphed_steps=on, value_share=0.5) for on in (False, True)
    )
    for seed in range(4):  # the first update runs eagerly (Adam has no state yet), then capture and replays
        stats = []
        for learn in (eager, graphed):
            torch.manual_seed(100 + seed)
            stats.append(learn.learn(rollout(learn.model, seed)))
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b)
        assert repr(stats[0]) == repr(stats[1])
    assert graphed.graphed is not None and graphed.graphed.graphs, "the steps ran graphed"


@NEEDS_GPU
@pytest.mark.parametrize("scale", ["raw", "popart"])
def test_graphed_gco_updates_at_half_give_the_eager_ones(scale: str) -> None:
    """The GCO heads and signals with Muon at 0.5, FP32, to the bit; also with ppo.head_loss_scale = 'popart',
    the heads' weighed losses on the value side's rows."""
    eager, graphed = (gco_learner(CUDA, on, value_share=0.5, head_loss_scale=scale) for on in (False, True))
    for number in range(4):
        stats = []
        for learn in (eager, graphed):
            r, heads = gco_rollout(learn, number)
            torch.manual_seed(200 + number)
            stats.append(learn.learn(r, heads=heads))
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b), number
        assert repr(stats[0]) == repr(stats[1]), number
    assert graphed.graphed is not None and graphed.graphed.graphs, "the steps ran graphed"
