"""Several cards (`run.cards`): two processes on the processor (gloo), each learning from half of a
batch, end with the networks and measurements one process gets from learning on the whole batch."""

from __future__ import annotations

import copy
import dataclasses
import faulthandler
import math
import os
import socket
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from nitro import multigpu
from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import HeadValues, Learner, head_values_on
from nitro.model import ActorCritic
from nitro.multigpu import Cards, GradientSums, join, leave
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout

CPU = torch.device("cpu")
TWO_CARDS = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two graphics cards")
# Where the cards run: processes on the processor (gloo), or two graphics cards (NCCL) where there are two.
DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=TWO_CARDS)]
OBS, ACTIONS, STEPS, AGENTS = 6, 4, 8, 8  # 64 samples; each card has 4 of the 8 agents

CASES: dict[str, dict[str, Any]] = {
    "separate networks": {"shared": (), "ppo": {}},
    "shared trunk, PopArt, parts": {"shared": (16,), "ppo": {"popart": True, "microbatch": 5}},
    "no reward clip": {"shared": (), "ppo": {"reward_clip": 0.0}},
    # The sum after the backward pass (the others sum while it runs).
    "summed after the backward pass, parts": {"shared": (16,), "ppo": {"microbatch": 5}, "overlap": False},
    # Gradients sent in bfloat16 (run.cards_sum_precision): close, not equal.
    "bfloat16 sums": {"shared": (16,), "ppo": {}, "bf16": True},
    # Learning steps replayed as CUDA graphs with the sums inside (run.cards_graphed_steps; on the processor
    # there are no graphs and these are the step-by-step cases again).
    "graphed steps": {"shared": (16,), "ppo": {}, "graphed": True},
    "graphed steps, summed after the backward pass": {
        "shared": (16,),
        "ppo": {},
        "graphed": True,
        "overlap": False,
    },
    # The one sum after the backward pass in bfloat16 too (it sent FP32 whatever the setting before).
    "graphed steps, bfloat16 sums after the backward pass": {
        "shared": (16,),
        "ppo": {},
        "graphed": True,
        "overlap": False,
        "bf16": True,
    },
}


def rollout_for(model: ActorCritic) -> Rollout:
    generator = torch.Generator().manual_seed(0)
    r = Rollout.allocate(STEPS, AGENTS, OBS, ACTIONS, CPU)
    r.obs.copy_(torch.randn(STEPS, AGENTS, OBS, generator=generator))
    r.mask.copy_(torch.rand(STEPS, AGENTS, ACTIONS, generator=generator) > 0.3)
    r.mask[..., 0] = True
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    # Its own random numbers: each card's learning draws a different number of them before the next batch.
    r.actions.copy_(
        torch.multinomial(log_probs.exp().reshape(-1, ACTIONS), 1, generator=generator).view(STEPS, AGENTS)
    )
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    r.rewards.copy_(torch.randn(STEPS, AGENTS, generator=generator) * 3)
    r.terminal[5, :3] = True
    r.truncated[2, 4:] = True
    # Each episode's steps counted (start_value): the cards' halves hold 7 and 8 fresh starts.
    r.episode_step.copy_(torch.arange(STEPS, dtype=torch.int32)[:, None].expand(STEPS, AGENTS))
    r.episode_step[6:, :3] -= 6
    r.episode_step[3:, 4:] -= 3
    r.truncated_values[2, 4:] = torch.randn(4, generator=generator)
    r.bootstrap_values.copy_(torch.randn(AGENTS, generator=generator))
    return r


def agents_of(rollout: Rollout, rows: slice) -> Rollout:
    """The rollout of some agents only: what one card collects."""
    fields = {f.name: getattr(rollout, f.name) for f in dataclasses.fields(Rollout)}
    return Rollout(
        **{
            name: None if t is None else t[rows] if name.startswith("bootstrap_") else t[:, rows]
            for name, t in fields.items()
        }
    )


def learned(
    case: str, cards: Cards | None, iterations: int = 2, device: torch.device = CPU
) -> dict[str, Any]:
    """Networks and measurements after `iterations` updates, on the whole batch or, with `cards`, on this
    card's half of it (on `cards.device`)."""
    spec = CASES[case]
    device = cards.device if cards is not None else device
    torch.manual_seed(1)
    model = ActorCritic(OBS, ACTIONS, ModelConfig(shared=spec["shared"], policy=(16,), value=(16,))).to(
        device
    )
    config = PPOConfig(epochs=2, minibatches=1, **spec["ppo"])
    learner = Learner(model, config, gamma=0.9, device=device, fp16=False)
    learner.cards = cards
    stats = []
    for _ in range(iterations):
        rollout = on(rollout_for(model.cpu()), device)
        model.to(device)
        if cards is not None:
            half = AGENTS // cards.size
            rollout = agents_of(rollout, slice(cards.rank * half, (cards.rank + 1) * half))
        result = learner.learn(rollout)
        stats.append(
            {
                name: getattr(result, name)
                for name in (
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                    "policy_lr",
                    "reward_scale",
                    "start_value",
                )
            }
        )
    graphs = len(learner.graphed.graphs) if learner.graphed is not None else 0
    return {
        "params": [p.detach().cpu().clone() for p in model.parameters()],
        "stats": stats,
        "graphs": graphs,
    }


def on(rollout: Rollout, device: torch.device) -> Rollout:
    fields = {f.name: getattr(rollout, f.name) for f in dataclasses.fields(Rollout)}
    return Rollout(**{name: t.to(device) if t is not None else None for name, t in fields.items()})


# A card process stuck for this long prints where every thread is and ends, so a hang fails the test with
# its stacks instead of blocking the run.
_STUCK_SECONDS = 240


def _card(rank: int, size: int, store: str, case: str, out: str, device: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    spec = CASES[case]
    dtype = torch.bfloat16 if spec.get("bf16") else torch.float32
    cards = join(
        Path(store),
        rank,
        size,
        torch.device(device),
        spec.get("overlap", True),
        dtype,
        spec.get("graphed", False),
    )
    torch.save(learned(case, cards), f"{out}.{rank}")
    leave(clean=True)


def _loopback_available() -> bool:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


# Several cards train one bot on Linux only (`multigpu.check` refuses elsewhere): the cards' processes meet
# over its loopback interface.
GLOO = pytest.mark.skipif(
    sys.platform != "linux" or not _loopback_available(),
    reason="several cards train on Linux only, over its loopback interface",
)


@GLOO
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_two_cards_learn_as_one_on_the_whole_batch(case: str, device: str, tmp_path: Path) -> None:
    one = learned(case, None, device=torch.device(device))
    out = tmp_path / "card"
    mp.spawn(_card, args=(2, str(tmp_path / "store"), case, str(out), device), nprocs=2, join=True)
    first, second = (torch.load(f"{out}.{rank}") for rank in (0, 1))
    for a, b in zip(first["params"], second["params"], strict=True):
        assert torch.equal(a, b), "every card holds the same networks"
    if device == "cuda":  # graphed on several cards only with run.cards_graphed_steps
        expected = bool(CASES[case].get("graphed"))
        assert (first["graphs"] > 0, second["graphs"] > 0) == (expected, expected)
    moved = any(
        not torch.equal(a, b) for a, b in zip(one["params"], learned(case, None, 0)["params"], strict=True)
    )
    assert moved, "the update changed the networks"
    # A graphics card adds sums up in another order when the batch is split: a little more rounding.
    rtol, atol = (1e-5, 1e-6) if device == "cpu" else (1e-4, 1e-5)
    if CASES[case].get("bf16"):  # each gradient rounded to about 3 significant digits on its way
        rtol, atol = 1e-3, 1e-4
    for a, b in zip(one["params"], first["params"], strict=True):
        torch.testing.assert_close(b, a, rtol=rtol, atol=atol)
    for whole, split in zip(one["stats"], first["stats"], strict=True):
        for name, value in whole.items():
            assert split[name] == pytest.approx(value, rel=rtol, abs=atol / 10), name


# The GCO learning signals on two cards. A fast entropy controller, so its bonus moves visibly;
# the twins restart before the third update; Muon, as the GCO starter. Keeping 30%: k of n tells the cards' n
# apart.
GCO = {
    "goal_critic_weight": 0.25,
    "vdag_beta": 0.3,
    "advantage_keep": 0.3,
    "entropy_target": 0.5,
    "entropy_target_rate": 5.0,
    "vdag_reinit_steps": 100.0,
    "optimizer": "muon",
}
GCO_HEADS = ("main", "goal", "vdag1", "vdag2")
# Card 0 learns from 3 of its 4 agents (the 4th a self-play opponent), card 1 from all 4: uneven shares.
GCO_ROWS = torch.tensor([True, True, False, True, True, True, True, True])


def gco_heads(model: ActorCritic, rollout: Rollout) -> HeadValues:
    """The extra heads on the whole rollout (each agent's own rows; a card takes its agents' columns)."""
    generator = torch.Generator().manual_seed(5)
    last = torch.randn(AGENTS, OBS, generator=generator)
    steps, agents = rollout.truncated.nonzero(as_tuple=True)
    cut_obs = torch.randn(steps.numel(), OBS, generator=generator)
    return head_values_on(model, GCO_HEADS[1:], False, rollout, last, (steps, agents, cut_obs))


def gco_learned(
    cards: Cards | None,
    iterations: int,
    state: dict[str, Any] | None = None,
    first: int = 0,
    device: torch.device = CPU,
) -> dict[str, Any]:
    """Updates `first` to `first + iterations - 1` with every GCO signal on, on the whole batch or this
    card's half of it (on `cards.device`), from `state` (card 0's learner state) when given: the networks,
    measurements and learner state after them."""
    device = cards.device if cards is not None else device
    torch.manual_seed(1 if state is None else 99)  # a resumed learner takes everything from `state`
    model = ActorCritic(
        OBS,
        ACTIONS,
        ModelConfig(shared=(16,), policy=(16,), value=(16,), value_head_layers=(16,), value_heads=GCO_HEADS),
    ).to(device)
    config = PPOConfig(epochs=2, minibatches=1, graphed_steps=False, **GCO)
    learner = Learner(model, config, gamma=0.9, device=device, fp16=False)
    learner.goal_gamma = 0.99
    learner.cards = cards
    if state is not None:
        learner.load_state_dict(copy.deepcopy(state))
    stats, restarted = [], []
    for number in range(first, first + iterations):
        rank = cards.rank if cards is not None else 0
        # Card 1 draws other numbers than card 0, its twins' fresh weights too: card 0's must reach it.
        torch.manual_seed(50 + number + 1000 * rank)
        restarted.append(learner.restart_twins(64.0 * number))
        rollout = rollout_for(model.cpu())
        heads, rows = gco_heads(model, rollout), GCO_ROWS
        model.to(device)
        rollout = on(rollout, device)
        heads = HeadValues(
            {k: v.to(device) for k, v in heads.now.items()},
            {k: v.to(device) for k, v in heads.reached.items()},
        )
        if cards is not None:
            half = AGENTS // cards.size
            mine = slice(cards.rank * half, (cards.rank + 1) * half)
            rollout, rows = agents_of(rollout, mine), rows[mine]
            heads = HeadValues(
                {k: v[:, mine] for k, v in heads.now.items()},
                {k: v[:, mine] for k, v in heads.reached.items()},
            )
        result = learner.learn(rollout, rows=rows.to(device), heads=heads)  # the rows on the card
        stats.append({"entropy": result.entropy, "policy_loss": result.policy_loss, **result.signals})
    return {
        "params": [p.detach().cpu().clone() for p in model.parameters()],
        "stats": stats,
        "restarted": restarted,
        "entropy_coef": learner.entropy_coef,
        "state": copy.deepcopy(learner.state_dict()),
    }


def _gco_card(rank: int, size: int, store: str, out: str, device: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, torch.device(device))
    straight = gco_learned(cards, 3)
    stopped = gco_learned(cards, 2)
    # Every card resumes from card 0's state, as a run's helpers start from card 0's start file.
    saved = cards.send(stopped["state"] if rank == 0 else None)
    assert isinstance(saved, dict)
    resumed = gco_learned(cards, 1, saved, first=2)
    torch.save({"straight": straight, "resumed": resumed}, f"{out}.{rank}")
    leave(clean=True)


@GLOO
@pytest.mark.parametrize("device", DEVICES)
def test_two_cards_with_every_gco_signal_learn_as_one_and_resume_exactly(tmp_path: Path, device: str) -> None:
    """The goal critic's and the twins' statistics, the filter's top share and the entropy controller's
    entropy are over both cards' samples, so two cards end where one process on the whole batch does
    (within rounding: the cards' statistics are FP64 sums where one process takes FP32 reductions, and the
    gradient sum is split); the twins restart on both at once with card 0's fresh weights; and two cards
    resumed from card 0's saved learner state carry on exactly as two that never stopped."""
    one = gco_learned(None, 3, device=torch.device(device))
    out = tmp_path / "card"
    mp.spawn(_gco_card, args=(2, str(tmp_path / "store"), str(out), device), nprocs=2, join=True)
    first, second = (torch.load(f"{out}.{rank}", weights_only=False) for rank in (0, 1))
    assert one["restarted"] == first["straight"]["restarted"] == [False, False, True]
    assert first["resumed"]["restarted"] == [True], "the twins restart after the save too"
    for run in ("straight", "resumed"):
        for a, b in zip(first[run]["params"], second[run]["params"], strict=True):
            assert torch.equal(a, b), "every card holds the same networks, the twins' fresh start too"
        assert first[run]["entropy_coef"] == second[run]["entropy_coef"], "the bonus alike on every card"
        assert first[run]["stats"] == second[run]["stats"], "every measurement alike on every card"
    # A graphics card adds sums up in another order when the batch is split: a little more rounding.
    rtol, atol = (1e-5, 1e-6) if device == "cpu" else (1e-4, 1e-5)
    for a, b in zip(one["params"], first["straight"]["params"], strict=True):
        torch.testing.assert_close(b, a, rtol=rtol, atol=atol)
    assert one["entropy_coef"] != PPOConfig().entropy, "the controller moved the bonus"
    assert first["straight"]["entropy_coef"] == pytest.approx(one["entropy_coef"], rel=rtol)
    for whole, split in zip(one["stats"], first["straight"]["stats"], strict=True):
        assert set(whole) == set(split)
        for name, value in whole.items():
            assert split[name] == pytest.approx(value, rel=10 * rtol, abs=atol), name
    # Both cards' 56 learning samples (24 + 32) a step: ceil(0.3 x 56) = 17 kept (each card alone: 8 and 10).
    assert first["straight"]["stats"][0]["advantage_kept"] == pytest.approx(17 / 56)
    for a, b in zip(first["straight"]["params"], first["resumed"]["params"], strict=True):
        assert torch.equal(a, b), "resumed on two cards exactly as never stopped"
    assert first["straight"]["entropy_coef"] == first["resumed"]["entropy_coef"]


def _pieces_card(rank: int, size: int, store: str, out: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, CPU)
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.Linear(32, 4)
    )
    data = torch.randn(2, 10, 6)  # the same on both cards; each takes its row
    sums = GradientSums(cards, list(model.parameters()), piece_bytes=64)
    assert len(sums.pieces) > 3, "several pieces, sent in order"
    sums.start()
    (model(data[rank]).pow(2).mean() / size).backward()
    assert sums.next == len(sums.pieces), "every piece started during the backward pass"
    sums.finish()
    torch.save([p.grad.clone() for p in model.parameters()], f"{out}.{rank}")
    leave(clean=True)


@GLOO
def test_gradients_summed_in_pieces_during_the_backward_pass_are_the_mean_gradient(tmp_path: Path) -> None:
    out = tmp_path / "grads"
    mp.spawn(_pieces_card, args=(2, str(tmp_path / "store"), str(out)), nprocs=2, join=True)
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.Linear(32, 4)
    )
    data = torch.randn(2, 10, 6)
    model(data.reshape(20, 6)).pow(2).mean().backward()
    for rank in (0, 1):
        for whole, summed in zip(model.parameters(), torch.load(f"{out}.{rank}"), strict=True):
            torch.testing.assert_close(summed, whole.grad, rtol=1e-5, atol=1e-7)


def _old_average_(
    grads: list[torch.Tensor], measured: torch.Tensor, samples: int | torch.Tensor
) -> torch.Tensor:
    """`Cards.average_` as it was in an earlier version: always one FP32 sum."""
    flat = torch.cat(
        [
            *(g.reshape(-1).float() for g in grads),
            measured.detach().float().reshape(-1),
            measured.new_ones(1, dtype=torch.float32),
        ]
    )
    flat.mul_(samples if isinstance(samples, torch.Tensor) else float(samples))
    torch.distributed.all_reduce(flat)
    flat.div_(flat[-1].clone())
    offset = 0
    for g in grads:
        g.copy_(flat[offset : offset + g.numel()].view_as(g))
        offset += g.numel()
    return flat[offset : offset + measured.numel()].view_as(measured).to(measured.dtype)


def card_gradients(rank: int) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Card `rank`'s gradients (two groups of different sizes), measured values and count of samples: a count
    bfloat16 cannot hold (its spacing there is 64)."""
    generator = torch.Generator().manual_seed(rank)
    early = [torch.randn(300, 7, generator=generator) * 1e-3, torch.randn(41, generator=generator)]
    late = [torch.randn(64, 5, generator=generator) * 30, torch.randn(9, generator=generator)]
    measured = torch.randn(5, generator=generator) / 3
    return early, late, measured, torch.tensor(12799.0 + rank)


def _average_card(rank: int, size: int, store: str, out: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, CPU)
    sent: list[tuple[torch.dtype, int]] = []
    reduce = torch.distributed.all_reduce

    def recorded(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        sent.append((tensor.dtype, tensor.numel()))
        return reduce(tensor, *args, **kwargs)

    torch.distributed.all_reduce = recorded  # in this process only
    found: dict[str, Any] = {}

    def run(name: str, average: Callable[..., torch.Tensor]) -> None:
        early, late, measured, samples = card_gradients(rank)
        sent.clear()
        result = average(early, late, measured, samples)
        found[name] = {"grads": early + late, "measured": result, "sent": list(sent)}

    run("old", lambda e, la, m, s: _old_average_(e + la, m, s))
    for name, mine in (("fp32", cards), ("bf16", dataclasses.replace(cards, sum_dtype=torch.bfloat16))):
        run(name, lambda e, la, m, s, mine=mine: mine.average_(e + la, m, s))
    torch.save(found, f"{out}.{rank}")
    leave(clean=True)


@GLOO
def test_the_cards_average_sends_the_precision_asked_and_in_two_sums_the_same_numbers(tmp_path: Path) -> None:
    """FP32 (the default): one sum of everything, the numbers to the bit as before. bfloat16: the
    gradients travel in it (on this path they did not before), the measured values and the count in a small
    FP32 sum (exactly as in FP32); each average is the cards' bfloat16 sum of their weighted gradients over
    the FP32 count."""
    out = tmp_path / "averages"
    mp.spawn(_average_card, args=(2, str(tmp_path / "store"), str(out)), nprocs=2, join=True)
    first, second = (torch.load(f"{out}.{rank}") for rank in (0, 1))
    early, late, measured, _ = card_gradients(0)
    n, m = sum(g.numel() for part in (early, late) for g in part), measured.numel()
    for found in (first, second):
        assert found["old"]["sent"] == found["fp32"]["sent"] == [(torch.float32, n + m + 1)]
        assert found["bf16"]["sent"] == [(torch.bfloat16, n), (torch.float32, m + 1)]
        assert all(map(torch.equal, found["old"]["grads"], found["fp32"]["grads"]))
        assert torch.equal(found["old"]["measured"], found["fp32"]["measured"])
        assert torch.equal(found["bf16"]["measured"], found["fp32"]["measured"]), "measured exactly"
    for name in first:
        assert all(map(torch.equal, first[name]["grads"], second[name]["grads"])), "every card alike"
        assert torch.equal(first[name]["measured"], second[name]["measured"])
    cards = [card_gradients(rank) for rank in (0, 1)]
    count = cards[0][3] + cards[1][3]
    for k, average in enumerate(first["bf16"]["grads"]):
        weighted = [(c[0] + c[1])[k] * c[3] for c in cards]
        assert torch.equal(average, (weighted[0].bfloat16() + weighted[1].bfloat16()).float() / count)
        # Each card's share and their sum rounded to bfloat16's 8 bits: off by at most 2^-7 of the shares.
        bound = 2**-7 * (weighted[0].abs() + weighted[1].abs()) / count
        assert ((average - first["fp32"]["grads"][k]).abs() <= bound).all()


def _drift_card(rank: int, size: int, store: str, out: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, CPU)
    torch.manual_seed(3)
    weights = [torch.randn(5000), torch.randn(7, 3)]
    found = [cards.same_weights(weights)]
    for change in ("last bit", "swapped", "other tensor"):
        changed = [w.clone() for w in weights]
        if rank == 1 and change == "last bit":
            changed[0][4321] = torch.nextafter(changed[0][4321], torch.tensor(float("inf")))
        elif rank == 1 and change == "swapped":  # the same values, two of them in each other's place
            changed[0][[10, 4500]] = changed[0][[4500, 10]]
        elif rank == 1:
            changed[1][6, 2] += 1.0
        found.append(cards.same_weights(changed))
    torch.save(found, f"{out}.{rank}")
    leave(clean=True)


@GLOO
def test_the_cards_notice_weights_that_drifted_apart(tmp_path: Path) -> None:
    """One card's weight a bit off, two of its weights swapped, or a weight off in another tensor: every
    card sees it (the check sums runs of weights)."""
    out = tmp_path / "found"
    mp.spawn(_drift_card, args=(2, str(tmp_path / "store"), str(out)), nprocs=2, join=True)
    for rank in (0, 1):
        assert torch.load(f"{out}.{rank}") == [True, False, False, False]


class _Summed:
    """Stands in for one card's `Cards`: keeps what each sum was given (one card's sum changes nothing)."""

    def __init__(self) -> None:
        self.given: list[torch.Tensor] = []

    def sum_(self, tensor: torch.Tensor) -> torch.Tensor:
        self.given.append(tensor.clone())
        return tensor


ONE_CARD = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graphics card")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=ONE_CARD)])
def test_the_count_made_on_the_card_is_the_count_copied_from_the_host(device: str) -> None:
    """The running moments and the value normaliser make their sample count on the card (torch.full)
    instead of copying it from the host (new_tensor, which waits for the card): the cards sum the same FP64
    numbers, to the bit."""
    from nitro.ppo import PopArt

    torch.manual_seed(5)
    for count in (1, 7, 4099):
        values = torch.randn(count, device=device) * 3 + 1
        wide = values.double()
        summed = _Summed()
        Cards.moments(summed, values)  # type: ignore[arg-type]
        assert torch.equal(summed.given[0], torch.stack([wide.new_tensor(wide.numel()), wide.sum()]))
        summed = _Summed()
        PopArt(torch.nn.Linear(4, 1).to(device)).update(values, summed)  # type: ignore[arg-type]
        before = torch.stack([wide.new_tensor(wide.numel()), wide.sum(), (wide**2).sum()])
        assert torch.equal(summed.given[0], before)


MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
RUN_SETTINGS = """
[run]
seed = 7
cards = 2
device = "cpu"
[env]
num_arenas = 8
threads = 4
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 320
minibatches = 2
[selfplay]
enabled = true
past_share = 0.5
active_opponents = 2
snapshot_steps = 1_000_000
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""
# A plain run.
PLAIN = ""
# 1v1 and 2v2 arenas side by side on each card: written into [env].
MIXED = "team_size = 2\nteam_sizes = [1, 2]\nobs_team_size = 2\n"


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@GLOO
@pytest.mark.parametrize(
    "where",
    [
        pytest.param('device = "cpu"', id="cpu"),
        pytest.param('device = "cuda"\nprecision = "fp16"', id="cuda-fp16", marks=TWO_CARDS),
        pytest.param(
            'device = "cuda"\nprecision = "fp16"\ncards_graphed_steps = true',
            id="cuda-fp16-graphed",
            marks=TWO_CARDS,
        ),
    ],
)
@pytest.mark.parametrize("extra", [PLAIN, MIXED], ids=["plain", "mixed sizes"])
def test_a_run_on_two_cards_trains_and_saves_in_step(tmp_path: Path, extra: str, where: str) -> None:
    """Card 0 and a helper process: self-play opponents drawn at a save, a plain run or mixed sizes, each
    iteration counting both cards' steps; every card's networks equal after each update
    (`Leader.after_iteration` stops the run otherwise) and the helper ends cleanly."""
    from nitro.config.loader import parse_toml
    from nitro.config.schema import NitroConfig
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    file = tmp_path / "settings.toml"
    text = RUN_SETTINGS.replace('device = "cpu"', where)
    text = text.replace("[env]\n", "[env]\n" + MIXED) if extra == MIXED else text + extra
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(
        parsed,
        paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES),
        selfplay=dataclasses.replace(parsed.selfplay, snapshot_steps=320),  # below the settings' minimum
    )
    folder = RunFolder.create(tmp_path, "cards", file)
    run = TrainingRun(settings, folder)
    assert run.config.env.num_arenas == 4 and run.config.ppo.steps_per_iteration == 160, "card 0's share"
    if extra == MIXED:
        assert run.envs.arena_team_sizes == [1, 1, 2, 2], "each card plays both sizes"
    try:
        run.iteration()
        # 8 rows a card take 20 steps for 160; the mixed card's 12 rows take 14 (168).
        per_iteration = 336 if extra == MIXED else 320
        assert run.agent_steps == per_iteration, "both cards' steps"
        run.iteration()
        run.iteration()
        run.checkpoint()  # new games and opponents on both cards
        row = run.iteration()
        assert run.agent_steps == 4 * per_iteration
        if extra == MIXED:
            assert row["1v1_car_step_share"] != "" and row["2v2_car_step_share"] != ""
        assert run.cards is not None
        helper = run.cards.helpers[0]
    finally:
        if run.cards is not None:
            run.cards.stop()
        run.metrics.close()
    assert helper.exitcode == 0, (folder.root / "cards" / "card1.log").read_text(encoding="utf-8")
    log = (folder.root / "cards" / "card1.log").read_text(encoding="utf-8")
    assert "Card 1 of 2: 4 games" in log


GCO_MODEL = 'value_head_layers = [16]\nvalue_heads = ["main", "goal", "vdag1", "vdag2"]\n'
GCO_PPO = (
    'graphed_steps = false\noptimizer = "muon"\ngoal_critic_weight = 0.25\nvdag_beta = 0.3\n'
    "vdag_reinit_steps = 640\nadvantage_keep = 0.5\nentropy_target = 0.5\nentropy_target_min = 1.0\n"
)


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@GLOO
def test_a_run_on_two_cards_with_every_gco_signal_trains_in_step(tmp_path: Path) -> None:
    """The GCO starter's learning signals in a real run on two processes: the twins restart at 640 steps on
    both cards together (card 0's fresh weights: the leader checks the networks equal after every iteration
    and stops the run otherwise), every signal's column is filled, and the helper ends cleanly."""
    from nitro.config.loader import parse_toml
    from nitro.config.schema import NitroConfig
    from nitro.learner import signal_columns
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    text = RUN_SETTINGS.replace("[model]\n", "[model]\n" + GCO_MODEL).replace("[ppo]\n", "[ppo]\n" + GCO_PPO)
    file = tmp_path / "settings.toml"
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, "cards-gco", file)
    run = TrainingRun(settings, folder)
    try:
        rows = [run.iteration() for _ in range(3)]  # the third starts at 640 steps: the twins restart
        assert run.trainer.learner.twins_started == 640
        run.checkpoint()
        rows.append(run.iteration())
        for column in signal_columns(run.config.ppo):
            assert all(math.isfinite(row[column]) for row in rows), column
        assert all(row["entropy_coef"] >= run.config.ppo.entropy for row in rows), "never below the floor"
        assert run.cards is not None
        helper = run.cards.helpers[0]
    finally:
        if run.cards is not None:
            run.cards.stop()
        run.metrics.close()
    log = (folder.root / "cards" / "card1.log").read_text(encoding="utf-8")
    assert helper.exitcode == 0, log
    assert "start again from fresh weights at 640 steps" in log


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@GLOO
def test_a_helper_that_stops_stops_the_run_with_its_log_named(tmp_path: Path) -> None:
    from nitro.config.loader import parse_toml
    from nitro.config.schema import NitroConfig
    from nitro.health import TrainingProblem
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    file = tmp_path / "settings.toml"
    file.write_text(RUN_SETTINGS, encoding="utf-8")
    parsed = parse_toml(RUN_SETTINGS, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    run = TrainingRun(settings, RunFolder.create(tmp_path, "cards", file))
    assert run.cards is not None
    try:
        run.iteration()
        run.cards.helpers[0].kill()
        run.cards.helpers[0].join()
        with pytest.raises(TrainingProblem, match=r"card1\.log"):
            run.iteration()
    finally:
        run.cards.stop()
        run.metrics.close()


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@GLOO
@pytest.mark.parametrize(
    "where",
    [
        pytest.param('device = "cpu"', id="cpu"),
        pytest.param('device = "cuda"\nprecision = "fp16"', id="cuda-fp16", marks=TWO_CARDS),
        pytest.param(
            'device = "cuda"\nprecision = "fp16"\ncards_graphed_steps = true',
            id="cuda-fp16-graphed",
            marks=TWO_CARDS,
        ),
    ],
)
@pytest.mark.parametrize("caught", ["at the third iteration", "at a save"])
def test_networks_checked_every_third_iteration_catch_a_drift_in_time_and_before_a_save(
    tmp_path: Path, caught: str, where: str
) -> None:
    """run.cards_check_every = 3: the cards check their networks every third iteration and before every
    save, a save with unchecked iterations behind it going on as usual. A drift of card 0's networks is caught
    within 3 iterations, or by the save that comes first, on both cards; that save is not written. Should the
    cards ever disagree on when to check, the stuck timer ends the test instead of NCCL's 10 minutes."""
    from nitro.config.loader import parse_toml
    from nitro.config.schema import NitroConfig
    from nitro.health import TrainingProblem
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    text = RUN_SETTINGS.replace('device = "cpu"', where)
    text = text.replace("cards = 2\n", "cards = 2\ncards_check_every = 3\n")
    file = tmp_path / "settings.toml"
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, "cards", file)
    run = TrainingRun(settings, folder)
    assert run.cards is not None and run.config.run.cards_check_every == 3
    checks = 0
    same_weights = Cards.same_weights

    def counted(cards: Cards, parameters: Any) -> bool:
        nonlocal checks
        checks += 1
        return same_weights(cards, parameters)

    faulthandler.dump_traceback_later(2 * _STUCK_SECONDS, exit=True)  # cancelled whatever happens below
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Cards, "same_weights", counted)
            run.iteration()
            run.checkpoint()  # one unchecked iteration: checked first, then saved; the helper carries on
            assert checks == 1 and len(folder.complete_checkpoints()) == 1
            run.iteration()
            run.iteration()
            run.iteration()  # the third since the save: checked
            assert checks == 2
            run.iteration()
            with torch.no_grad():  # card 0's networks drift a little
                next(run.model.parameters()).view(-1)[0] += 1e-3
            with pytest.raises(TrainingProblem, match="drifted apart"):
                if caught == "at a save":
                    run.checkpoint()
                else:
                    run.iteration()  # the second since the last check: not checked yet
                    assert checks == 2
                    run.iteration()
        assert len(folder.complete_checkpoints()) == 1, "nothing drifted is saved"
        helper = run.cards.helpers[0]
        helper.join(timeout=60)  # it found the drift too
    finally:
        try:  # a stop that hangs still meets the timer; one that fails still cancels it
            run.cards.stop()
            run.metrics.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
    log = (folder.root / "cards" / "card1.log").read_text(encoding="utf-8")
    assert helper.exitcode == 1 and "this card's networks drifted from card 0's" in log, log


def test_settings_several_cards_cannot_train_with_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arenas and steps that do not split evenly between the cards are refused, saying what to change;
    several cards anywhere but on Linux too, one card everywhere."""
    from nitro.config import default_config
    from nitro.multigpu import CardsError, check

    base = default_config(Path(__file__).parents[1])
    two = dataclasses.replace(base, run=dataclasses.replace(base.run, cards=2))
    monkeypatch.setattr(sys, "platform", "linux")
    check(two, CPU)
    odd = dataclasses.replace(two, env=dataclasses.replace(two.env, num_arenas=two.env.num_arenas + 1))
    with pytest.raises(CardsError, match=r"must divide by run\.cards"):
        check(odd, CPU)
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(sys, "platform", platform)
        for device in (CPU, torch.device("cuda")):
            with pytest.raises(CardsError, match=r"on Linux only; here use run\.cards = 1"):
                check(two, device)
        check(base, CPU)


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@GLOO
@pytest.mark.parametrize(
    "where",
    [
        pytest.param('device = "cpu"', id="cpu"),
        pytest.param('device = "cuda"\nprecision = "fp16"', id="cuda-fp16", marks=TWO_CARDS),
    ],
)
def test_a_run_on_two_cards_copies_replays_in_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """[replay_mix] on two cards: every card takes the same copy steps, each on its own replays'
    decisions (the copy step's decisions shared out), and applies the cards' average gradient, so the networks
    stay the same on both (the leader checks it after every iteration)."""
    from nitro import replays
    from nitro.config.loader import parse_toml
    from nitro.config.schema import NitroConfig
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun
    from test_pretrain_large import lean_file
    from test_replays import replicar_file

    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    replicar_file(tmp_path / "one.parquet")
    lean_file(tmp_path / "one.lean", *replays.read_replicar(tmp_path / "one.parquet"))
    data = (tmp_path / "one.lean").read_bytes()
    for name in ("r0", "r1", "r2"):
        assert replays.keep(data, name, 1)
    # Named, so the helper's process (which has none of this test's changes) finds them too.
    store = (replays.store_folder() / "1v1").as_posix()
    mix = f"[replay_mix]\nenabled = true\nshare = 0.5\nlr = 1e-3\nminibatch = 64\nreplays = ['{store}']\n"
    file = tmp_path / "settings.toml"
    text = RUN_SETTINGS.replace('device = "cpu"', where) + mix
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, "cards-mix", file)
    run = TrainingRun(settings, folder)
    try:
        rows = [run.iteration() for _ in range(3)]
        assert run.replay_mix is not None and run.replay_mix.minibatch == 32, "each card's share of 64"
        assert sum(int(row["replay_mix_steps"] or 0) for row in rows) > 0, "copy steps were taken"
        assert run.cards is not None
        helper = run.cards.helpers[0]
    finally:
        if run.cards is not None:
            run.cards.stop()
        run.metrics.close()
    assert helper.exitcode == 0, (folder.root / "cards" / "card1.log").read_text(encoding="utf-8")


def test_each_card_gets_whole_cache_groups_of_the_processor(tmp_path: Path) -> None:
    """run.cards_pin_threads: an EPYC-like machine of 8 cache groups (4 cores and their SMT siblings
    each) splits into two halves of whole groups; Linux's cache lists are read from its files; with fewer
    groups than cards, plain even runs."""
    for ccx in range(8):
        cores = [*range(4 * ccx, 4 * ccx + 4), *range(32 + 4 * ccx, 32 + 4 * ccx + 4)]
        text = f"{4 * ccx}-{4 * ccx + 3},{32 + 4 * ccx}-{32 + 4 * ccx + 3}"
        for cpu in cores:
            (tmp_path / f"cpu{cpu}" / "cache" / "index3").mkdir(parents=True)
            (tmp_path / f"cpu{cpu}" / "cache" / "index3" / "shared_cpu_list").write_text(text + "\n")
    groups = multigpu.cache_groups(range(64), tmp_path)
    assert len(groups) == 8 and groups[1] == [4, 5, 6, 7, 36, 37, 38, 39]
    first, second = (multigpu.card_cpus(rank, 2, range(64), groups) for rank in (0, 1))
    assert sorted(first + second) == list(range(64)) and len(first) == len(second) == 32
    assert set(first) == {c for g in groups[:4] for c in g}
    assert multigpu.cache_groups([0, 1], tmp_path / "missing") == [[0], [1]]
    assert multigpu.card_cpus(1, 2, [0, 1, 2, 3], [[0, 1, 2, 3]]) == [2, 3]


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_pinning_moves_every_thread_of_the_process() -> None:
    before = sorted(os.sched_getaffinity(0))
    started = threading.Event()
    stop = threading.Event()
    worker = threading.Thread(target=lambda: (started.set(), stop.wait()))
    worker.start()
    started.wait()
    try:
        multigpu.pin_threads(before[:1])
        assert sorted(os.sched_getaffinity(worker.native_id)) == before[:1]
    finally:
        multigpu.pin_threads(before)
        stop.set()
        worker.join()
    assert sorted(os.sched_getaffinity(0)) == before
