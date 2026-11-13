"""`ppo.graphed_steps` with the GCO bot's settings: Muon (per matrix and batched) and
every GCO learning signal (the goal critic, the twin critics with a restart, advantage_keep and the
entropy controller), on one card and on two.

On the processor (no graphs here): a graphed step's padding is left out of advantage_keep's choice exactly as
the step without it chooses, k and n worked out on the card as `kept_count` does on the host; a padded step
with every signal and Muon is the plain step's maths (to FP32's rounding, as padded steps); the entropy
bonus as a tensor gives the float's step to the bit; what the graphed steps put back after a capture's eager
run holds the extra heads, Muon's state and the filter's sums; the settings are no longer refused.

On graphics cards (skipped without one): graphed updates against eager ones on a small network shaped like
the GCO starter's, FP32 and FP16, Muon per matrix and batched: on one card to the bit, a value warm-up's
value-only steps too; on two cards (NCCL, the sum after the backward pass, which is the graphed steps' way)
to the bit, every card the same."""

from __future__ import annotations

import faulthandler
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from nitro.config import load_config
from nitro.config.loader import parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig
from nitro.graphed_steps import GraphedSteps
from nitro.learner import HeadValues, Learner, head_values_on
from nitro.model import ActorCritic
from nitro.multigpu import join, leave
from nitro.ppo import kept_advantages, kept_count, kept_counts
from nitro.rollout import Rollout
from test_graphed_steps import (
    ACTIONS,
    CUDA,
    NEEDS_GPU,
    OBS,
    pad,
    rollout,
)
from test_multigpu import GLOO, TWO_CARDS, agents_of

ROOT = Path(__file__).resolve().parents[1]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
CPU = torch.device("cpu")
_STUCK_SECONDS = 300  # a card test stuck this long prints every thread's stack and ends (a hang fails)
# The GCO starter's shape a few dozen numbers wide: residual trunks, a critic trunk, four heads with their own
# layers.
GCO_MODEL = ModelConfig(
    shared=(24, 24),
    policy=(32, 32),
    value=(24, 24),
    value_head_layers=(24, 24),
    value_heads=("main", "goal", "vdag1", "vdag2"),
    residual=True,
    activation="leaky_relu",
)
# The GCO starter's signals; an entropy controller whose bonus moves every update (and stays within its
# range); the twins restart at the fourth of six updates (64 agent steps an update here), so the graphs the
# second and the fifth capture replay in the third and the sixth, with another bonus (`LateReplays`).
GCO_PPO: dict[str, Any] = {
    "epochs": 2,
    "minibatches": 2,
    "policy_lr": 0.003,
    "value_lr": 0.003,
    "entropy": 0.035,
    "optimizer": "muon",
    "goal_critic_weight": 0.25,
    "vdag_beta": 0.3,
    "vdag_reinit_steps": 150.0,
    "advantage_keep": 0.5,
    "entropy_target": 0.5,
    "entropy_target_rate": 0.5,
}


@pytest.fixture
def watchdog() -> Iterator[None]:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def gco_learner(
    device: torch.device, graphed: bool, fp16: bool = False, batched: bool = False, **ppo: Any
) -> Learner:
    torch.manual_seed(3)
    model = ActorCritic(OBS, ACTIONS, GCO_MODEL)
    config = PPOConfig(**(GCO_PPO | {"graphed_steps": graphed, "muon_batched": batched} | ppo))
    learn = Learner(model.to(device), config, gamma=0.9, device=device, fp16=fp16)
    learn.goal_gamma = 0.99
    return learn


def gco_rollout(learn: Learner, seed: int, agents: int = 4) -> tuple[Rollout, HeadValues]:
    """`test_graphed_steps.rollout` (8 steps) with agents 0 and 1 cut at step 2, valued as the trainer values
    a rollout: the cut states' and the last states' values, and the extra heads on every state."""
    model = learn.model
    device = next(model.parameters()).device
    r = rollout(model, seed, agents=agents)
    generator = torch.Generator().manual_seed(500 + seed)
    last = torch.randn(agents, OBS, generator=generator).to(device)
    cut_obs = torch.randn(2, OBS, generator=generator).to(device)
    cut_steps, cut_agents = torch.tensor([2, 2], device=device), torch.tensor([0, 1], device=device)
    r.terminal[2, :2] = False
    r.truncated[2, :2] = True
    with torch.no_grad():
        r.truncated_values[2, :2] = model.normalised_value(cut_obs)
        r.bootstrap_values.copy_(model.normalised_value(last))
    heads = head_values_on(model, learn.extra_heads, learn.fp16, r, last, (cut_steps, cut_agents, cut_obs))
    return r, heads


# --- On the processor ------------------------------------------------------------------------------------


def test_k_and_n_on_the_card_are_kept_counts() -> None:
    """The FP64 numbers on the card are the host's double arithmetic: ceil(keep x n) to the same integer,
    and n / k (which scales the kept advantages) to the same FP32."""
    for keep in (0.5, 0.3, 0.1, 0.7, 1 / 3, 0.37, 0.999, 0.01):
        for n in range(1, 300):
            samples = torch.cat([torch.ones(n), torch.zeros(7)])
            count, k = kept_counts(samples, keep)
            assert (int(count), int(k)) == (n, kept_count(n, keep)), (keep, n)
            assert float((count / k).float()) == float(torch.tensor(n / kept_count(n, keep)))


@pytest.mark.parametrize("keep", [0.5, 0.3, 0.37])
def test_the_filter_leaves_a_graphed_steps_padding_out_as_the_unpadded_step(keep: float) -> None:
    """Padded with copies of the first sample (the graphed steps' padding, weight 0): the real samples keep
    exactly the advantages the step without the padding keeps (the same choice, the same n / k), the padding
    none; the kept share of |A| to rounding (a longer sum)."""
    generator = torch.Generator().manual_seed(7)
    for n in (2, 9, 40, 333):
        advantages = torch.randn(n, generator=generator)
        plain, share = kept_advantages(advantages, keep)
        extra = 13
        samples = torch.cat([torch.ones(n), torch.zeros(extra)])
        padded, padded_share = kept_advantages(pad(advantages, extra), keep, samples=samples)
        assert torch.equal(padded[:n], plain), n
        assert torch.equal(padded[n:], torch.zeros(extra))
        torch.testing.assert_close(padded_share, share, rtol=1e-6, atol=0)
        assert int((plain != 0).sum()) == kept_count(n, keep)
        # Counted as samples, the padding changes the choice: n and k would be the padded ones.
        counted, _ = kept_advantages(pad(advantages, extra), keep)
        assert not torch.equal(counted[:n], plain)


def test_ties_at_the_kth_largest_are_kept_by_place_and_exactly_k_are_kept() -> None:
    advantages = torch.tensor([0.5, -1.0, 1.0, 0.2, 1.0, -1.0, 0.1])
    samples = torch.cat([torch.ones(7), torch.zeros(3)])
    kept, _ = kept_advantages(pad(advantages, 3), 0.3, samples=samples)  # k = 3 of 7: four tie at 1.0
    expected = torch.tensor([0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0]) * (7 / 3)  # the first three by place
    assert torch.equal(kept, torch.cat([expected, torch.zeros(3)]))


@pytest.mark.parametrize("batched", [False, True], ids=["muon", "muon batched"])
@pytest.mark.parametrize("microbatch", [0, 7], ids=["whole", "parts"])
def test_a_padded_step_with_every_gco_signal_is_the_plain_steps_maths(batched: bool, microbatch: int) -> None:
    """The padded-step check with the GCO settings: the heads' losses, advantage_keep and the entropy bonus
    (a tensor, as the graphed steps pass it) on a step padded with weight-0 copies give the plain step's
    weights, measurements, Muon's momentum and the filter's sums, to FP32's rounding."""
    plain, padded = (gco_learner(CPU, False, batched=batched, microbatch=microbatch) for _ in range(2))
    rows, extra = 40, 13
    generator = torch.Generator().manual_seed(1)
    obs = torch.randn(rows, OBS, generator=generator)
    mask = torch.rand(rows, ACTIONS, generator=generator) < 0.8
    mask[:, 0] = True
    actions = (torch.rand(rows, ACTIONS, generator=generator) * mask).argmax(1)
    old = -torch.rand(rows, generator=generator) - 2.0
    advantages, targets = torch.randn(rows, generator=generator), torch.randn(rows, generator=generator)
    heads = {"vdag": torch.randn(rows, generator=generator), "goal": torch.randn(rows, generator=generator)}
    samples = torch.cat([torch.ones(rows), torch.zeros(extra)])
    bonus = torch.tensor(plain.entropy_coef)
    for value_only in (False, False, True):
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
            head_targets={name: pad(t, extra) for name, t in heads.items()},
            entropy_coef=bonus,
        )
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
    for p, q in zip(plain.model.parameters(), padded.model.parameters(), strict=True):
        torch.testing.assert_close(q, p, rtol=1e-5, atol=1e-6)
        if "momentum_buffer" in plain.optimizer.state[p]:
            torch.testing.assert_close(
                padded.optimizer.state[q]["momentum_buffer"], plain.optimizer.state[p]["momentum_buffer"]
            )
    torch.testing.assert_close(padded._kept, plain._kept, rtol=1e-6, atol=0)
    assert float(plain._kept[0]) == 2 * kept_count(rows, 0.5) / rows


def test_the_entropy_bonus_as_a_tensor_gives_the_floats_step_to_the_bit() -> None:
    first, second = (gco_learner(CPU, False) for _ in range(2))
    for learn in (first, second):
        learn.entropy_coef = 0.0123456789  # not an FP32 number: the product rounds it the same both ways
    batch = rollout(first.model, 0)
    obs, mask = batch.obs.reshape(32, -1), batch.mask.reshape(32, -1)
    actions, old = batch.actions.reshape(32), batch.log_probs.reshape(32)
    advantages, targets = torch.randn(32), torch.randn(32)
    a, _ = first._step(obs, mask, actions, old, advantages, targets, False)
    bonus = torch.tensor(first.entropy_coef)
    b, _ = second._step(obs, mask, actions, old, advantages, targets, False, entropy_coef=bonus)
    assert torch.equal(a, b)
    for p, q in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(p, q)


def test_what_a_capture_puts_back_holds_the_heads_muons_state_and_the_filters_sums() -> None:
    """`GraphedSteps._state` (what the eager run before a capture changes, put back after it): every weight
    of the extra heads, Muon's momentum and step counts, Adam's state, and advantage_keep's sums."""
    learn = gco_learner(CPU, False)
    r, heads = gco_rollout(learn, 0)
    learn.learn(r, heads=heads)
    steps = GraphedSteps.__new__(GraphedSteps)  # its buffers need a card; `_state` reads the learner only
    steps.learner = learn
    held = {t.data_ptr() for t in steps._state()}
    for name in ("goal", "vdag1", "vdag2"):
        assert all(p.data_ptr() in held for p in learn.model.value_heads[name].parameters()), name
    muon = {id(p) for p in learn.model.muon_parameters()}
    assert muon and all(
        id(p) in muon for p in learn.optimizer.state if "momentum_buffer" in learn.optimizer.state[p]
    )
    for p, state in learn.optimizer.state.items():
        assert ("momentum_buffer" in state) == (id(p) in muon)
        assert all(v.data_ptr() in held for v in state.values() if isinstance(v, torch.Tensor))
    assert learn._kept.data_ptr() in held


def test_graphed_steps_are_allowed_with_muon_and_every_gco_signal(tmp_path: Path) -> None:
    """The two refusals are lifted: the GCO starter with graphed_steps = true, on 1 card and 2."""
    text = GCO_STARTER.read_text(encoding="utf-8").replace("graphed_steps = false", "graphed_steps = true")
    for cards in (1, 2):
        config = parse_toml(
            text.replace("[run]\n", f"[run]\ncards = {cards}\n", 1),
            NitroConfig,
            file=GCO_STARTER,
            base_dir=GCO_STARTER.parent,
        )
        assert config.run.cards == cards
        assert config.ppo.graphed_steps and config.ppo.optimizer == "muon"
        assert config.ppo.signals_on() == [
            "goal_critic_weight",
            "vdag_beta",
            "advantage_keep",
            "entropy_target",
        ]
    path = tmp_path / "settings.toml"
    path.write_text(
        '[ppo]\noptimizer = "muon"\nmuon_batched = true\nentropy_target = 0.5\n', encoding="utf-8"
    )
    assert load_config(path).ppo.graphed_steps, "graphed_steps is on by default"


# --- On graphics cards -----------------------------------------------------------------------------------


UPDATES = 6
RESTARTS = [False, False, False, True, False, False]


class LateReplays:
    """Counts each update's captures and records the joint steps' replays of graphs captured in an earlier
    update, with the entropy bonus each update used. A number fixed in a graph at its capture and moved since
    (the controller's bonus) shows only in such a replay: within an update the bonus stays."""

    def __init__(self, steps: GraphedSteps) -> None:
        self.update = 0
        self.bonus: dict[int, float] = {}
        self.captured: dict[int, int] = {}
        self.late: list[tuple[int, int]] = []  # (the update that captured, a later one that replayed)
        capture = steps._capture

        def counted(key: tuple[Any, ...], *args: Any) -> tuple[Callable[[], None], Any]:
            replay, results = capture(key, *args)
            born = self.update
            self.captured[born] = self.captured.get(born, 0) + 1

            def replayed() -> None:
                if self.update != born and not key[0]:  # key[0]: value_only, which has no bonus
                    self.late.append((born, self.update))
                replay()

            steps.graphs[key] = (replayed, results)
            return replayed, results

        steps._capture = counted  # type: ignore[method-assign, assignment]

    def begin(self, number: int, bonus: float) -> None:
        self.update = number
        self.bonus[number] = bonus

    def found(self) -> tuple[dict[int, float], dict[int, int], list[tuple[int, int]]]:
        return self.bonus, self.captured, self.late


def check_late(found: tuple[dict[int, float], dict[int, int], list[tuple[int, int]]]) -> None:
    """`LateReplays.found()`: some joint step replayed, in a later update, a graph captured with another
    bonus; and some graphed update captured nothing."""
    bonus, captured, late = found
    assert any(bonus[born] != bonus[now] for born, now in late), late
    assert any(captured.get(n, 0) == 0 for n in range(2, UPDATES)), captured


def _same_optimizer_state(a: Learner, b: Learner) -> None:
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        # `get`, not `[]`: indexing gives a weight without state yet (a held policy) an empty one, and a
        # capture then keeps that weight's first state instead of putting it back to a fresh start
        mine, theirs = a.optimizer.state.get(p, {}), b.optimizer.state.get(q, {})
        assert set(mine) == set(theirs)
        for key, value in mine.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, theirs[key]), key


@NEEDS_GPU
@pytest.mark.usefixtures("watchdog")
@pytest.mark.parametrize("scale", ["raw", "popart"])
@pytest.mark.parametrize("batched", [False, True], ids=["muon", "muon batched"])
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_graphed_gco_steps_give_the_eager_updates_to_the_bit(fp16: bool, batched: bool, scale: str) -> None:
    """No padding: every graphed update is the eager one to the bit (weights, Muon's and Adam's state, every
    measurement, the entropy bonus), through a twin restart (its state dropped: the graphs are recaptured) and
    replays, in the third and sixth updates, of graphs captured in an earlier one with another bonus. With
    ppo.head_loss_scale = 'popart' such replays read PopArt's statistics as the update moved them."""
    eager, graphed = (gco_learner(CUDA, on, fp16, batched, head_loss_scale=scale) for on in (False, True))
    assert graphed.graphed is not None
    late = LateReplays(graphed.graphed)
    restarts = []
    for number in range(UPDATES):  # the first update runs eagerly (no optimizer state yet), then graphs
        stats, restarted = [], []
        late.begin(number, graphed.entropy_coef)
        for learn in (eager, graphed):
            torch.manual_seed(100 + number)  # the same fresh twins
            restarted.append(learn.restart_twins(64.0 * number))
            r, heads = gco_rollout(learn, number)
            torch.manual_seed(200 + number)  # the same minibatches (drawn on the card)
            stats.append(learn.learn(r, heads=heads))
        assert restarted[0] == restarted[1]
        restarts.append(restarted[0])
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b), number
        assert repr(stats[0]) == repr(stats[1]), number
        assert eager.entropy_coef == graphed.entropy_coef
        _same_optimizer_state(eager, graphed)
    assert restarts == RESTARTS
    assert eager.entropy_coef != GCO_PPO["entropy"], "the controller moved the bonus"
    assert graphed.graphed.graphs, "the steps ran graphed"
    check_late(late.found())
    keys = {(key[0], key[1]) for key in graphed.graphed.graphs}
    assert keys == {(False, False)}, "joint steps, none padded"


@NEEDS_GPU
@pytest.mark.usefixtures("watchdog")
@pytest.mark.parametrize("batched", [False, True], ids=["muon", "muon batched"])
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_a_graphed_value_warm_up_with_every_gco_signal_gives_the_eager_updates_to_the_bit(
    fp16: bool, batched: bool
) -> None:
    """Value-only steps come from a value warm-up (pretrain.value_warmup_steps: the policy and the shared
    trunk held): graphed, its updates are the eager ones to the bit, every head learning, and so are the
    joint updates after it, whose first capture gives the policy and the trunk their first optimizer state
    (put back to a fresh start, as the eager step makes it)."""
    eager, graphed = (gco_learner(CUDA, on, fp16, batched) for on in (False, True))
    assert graphed.graphed is not None
    captured: list[tuple[bool, bool, bool]] = []
    capture = graphed.graphed._capture

    def counted(key: tuple[Any, ...], *args: Any) -> tuple[Callable[[], None], Any]:
        captured.append((key[0], key[1], key[2]))  # value_only, padded, policy_frozen
        return capture(key, *args)

    graphed.graphed._capture = counted  # type: ignore[method-assign, assignment]
    for number in range(4):  # a warm-up of two updates (the first eager: no optimizer state yet), then two
        stats = []
        for learn in (eager, graphed):
            learn.policy_frozen = number < 2
            torch.manual_seed(100 + number)
            learn.restart_twins(64.0 * number)
            r, heads = gco_rollout(learn, number)
            torch.manual_seed(200 + number)
            stats.append(learn.learn(r, heads=heads))
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b), number
        assert repr(stats[0]) == repr(stats[1]), number
        assert stats[1].value_only_steps == (4 if number < 2 else 0), number
        assert eager.entropy_coef == graphed.entropy_coef
        _same_optimizer_state(eager, graphed)
    assert (True, False, True) in captured and (False, False, False) in captured, captured
    assert all(not padded for _, padded, _ in captured), "none padded"


# Card 0 learns from 3 of its 4 agents (the 4th a self-play opponent), card 1 from all 4: uneven step sizes.
ROWS = torch.tensor([True, True, False, True, True, True, True, True])


def two_card_updates(rank: int, size: int, store: str, out: str, fp16: bool, batched: bool) -> None:
    """One card's process: an eager and a graphed learner (run.cards_graphed_steps), each on this card's half
    of the same rollouts, the cards' sum after the backward pass (as graphed steps always sum)."""
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, torch.device("cuda"), overlap=False, graphed=True)
    found: dict[str, Any] = {}
    for graphed in (False, True):
        learn = gco_learner(cards.device, graphed, fp16, batched)
        learn.cards = cards
        late = LateReplays(learn.graphed) if learn.graphed is not None else None
        stats, restarts = [], []
        for number in range(UPDATES):
            if late is not None:
                late.begin(number, learn.entropy_coef)
            torch.manual_seed(
                100 + number + 1000 * rank
            )  # each card draws its own: card 0's twins reach both
            restarts.append(learn.restart_twins(64.0 * number))
            r, heads = gco_rollout(learn, number, agents=8)
            half = 8 // size
            mine = slice(rank * half, (rank + 1) * half)
            r = agents_of(r, mine)
            heads = HeadValues(
                {k: v[:, mine] for k, v in heads.now.items()},
                {k: v[:, mine] for k, v in heads.reached.items()},
            )
            torch.manual_seed(200 + number)
            stats.append(learn.learn(r, rows=ROWS[mine].to(cards.device), heads=heads))
        found[str(graphed)] = {
            "params": [p.detach().cpu().clone() for p in learn.model.parameters()],
            "state": {
                i: {
                    k: v.cpu().clone()
                    for k, v in learn.optimizer.state[p].items()
                    if isinstance(v, torch.Tensor)
                }
                for i, p in enumerate(learn.model.parameters())
            },
            "stats": [repr(one) for one in stats],
            "signals": [one.signals for one in stats],  # every card's samples: alike on every card
            "entropy_coef": learn.entropy_coef,
            "graphs": len(learn.graphed.graphs) if learn.graphed is not None else 0,
            "restarts": restarts,
            "replays": late.found() if late is not None else None,  # checked by the test, not on a card
        }
    torch.save(found, f"{out}.{rank}")
    leave(clean=True)


@TWO_CARDS
@GLOO
@pytest.mark.parametrize("batched", [False, True], ids=["muon", "muon batched"])
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_graphed_gco_steps_on_two_cards_give_the_eager_updates_to_the_bit(
    tmp_path: Path, fp16: bool, batched: bool
) -> None:
    """Two cards (NCCL): the standardisation and advantage_keep's all-gather of every card's |A| outside the
    graphs before each step's first half, the cards' sum between the halves, Muon in the second; against the
    eager steps on the same two cards: the same bits on each card, and every card the same."""
    out = tmp_path / "card"
    mp.spawn(  # type: ignore[attr-defined, no-untyped-call]
        two_card_updates, args=(2, str(tmp_path / "store"), str(out), fp16, batched), nprocs=2, join=True
    )
    cards = [torch.load(f"{out}.{rank}", weights_only=False) for rank in (0, 1)]
    for found in cards:
        eager, graphed = found["False"], found["True"]
        assert graphed["graphs"] > 0 and eager["graphs"] == 0
        assert eager["restarts"] == graphed["restarts"] == RESTARTS
        check_late(graphed["replays"])
        for a, b in zip(eager["params"], graphed["params"], strict=True):
            assert torch.equal(a, b)
        for i, state in eager["state"].items():
            assert set(state) == set(graphed["state"][i])
            assert all(torch.equal(v, graphed["state"][i][k]) for k, v in state.items())
        assert eager["stats"] == graphed["stats"]
        assert eager["entropy_coef"] == graphed["entropy_coef"] != GCO_PPO["entropy"]
    for a, b in zip(cards[0]["True"]["params"], cards[1]["True"]["params"], strict=True):
        assert torch.equal(a, b), "every card holds the same networks"
    assert cards[0]["True"]["signals"] == cards[1]["True"]["signals"], "the signals alike on every card"
    assert cards[0]["True"]["entropy_coef"] == cards[1]["True"]["entropy_coef"]
