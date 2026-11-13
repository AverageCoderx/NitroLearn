"""ppo.muon_lr_scale, ppo.weight_decay and the weight-size columns (weight_norm_, turn_, growth_).

On the processor: a Muon step at scale 0.2 moves each Muon matrix 0.2x as far from the same state and every
Adam-held parameter alike, batched as one at a time; weight decay shrinks each LayerNorm-fed pair by
(1 - rate x lambda)^steps and nothing else, the outputs unchanged to 1e-5 (residual on and off), a frozen
policy not at all; an exact resume with both on; two cards (gloo) hold the same weights; the columns against
a hand computation, a fresh orthogonal network's growth 1; the settings refusals; a real run's metrics.csv.
On a graphics card (skipped without one): graphed updates with both on give the eager ones to the bit."""

from __future__ import annotations

import copy
import csv
import dataclasses
import faulthandler
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig
from nitro.learner import Learner, LearnStats
from nitro.model import ActorCritic
from nitro.multigpu import join, leave
from nitro.muon import MuonAdam
from test_graphed_gco import gco_learner, gco_rollout
from test_multigpu import GLOO, agents_of
from test_optimizer_options import ACTIONS, OBS, rollout_for

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
CUDA = torch.device("cuda")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
MUON_MODEL = ModelConfig(
    shared=(8, 8), policy=(16, 16), value=(12, 12), residual=True, activation="leaky_relu"
)


def plain_learner(fp16: bool = False, model: ModelConfig = MUON_MODEL, **ppo: Any) -> Learner:
    torch.manual_seed(5)
    config = PPOConfig(**({"epochs": 2, "minibatches": 2, "graphed_steps": False} | ppo))
    return Learner(ActorCritic(OBS, ACTIONS, model), config, gamma=0.9, device=CPU, fp16=fp16)


def plain_updates(learn: Learner, updates: int = 3, first: int = 0) -> list[LearnStats]:
    out = []
    for number in range(first, first + updates):
        torch.manual_seed(20 + number)
        out.append(learn.learn(rollout_for(learn.model, number)))
    return out


def gco_updates(learn: Learner, updates: int = 3, first: int = 0) -> list[LearnStats]:
    out = []
    for number in range(first, first + updates):
        r, heads = gco_rollout(learn, number)
        torch.manual_seed(200 + number)
        out.append(learn.learn(r, heads=heads))
    return out


# --- ppo.muon_lr_scale -------------------------------------------------------------------------------------


def one_step_learner(**ppo: Any) -> Learner:
    """One learning step an update, at rates where a step is far above FP32's rounding of the weights."""
    return plain_learner(epochs=1, minibatches=1, policy_lr=0.01, value_lr=0.01, optimizer="muon", **ppo)


@pytest.mark.parametrize("batched", [False, True], ids=["muon", "muon batched"])
def test_a_scaled_muon_step_moves_each_matrix_by_the_scale_and_adams_part_alike(batched: bool) -> None:
    """From the same state (momentum included: a resume of one save), one step at 0.2 moves every Muon matrix
    0.2x as far as at 1, to FP32's rounding, and every parameter Adam holds exactly as at 1."""
    whole = one_step_learner(muon_batched=batched)
    plain_updates(whole, 2)
    saved = copy.deepcopy(whole.state_dict())
    fifth = one_step_learner(muon_batched=batched, muon_lr_scale=0.2)
    fifth.load_state_dict(copy.deepcopy(saved))
    assert isinstance(fifth.optimizer, MuonAdam) and fifth.optimizer.scale == 0.2, (
        "a resume keeps the setting"
    )
    start = [p.detach().clone() for p in whole.model.parameters()]
    for learn in (whole, fifth):
        plain_updates(learn, 1, first=2)
    muon = {id(p) for p in whole.model.muon_parameters()}
    for begin, a, b in zip(start, whole.model.parameters(), fifth.model.parameters(), strict=True):
        if id(a) in muon:
            assert (a - begin).abs().max() > 1e-3
            torch.testing.assert_close(b - begin, 0.2 * (a - begin), rtol=1e-4, atol=1e-6)
        else:
            assert torch.equal(a, b)


def test_scaled_batched_steps_are_the_steps_one_at_a_time() -> None:
    learners = [one_step_learner(muon_batched=batched, muon_lr_scale=0.2) for batched in (False, True)]
    for learn in learners:
        plain_updates(learn, 3)
    for a, b in zip(learners[0].model.parameters(), learners[1].model.parameters(), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_the_scale_travels_with_copies_and_stays_out_of_the_saves() -> None:
    learn = one_step_learner(muon_lr_scale=0.2)
    assert isinstance(learn.optimizer, MuonAdam)
    assert copy.deepcopy(learn.optimizer).scale == 0.2
    assert "scale" not in learn.optimizer.state_dict(), "a save holds the optimizer's state, not its settings"
    learn.optimizer.load_state_dict(learn.optimizer.state_dict())
    assert learn.optimizer.scale == 0.2, "a load keeps the setting"


# --- ppo.weight_decay --------------------------------------------------------------------------------------

DECAY = 0.5


def heads_model(residual: bool) -> ModelConfig:
    return ModelConfig(
        shared=(8, 8),
        policy=(16, 16),
        value=(12, 12),
        value_head_layers=(12,),
        value_heads=("main", "goal"),
        residual=residual,
        activation="leaky_relu",
    )


def decay_learner(residual: bool, optimizer: str, **ppo: Any) -> Learner:
    config = {"policy_lr": 0.003, "value_lr": 0.002, "optimizer": optimizer}
    return plain_learner(model=heads_model(residual), **(config | ppo))


def pair_factors(learn: Learner, rates: dict[str, float], joint: int, value: int) -> dict[int, float]:
    """By parameter id: what the decay multiplies it by after an update of `joint` joint and `value` value
    steps at `rates`."""
    out = {}
    for name, pairs in learn.scale_free.items():
        steps = value if name == "value" else joint
        factor = (1.0 - rates[name] * DECAY) ** steps
        out |= {id(p): factor for pair in pairs for p in pair}
    return out


@pytest.mark.parametrize("residual", [False, True], ids=["plain", "residual"])
@pytest.mark.parametrize("optimizer", ["adam", "muon"])
def test_weight_decay_shrinks_each_pair_by_its_steps_and_the_outputs_stay(
    residual: bool, optimizer: str
) -> None:
    """Each LayerNorm-fed pair is the undecayed update's times (1 - rate x lambda)^steps (the update's 4
    joint steps), at the rate the steps ran at; everything else is the undecayed update's; on 500
    observations the outputs are the undecayed networks' to 1e-5."""
    plain, decayed = (
        decay_learner(residual, optimizer),
        decay_learner(residual, optimizer, weight_decay=DECAY),
    )
    pairs = decayed.model.scale_free_pairs()
    assert {name for name, _, _ in pairs} == {"shared", "policy", "value"}
    assert len(pairs) == 2 + 2 + 2 + 2, "the trunk's, the policy's, the critic's and both heads' layers"
    x = torch.randn(500, OBS, generator=torch.Generator().manual_seed(9))
    for number in range(2):
        rates = dict(decayed.learning_rates)
        stats = plain_updates(plain, 1, first=number) + plain_updates(decayed, 1, first=number)
        assert stats[0].gradient_steps == 4 and stats[0].value_only_steps == 0
        assert repr(stats[0]) == repr(stats[1]), "measured before the decay"
        factors = pair_factors(decayed, rates, joint=4, value=4)
        for a, b in zip(plain.model.parameters(), decayed.model.parameters(), strict=True):
            if id(b) in factors:
                assert factors[id(b)] < 1
                assert torch.equal(b, a * factors[id(b)])
            else:
                assert torch.equal(a, b)
        assert plain.learning_rates == decayed.learning_rates == rates
        assert_same_outputs(plain.model, decayed.model, x)
        with torch.no_grad():  # the plain run takes the same shrink, so the next update starts alike
            for a, b in zip(plain.model.parameters(), decayed.model.parameters(), strict=True):
                a.copy_(b)
    # 0.9 at once, about 70 updates' decay here: still the same networks (LayerNorm's eps shows as 1 / c^2:
    # 0.5 gives 2e-5 on these small layers).
    shrunk = decay_learner(residual, optimizer)
    with torch.no_grad():
        for _, w, b in shrunk.model.scale_free_pairs():
            w.mul_(0.9)
            b.mul_(0.9)
    assert_same_outputs(decay_learner(residual, optimizer).model, shrunk.model, x)


@torch.no_grad()
def assert_same_outputs(model: ActorCritic, other: ActorCritic, x: torch.Tensor) -> None:
    """Every output (the logits and each value head's) within 1e-5 of `model`'s, relative."""
    logits, heads = model.forward_heads(x)
    other_logits, other_heads = other.forward_heads(x)
    for mine, theirs in zip([logits, *heads.values()], [other_logits, *other_heads.values()], strict=True):
        assert float((theirs - mine).norm() / mine.norm()) < 1e-5


def test_a_frozen_policy_is_not_decayed_and_the_value_side_by_its_steps() -> None:
    plain, decayed = (decay_learner(False, "muon", weight_decay=w) for w in (0.0, DECAY))
    for learn in (plain, decayed):
        learn.policy_frozen = True
    rates = dict(decayed.learning_rates)
    for learn in (plain, decayed):
        torch.manual_seed(3)
        learn.learn(rollout_for(learn.model, 0))
    factors = pair_factors(decayed, rates, joint=0, value=4)
    shrunk = {p for p, f in factors.items() if f < 1}
    assert shrunk == {id(p) for pair in decayed.scale_free["value"] for p in pair}
    for a, b in zip(plain.model.parameters(), decayed.model.parameters(), strict=True):
        if id(b) in shrunk:
            torch.testing.assert_close(b, a * factors[id(b)], rtol=1e-6, atol=0)
        else:
            assert torch.equal(a, b)


def test_a_resumed_run_with_both_learns_exactly_as_one_that_never_stopped() -> None:
    """The GCO bot's networks and signals, as test_muon's resume (the update after the save)."""
    both = {"muon_lr_scale": 0.5, "weight_decay": DECAY}
    straight = gco_learner(CPU, False, **both)
    gco_updates(straight, 2)
    saved = copy.deepcopy(straight.state_dict())
    gco_updates(straight, 1, first=2)
    resumed = gco_learner(CPU, False, **both)
    torch.manual_seed(77)
    with torch.no_grad():
        for p in resumed.model.parameters():
            p.add_(torch.randn_like(p))  # other weights: everything comes from the save
    resumed.load_state_dict(saved)
    gco_updates(resumed, 1, first=2)
    for a, b in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    for p, q in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        for key, value in straight.optimizer.state[p].items():
            assert torch.equal(value, resumed.optimizer.state[q][key]), key


# --- Two cards (gloo): the decay needs no broadcast --------------------------------------------------------

_STUCK_SECONDS = 240


def _card(rank: int, size: int, store: str, out: str) -> None:
    faulthandler.dump_traceback_later(_STUCK_SECONDS, exit=True)
    cards = join(Path(store), rank, size, CPU)
    learn = decay_learner(True, "muon", weight_decay=DECAY, muon_lr_scale=0.5)
    learn.cards = cards
    for number in range(3):
        torch.manual_seed(20 + number)  # every card makes the whole rollout alike and learns its half
        r = rollout_for(learn.model, number)
        half = r.agents // size
        torch.manual_seed(30 + number + 100 * rank)
        learn.learn(agents_of(r, slice(rank * half, (rank + 1) * half)))
    torch.save([p.detach().clone() for p in learn.model.parameters()], f"{out}.{rank}")
    leave(clean=True)


@GLOO
def test_on_two_cards_every_card_decays_alike(tmp_path: Path) -> None:
    out = tmp_path / "card"
    mp.spawn(_card, args=(2, str(tmp_path / "store"), str(out)), nprocs=2, join=True)
    first, second = (torch.load(f"{out}.{rank}") for rank in (0, 1))
    assert all(torch.equal(a, b) and a.isfinite().all() for a, b in zip(first, second, strict=True))


# --- The columns -------------------------------------------------------------------------------------------


def _norm(tensors: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(t.detach().double().pow(2).sum()) for t in tensors))


@pytest.mark.parametrize(
    ("optimizer", "residual"), [("adam", False), ("muon", False), ("adam", True), ("muon", True)]
)
def test_the_size_columns_are_the_hand_computed_ones(optimizer: str, residual: bool) -> None:
    learn = plain_learner(model=heads_model(residual), optimizer=optimizer, popart=False)
    pairs = learn.model.scale_free_pairs()
    start = {
        name: math.sqrt(2 * sum(min(w.shape) for n, w, _ in pairs if n == name)) for name in learn.groups
    }
    for number in range(3):
        before = {id(w): w.detach().clone() for _, w, _ in pairs}
        stats = plain_updates(learn, 1, first=number)[0]
        params = list(learn.model.parameters())
        for name, group in learn.groups.items():
            assert stats.weight_norms[name] == pytest.approx(_norm(list(group)), rel=1e-6)
        assert math.sqrt(sum(v**2 for v in stats.weight_norms.values())) == pytest.approx(
            _norm(params), rel=1e-6
        )
        assert list(stats.turns) == list(stats.growths) == ["policy", "value", "shared"]
        for name in learn.groups:
            mine = [w for n, w, _ in pairs if n == name]
            size = _norm([before[id(w)] for w in mine])
            change = _norm([w.detach() - before[id(w)] for w in mine])
            assert stats.turns[name] == pytest.approx(change / size, rel=1e-5)
            assert stats.growths[name] == pytest.approx(size / start[name], rel=1e-6)
            if number == 0:  # a fresh orthogonal network
                assert stats.growths[name] == pytest.approx(1.0, abs=1e-6)


def test_growth_needs_orthogonal_starts_and_turns_a_layernorm() -> None:
    pytorch = plain_learner(model=dataclasses.replace(MUON_MODEL, init="pytorch"))
    stats = plain_updates(pytorch, 1)[0]
    assert set(stats.turns) == {"policy", "value", "shared"} and stats.growths == {}
    bare = plain_learner(model=ModelConfig(shared=(8,), policy=(16,), value=(12,), layer_norm=False))
    stats = plain_updates(bare, 1)[0]
    assert stats.turns == stats.growths == {} and bare.scale_free == {}
    assert set(stats.weight_norms) == {"policy", "value", "shared"}


SETTINGS = """
[run]
seed = 7
[env]
num_arenas = 4
threads = 2
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 160
minibatches = 2
optimizer = "muon"
[[state_setters]]
name = "kickoff"
"""


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_runs_metrics_have_the_columns(tmp_path: Path) -> None:
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    parsed = parse_toml(SETTINGS, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, "sizes", file)
    training = TrainingRun(settings, folder)
    rows = [training.iteration() for _ in range(2)]
    training.metrics.close()
    with folder.metrics.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    columns = [f"{kind}_{net}" for kind in ("weight_norm", "turn", "growth") for net in ("policy", "value")]
    assert set(columns) <= set(header)
    assert header.index("update_norm_value") < header.index("weight_norm_policy")
    assert rows[0]["growth_policy"] == pytest.approx(1.0, abs=1e-6)
    for row in rows:
        assert all(math.isfinite(row[c]) and row[c] > 0 for c in columns)


# --- Settings ----------------------------------------------------------------------------------------------


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path("."))


def test_the_settings_default_to_as_before_and_take_their_ranges() -> None:
    config = parse("")
    assert config.ppo.muon_lr_scale == 1.0 and config.ppo.weight_decay == 0.0
    both = parse('[ppo]\noptimizer = "muon"\nmuon_lr_scale = 0.067\nweight_decay = 1.0\n')
    assert both.ppo.muon_lr_scale == 0.067 and both.ppo.weight_decay == 1.0
    assert parse("[ppo]\nweight_decay = 0.5\n").ppo.weight_decay == 0.5, "Adam takes it too"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('[ppo]\noptimizer = "muon"\nmuon_lr_scale = 0\n', "muon_lr_scale: must be above 0"),
        ("[ppo]\nmuon_lr_scale = 0.2\n", "muon_lr_scale: only scales Muon's matrices"),
        (
            "[ppo]\nweight_decay = 0.1\n[model]\nlayer_norm = false\n",
            "ppo.weight_decay: needs model.layer_norm",
        ),
        ("[ppo]\nweight_decay = 10\n", "weight_decay: x the larger of policy_lr and value_lr"),
        ("[ppo]\nweight_decay = -0.1\n", "weight_decay"),
    ],
)
def test_settings_that_cannot_work_are_refused(text: str, problem: str) -> None:
    with pytest.raises(ConfigError) as caught:
        parse(text)
    assert re.search(re.escape(problem), str(caught.value)), str(caught.value)


# --- On a graphics card (not run without one) -------------------------------------------------------------


@NEEDS_GPU
@pytest.mark.parametrize("batched", [False, True], ids=["muon", "muon batched"])
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_graphed_gco_steps_with_both_give_the_eager_updates_to_the_bit(fp16: bool, batched: bool) -> None:
    """The scale is captured in the graphed step (a change needs a restart) and the decay runs outside the
    graphs on the same tensors: graphed updates are the eager ones, weights and every measurement."""
    both = {"muon_lr_scale": 0.2, "weight_decay": DECAY}
    eager, graphed = (gco_learner(CUDA, on, fp16, batched, **both) for on in (False, True))
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
