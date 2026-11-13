"""The extra value heads' pass before each update: it runs only the heads it is asked for
(never the main one), the same numbers to the bit; with ppo.lean_heads_pass it values only the learner's
agents and keeps run.fp16_sums' FP16 sums, the same learning to rounding. Old-code copies below are the
pass as it was in an earlier version; the trainers' runs were also checked against an earlier commit."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import nitro.trainer
from nitro.config import load_config
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig, PPOConfig
from nitro.learner import HeadValues, Learner, LearnStats, head_values_on
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout
from nitro.runs import RunFolder
from nitro.trainer import Trainer
from nitro.training import TrainingRun
from test_residual_heads import HEADS
from test_trainer import CountingEnvs

ROOT = Path(__file__).resolve().parents[1]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
CPU = torch.device("cpu")
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
OBS, ACTIONS, STEPS, AGENTS = 4, 3, 7, 8
NAMES = ("goal", "vdag1", "vdag2")
ROWS = torch.tensor([True, True, False, True, True, False, True, True])  # 2 and 5: past versions' cars
GCO = {"goal_critic_weight": 0.25, "vdag_beta": 0.3, "graphed_steps": False}


@torch.no_grad()
def old_head_values_on(
    model: ActorCritic,
    names: tuple[str, ...],
    fp16: bool,
    rollout: Rollout,
    last_obs: torch.Tensor,
    cuts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    chunk: int = 65_536,
) -> HeadValues:
    """`head_values_on` in an earlier version, to the letter: every head, every agent."""

    def run(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        pieces: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        for part in torch.split(obs, chunk):
            with torch.autocast(part.device.type, dtype=torch.float16, enabled=fp16):
                values = model.head_values(part)
            for name in names:
                pieces[name].append(values[name].float())
        return {name: torch.cat(v) for name, v in pieces.items()}

    steps, agents = rollout.steps, rollout.agents
    now = {k: v.view(steps, agents) for k, v in run(rollout.obs.reshape(steps * agents, -1)).items()}
    last = run(last_obs)
    reached = {name: torch.cat([now[name][1:], last[name][None]]) for name in names}
    if cuts is not None and cuts[0].numel():
        cut_steps, cut_agents, cut_obs = cuts
        for name, values in run(cut_obs).items():
            reached[name][cut_steps, cut_agents] = values
    return HeadValues(now, reached)


def old_pass(model: ActorCritic, names: tuple[str, ...], *args: Any, **lean: Any) -> HeadValues:
    """The trainer's call answered by the old code (the setting off: nothing lean asked for)."""
    assert lean == {"columns": None, "sums": False}
    return old_head_values_on(model, names, *args)


def model_with_heads(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(OBS, ACTIONS, HEADS)


Cuts = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def rollout_with_cuts(model: ActorCritic, seed: int = 0) -> tuple[Rollout, torch.Tensor, Cuts]:
    """A rollout with a goal at step 4 and cut episodes at step 1 (agents 0 and 2) and step 5 (agent 7), their
    final states and the states after its last step."""
    generator = torch.Generator().manual_seed(seed)
    r = Rollout.allocate(STEPS, AGENTS, OBS, ACTIONS, CPU)
    r.obs.copy_(torch.randn(STEPS, AGENTS, OBS, generator=generator))
    r.mask.fill_(True)
    torch.manual_seed(seed)
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    r.actions.copy_(torch.distributions.Categorical(logits=log_probs).sample())
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    r.rewards.copy_(torch.randn(STEPS, AGENTS, generator=generator))
    r.terminal[4, :] = True
    cuts = (torch.tensor([1, 1, 5]), torch.tensor([0, 2, 7]), torch.randn(3, OBS, generator=generator))
    last = torch.randn(AGENTS, OBS, generator=generator)
    r.truncated[cuts[0], cuts[1]] = True
    with torch.no_grad():
        r.truncated_values[cuts[0], cuts[1]] = model.normalised_value(cuts[2])
        r.bootstrap_values.copy_(model.normalised_value(last))
    return r, last, cuts


def _calls(head: torch.nn.Module) -> list[int]:
    """The rows each run of `head` (its output layer) saw."""
    seen: list[int] = []
    assert isinstance(head, torch.nn.Sequential)
    head[-1].register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0].shape[0]))
    return seen


@pytest.mark.parametrize("names", [("goal",), NAMES, ("vdag2", "main")])
def test_named_heads_give_every_heads_own_numbers_to_the_bit(names: tuple[str, ...]) -> None:
    model = model_with_heads()
    x = torch.randn(32, OBS)
    main = _calls(model.value_heads["main"])
    with torch.no_grad():
        every = model.head_values(x)
        assert len(main) == 1 and list(every) == list(model.value_heads), "no names: every head, as before"
        some = model.head_values(x, names)
    assert list(some) == [name for name in model.value_heads if name in names], "in value_heads' order"
    assert len(main) == 1 + ("main" in names)
    for name in names:
        assert torch.equal(some[name], every[name])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=NEEDS_GPU)])
def test_named_heads_on_the_gco_starters_networks_to_the_bit(device: str) -> None:
    """A big GCO bot's networks (22.9M weights, observation 148): FP32, and FP16 autocast on a card."""
    model_config = load_config(GCO_STARTER).model
    torch.manual_seed(0)
    model = ActorCritic(148, 90, model_config).to(device)
    x = torch.randn(300, 148, device=device)
    for fp16 in (False, True) if device == "cuda" else (False,):
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=fp16):
            every = model.head_values(x)
            some = model.head_values(x, NAMES)
        assert list(some) == list(NAMES)
        for name in NAMES:
            assert torch.equal(some[name], every[name]), (name, fp16)


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("chunk", [65_536, 5])
def test_the_pass_without_the_main_head_is_the_old_codes_to_the_bit(fp16: bool, chunk: int) -> None:
    model = model_with_heads()
    r, last, cuts = rollout_with_cuts(model)
    main = _calls(model.value_heads["main"])
    new = head_values_on(model, NAMES, fp16, r, last, cuts, chunk)
    assert not main, "the main head's answer was never used: it no longer runs"
    old = old_head_values_on(model, NAMES, fp16, r, last, cuts, chunk)
    for name in NAMES:
        assert torch.equal(new.now[name], old.now[name]) and torch.equal(new.reached[name], old.reached[name])


@pytest.mark.parametrize("chunk", [65_536, 13, 3])  # 13: two steps a part; 3: one step, more columns than 3
def test_with_the_learners_columns_their_values_match_and_the_others_are_zero(chunk: int) -> None:
    model = model_with_heads()
    r, last, cuts = rollout_with_cuts(model)
    columns = ROWS.nonzero().squeeze(1)
    full = head_values_on(model, NAMES, False, r, last, cuts)
    lean = head_values_on(model, NAMES, False, r, last, cuts, chunk, columns=columns)
    with torch.no_grad():
        cut = model.head_values(cuts[2], NAMES)
    for name in NAMES:
        for got, expected in ((lean.now[name], full.now[name]), (lean.reached[name], full.reached[name])):
            assert got.shape == (STEPS, AGENTS)
            torch.testing.assert_close(got[:, ROWS], expected[:, ROWS], atol=1e-5, rtol=0)
            assert not got[:, ~ROWS].any(), "0 on the past versions' columns"
        reached = lean.reached[name]
        torch.testing.assert_close(reached[1, 0], cut[name][0], atol=1e-5, rtol=0)  # a learner's cut
        torch.testing.assert_close(reached[5, 7], cut[name][2], atol=1e-5, rtol=0)  # the last learner column
        assert full.reached[name][1, 2] == cut[name][1], "the past version's cut, in the full pass"
        assert reached[1, 2] == 0, "dropped"
        torch.testing.assert_close(reached[-1, ROWS], full.reached[name][-1, ROWS], atol=1e-5, rtol=0)


def test_with_the_learners_columns_the_gco_starters_networks_match_the_full_pass() -> None:
    """A big GCO bot's networks (FP32), 64 agents, every 10th a past version: here at most 9.5e-7 apart."""
    model_config = load_config(GCO_STARTER).model
    torch.manual_seed(0)
    model = ActorCritic(148, 90, model_config)
    generator = torch.Generator().manual_seed(0)
    steps, agents = 4, 64
    r = Rollout.allocate(steps, agents, 148, 90, CPU)
    r.obs.copy_(torch.randn(steps, agents, 148, generator=generator))
    last = torch.randn(agents, 148, generator=generator)
    rows = torch.ones(agents, dtype=torch.bool)
    rows[::10] = False
    cuts = (torch.tensor([1, 2]), torch.tensor([1, 10]), torch.randn(2, 148, generator=generator))
    full = head_values_on(model, NAMES, False, r, last, cuts)
    lean = head_values_on(model, NAMES, False, r, last, cuts, columns=rows.nonzero().squeeze(1))
    for name in NAMES:
        for got, expected in ((lean.now[name], full.now[name]), (lean.reached[name], full.reached[name])):
            torch.testing.assert_close(got[:, rows], expected[:, rows], atol=1e-5, rtol=0)
            assert not got[:, ~rows].any()
        assert full.reached[name][2, 10] != 0 and lean.reached[name][2, 10] == 0, "a past version's cut"


def test_the_other_agents_states_are_never_read() -> None:
    torch.manual_seed(0)
    learner = Learner(
        ActorCritic(OBS, ACTIONS, HEADS), PPOConfig(epochs=1, minibatches=2, **GCO), 0.9, CPU, fp16=False
    )
    learner.goal_gamma = 0.99
    r, last, (steps, agents, cut_obs) = rollout_with_cuts(learner.model)
    r.obs[:, ~ROWS] = math.nan
    last[~ROWS] = math.nan
    cut_obs[1] = math.nan  # agent 2's
    cuts = (steps, agents, cut_obs)
    every = head_values_on(learner.model, NAMES, False, r, last, cuts)
    assert every.now["goal"][:, ~ROWS].isnan().all() and every.reached["goal"][1, 2].isnan(), "read there"
    heads = head_values_on(learner.model, NAMES, False, r, last, cuts, columns=ROWS.nonzero().squeeze(1))
    for values in (*heads.now.values(), *heads.reached.values()):
        assert values.isfinite().all()
    stats = learner.learn(r, ROWS, heads=heads)
    assert stats.losses_finite and stats.signals
    assert all(math.isfinite(v) for v in stats.signals.values()), stats.signals


def test_fp16_sums_are_kept_only_during_the_pass_it_is_asked_to() -> None:
    model = model_with_heads()
    r, last, cuts = rollout_with_cuts(model)
    flags: list[bool] = []
    model.value.register_forward_hook(
        lambda *_: flags.append(torch.backends.cuda.matmul.allow_fp16_accumulation)
    )
    before = torch.backends.cuda.matmul.allow_fp16_accumulation
    head_values_on(model, NAMES, False, r, last, cuts)
    head_values_on(model, NAMES, False, r, last, cuts, sums=True)
    assert flags == [before] * 3 + [True] * 3  # the states, the last states, the cut ones
    assert torch.backends.cuda.matmul.allow_fp16_accumulation == before


def counting_trainer(lean: bool, fp16_sums: str = "off") -> Trainer:
    """5 agents, 2 and 4 driven by someone else (self-play's past versions, here a stand-in): their rows are
    cut every third step, so the cuts fall on both kinds of agent."""
    envs = CountingEnvs((2, 3))
    ppo = PPOConfig(epochs=1, minibatches=1, lean_heads_pass=lean, **GCO)  # type: ignore[arg-type]
    torch.manual_seed(0)
    model = ActorCritic(2, 3, HEADS)
    learner = Learner(model, ppo, gamma=0.9, device=CPU, fp16=False)
    learner.goal_gamma = 0.99
    learner.fp16_sums = fp16_sums
    rows = torch.tensor([True, True, False, True, False])
    past = SimpleNamespace(learner_rows=rows, group_opponents=lambda group: ())
    return Trainer(envs, model, learner, 5 * 6, CPU, fp16=False, cuda_graphs=False, selfplay=past)  # type: ignore[arg-type]


def digest(model: torch.nn.Module, stats: list[LearnStats]) -> tuple[str, str]:
    weights = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        weights.update(name.encode() + tensor.cpu().contiguous().numpy().tobytes())
    measured = repr([dataclasses.asdict(s) for s in stats]).encode()
    return weights.hexdigest(), hashlib.sha256(measured).hexdigest()


def recorded(monkeypatch: pytest.MonkeyPatch, answer: Callable[..., HeadValues]) -> list[dict[str, Any]]:
    """What the trainer asks the pass for, answered by `answer`."""
    asked: list[dict[str, Any]] = []

    def recording(*args: Any, **kwargs: Any) -> HeadValues:
        asked.append(kwargs)
        return answer(*args, **kwargs)

    monkeypatch.setattr(nitro.trainer, "head_values_on", recording)
    return asked


def counting_run(
    lean: bool, monkeypatch: pytest.MonkeyPatch, old: bool = False
) -> tuple[Trainer, list[LearnStats], list[dict[str, Any]]]:
    """Three updates of `counting_trainer`, the pass by the old code if `old`."""
    trainer = counting_trainer(lean)
    asked = recorded(monkeypatch, old_pass if old else head_values_on)
    stats = [trainer.iteration().learn for _ in range(3)]
    assert len(asked) == 3
    return trainer, stats, asked


def test_off_three_updates_are_the_old_codes_to_the_bit_and_on_to_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, old_stats, _ = counting_run(False, monkeypatch, old=True)
    off, off_stats, asked = counting_run(False, monkeypatch)
    assert all(a == {"columns": None, "sums": False} for a in asked)
    assert digest(off.model, off_stats) == digest(old.model, old_stats)
    _, on_stats, asked = counting_run(True, monkeypatch)
    assert all(a["columns"].tolist() == [0, 1, 3] and a["sums"] is False for a in asked)
    for a, b in zip(on_stats, off_stats, strict=True):
        for field in ("policy_loss", "value_loss", "entropy", "explained_variance", "policy_change"):
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-4, abs=1e-6), field
        assert a.signals == pytest.approx(b.signals, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize(
    ("sums", "expected"), [("off", False), ("playing", False), ("forward", True), ("all", True)]
)
def test_the_lean_pass_keeps_fp16_sums_as_learnings_forward_passes_do(
    monkeypatch: pytest.MonkeyPatch, sums: str, expected: bool
) -> None:
    for lean in (False, True):
        trainer = counting_trainer(lean, sums)
        asked = recorded(monkeypatch, head_values_on)
        trainer.iteration()
        assert asked[0]["sums"] is (lean and expected)


def test_the_setting_is_off_by_default_and_loads(tmp_path: Path) -> None:
    assert PPOConfig().lean_heads_pass is False
    path = tmp_path / "settings.toml"
    path.write_text("[ppo]\ngraphed_steps = false\nlean_heads_pass = true\n", encoding="utf-8")
    assert load_config(path).ppo.lean_heads_pass is True


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
advantage_keep = 0.5
entropy_target = 0.5
lean_heads_pass = {lean}
[selfplay]
enabled = true
past_share = 0.5
active_opponents = 1
snapshot_steps = 1000000
[[state_setters]]
name = "kickoff"
"""


@pytest.fixture
def kept_determinism() -> Any:
    """run.deterministic turns PyTorch's deterministic algorithms on for the process; set back after."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


def self_play_run(tmp_path: Path, name: str, lean: bool) -> tuple[TrainingRun, list[LearnStats]]:
    """A past version joins at the first save and drives Orange in half the arenas; then three updates."""
    text = SETTINGS.format(lean=str(lean).lower())
    file = tmp_path / f"{name}.toml"
    file.write_text(text, encoding="utf-8")
    config = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))
    run = TrainingRun(config, RunFolder.create(tmp_path / "runs", name, file))
    run.selfplay.next_snapshot_steps = 1
    run.iteration()
    run.checkpoint()
    rows = run.selfplay.learner_rows
    assert run.selfplay.active[0] is not None and rows is not None and 0 < int(rows.sum()) < rows.numel()
    stats: list[LearnStats] = []
    learn = run.trainer.learner.learn

    def recording(*args: Any, **kwargs: Any) -> LearnStats:
        stats.append(learn(*args, **kwargs))
        return stats[-1]

    run.trainer.learner.learn = recording  # type: ignore[method-assign]
    for _ in range(3):
        run.iteration()
    run.metrics.close()
    return run, stats


@NEEDS_MESHES
@pytest.mark.usefixtures("kept_determinism")
def test_a_self_play_run_off_is_the_old_codes_to_the_bit_and_on_to_rounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        recorded(patched, old_pass)
        old, old_stats = self_play_run(tmp_path, "old", lean=False)
    asked = recorded(monkeypatch, head_values_on)
    off, off_stats = self_play_run(tmp_path, "off", lean=False)
    assert asked and all(a == {"columns": None, "sums": False} for a in asked)
    assert digest(off.model, off_stats) == digest(old.model, old_stats)
    asked.clear()
    on, on_stats = self_play_run(tmp_path, "on", lean=True)
    rows = on.selfplay.learner_rows
    assert rows is not None
    assert torch.equal(asked[-1]["columns"], rows.nonzero().squeeze(1)), "the learner's cars only"
    for a, b in zip(on_stats, off_stats, strict=True):
        for field in ("policy_loss", "value_loss", "entropy", "explained_variance"):
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-4, abs=1e-6), field
        assert a.signals == pytest.approx(b.signals, rel=1e-4, abs=1e-6)
