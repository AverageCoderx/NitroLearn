"""Copying converted replays while a bot trains (`[replay_mix]`): the copy steps move the
policy toward the players' choices and never the value network, take their share of the learning steps, keep
their optimizer and place in the replays in the saves, and change nothing when off. Built on a small store of
replicar-style conversions, as tests/test_pretrain_large.py makes them."""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("pyarrow")

from nitro import _core, console, replays, reward_check
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig, ReplayMixConfig
from nitro.model import build_networks
from nitro.replay_mix import ReplayMix, shown
from nitro.runs import RunFolder
from nitro.training import TrainingRun
from test_pretrain_large import lean_file
from test_replays import replicar_file

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
OBS = 148  # the default observation's size in 1v1 (nitro_v2), as the store's samples have it
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
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 320
minibatches = 2
[[state_setters]]
name = "kickoff"
"""
MIX = ReplayMixConfig(enabled=True, share=0.5, lr=1e-3, minibatch=64)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A store of three converted 1v1 replays in the project's replays folder (moved into `tmp_path`)."""
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    replicar_file(tmp_path / "one.parquet")
    lean_file(tmp_path / "one.lean", *replays.read_replicar(tmp_path / "one.parquet"))
    data = (tmp_path / "one.lean").read_bytes()
    for name in ("r0", "r1", "r2"):
        assert replays.keep(data, name, 1)
    return replays.store_folder()


def settings(tmp_path: Path, mix: ReplayMixConfig = MIX, **sections: Any) -> NitroConfig:
    parsed = parse_toml(SETTINGS, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    paths = dataclasses.replace(parsed.paths, collision_meshes=MESHES)
    return dataclasses.replace(parsed, paths=paths, replay_mix=mix, **sections)


def update(policy_steps: int, value_only: int = 0) -> Any:
    """What the copy steps read of an update's statistics."""
    return SimpleNamespace(gradient_steps=policy_steps + value_only, value_only_steps=value_only)


def copy_and_value(model: torch.nn.Module, mix: ReplayMix) -> tuple[dict[str, Any], dict[str, Any]]:
    mine = {id(p) for p in mix.params}
    copied = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) in mine}
    rest = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in mine}
    return copied, rest


@pytest.mark.parametrize("shared", [(), (32,)])
def test_copy_steps_move_the_policy_toward_the_players_and_never_the_value_network(
    tmp_path: Path, store: Path, shared: tuple[int, ...]
) -> None:
    config = settings(tmp_path)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, shared=shared))
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, torch.device("cpu"), fp16=False)
    try:
        policy_before, value_before = copy_and_value(model, mix)
        assert not any(n.startswith("value.") for n in policy_before) and value_before
        assert any(n.startswith("shared.") for n in policy_before) == bool(shared), "a shared trunk is copied"
        first = mix.learn(update(4))
        assert first is not None and first.steps == 4, "share 0.5: one copy step per policy step"
        for _ in range(20):
            last = mix.learn(update(4))
        assert last is not None and last.loss < first.loss - 0.2, (first, last)
        assert last.agreement == 1.0 > first.agreement
        policy_after, value_after = copy_and_value(model, mix)
        for name, weights in value_after.items():
            assert torch.equal(weights, value_before[name]), f"{name}: the value network is never copied into"
        assert any(not torch.equal(policy_after[n], policy_before[n]) for n in policy_before)
        assert all(p.grad is None for p in model.parameters()), "no gradient is left for PPO's step"
        assert mix.steps == 84 and len(mix.window) == 3 and mix.next > 3, "the store's 3 replays, in cycles"
    finally:
        mix.close()


def test_the_steps_are_their_share_of_the_policy_steps_and_none_while_the_policy_is_held(
    tmp_path: Path, store: Path
) -> None:
    config = settings(tmp_path, dataclasses.replace(MIX, share=0.2))
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, torch.device("cpu"), fp16=False)
    try:
        counts = [mix.learn(update(3)) for _ in range(4)]
        assert [c.steps for c in counts if c is not None] == [0, 1, 1, 1], "0.75 an update, carried over"
        held = {n: p.detach().clone() for n, p in model.named_parameters()}
        warm_up = mix.learn(update(0, value_only=8))  # a value warm-up: every step trains the value only
        assert warm_up is not None and warm_up.steps == 0
        assert all(torch.equal(p, held[n]) for n, p in model.named_parameters())
    finally:
        mix.close()


def test_the_saved_copy_steps_carry_on_exactly_and_older_or_misfit_saves_start_afresh(
    tmp_path: Path, store: Path
) -> None:
    config = settings(tmp_path)
    cpu = torch.device("cpu")
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, cpu, fp16=False)
    try:
        for _ in range(3):
            mix.learn(update(3))
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "mix": mix.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        assert saved["mix"]["optimizer"]["state"], "the copy optimizer's moments are saved"
        again = build_networks(config, OBS, _core.num_actions())
        again.load_state_dict(saved["model"])
        resumed = ReplayMix(config, again, OBS, cpu, fp16=False, state=saved["mix"])
        try:
            for a, b in zip(mix.optimizer.state.values(), resumed.optimizer.state.values(), strict=True):
                assert all(torch.equal(a[k], b[k]) for k in a)
            for _ in range(3):
                mix.learn(update(3))
                resumed.learn(update(3))
            for (name, a), b in zip(model.state_dict().items(), again.state_dict().values(), strict=True):
                assert torch.equal(a, b), f"{name} differs after carrying on"
        finally:
            resumed.close()
        wider = dataclasses.replace(config, model=dataclasses.replace(config.model, policy=(48, 32)))
        other = build_networks(wider, OBS, _core.num_actions())
        fresh = ReplayMix(wider, other, OBS, cpu, fp16=False, state=saved["mix"])
        fresh.close()
        assert not fresh.optimizer.state and fresh.next == saved["mix"]["next"], "a fresh optimizer only"
    finally:
        mix.close()


def test_without_converted_replays_of_its_size_it_says_so_once_and_copies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    config = settings(tmp_path)
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, torch.device("cpu"), fp16=False)
    assert not mix.off and not mix.notices, "nothing is looked for before the first update"
    with caplog.at_level(logging.WARNING, logger="nitro"):
        assert mix.learn(update(8)) is None and mix.learn(update(8)) is None
    assert mix.off and len(mix.notices) == 1
    assert "no converted 1v1 replays" in mix.notices[0] and "training goes on" in mix.notices[0]
    assert str(tmp_path) not in mix.notices[0], "no folder of this PC in the message"
    assert sum("no converted" in r.message for r in caplog.records) == 1
    mix.close()


@NEEDS_MESHES
def test_a_run_without_converted_replays_says_so_once_and_trains_as_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    weights = []
    for name, mix in [("off", ReplayMixConfig()), ("on", MIX)]:
        run = _run(settings(tmp_path, mix), _folder(tmp_path, name))
        rows = [run.iteration() for _ in range(2)]
        _closed(run)
        weights.append(run.model.state_dict())
        if name == "on":
            told = [w for w in run.console_warnings if "no converted 1v1 replays" in w]
            assert len(told) == 1 and all(row["replay_mix_steps"] == "" for row in rows)
    for key, value in weights[0].items():
        assert torch.equal(value, weights[1][key]), f"{key} differs"


def test_a_replay_folder_that_is_not_here_is_named_and_copies_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The folder is shown as it resolves now (a pack's unpacked settings point it elsewhere), a bot
    unpacked here hears that packs carry no folders, and the check needs nothing started."""
    gone = tmp_path / "runs" / "replays" / "kickoffs-1v1"
    config = settings(tmp_path, dataclasses.replace(MIX, replays=(gone,)))
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, torch.device("cpu"), fp16=False, moved=True)
    refusal = mix.missing()  # what run.missing_data = 'stop' lists: no "training goes on"
    assert refusal and mix._pool is None and not mix.notices, "found without starting anything"
    assert refusal == (
        f"replay_mix: no converted 1v1 replays: {shown(gone)} is not here (packs do not carry replay "
        "folders). Copy the replays there (or point replay_mix.replays at where they are)."
    )
    with caplog.at_level(logging.WARNING, logger="nitro"):
        assert mix.learn(update(8)) is None and mix.learn(update(8)) is None
    why = mix.off
    assert mix.notices == [why]
    assert sum(r.message == why for r in caplog.records) == 1
    assert shown(gone) in why and shown(gone).endswith(str(Path("runs", "replays", "kickoffs-1v1")))
    assert why == (
        f"replay_mix: no converted 1v1 replays: {shown(gone)} is not here (packs do not carry replay "
        "folders); training goes on without copying replays. Copy the replays there (or point "
        "replay_mix.replays at where they are) and start the bot again."
    ), "the folder named once"
    assert str(tmp_path) not in why, "no folder of this PC in the message"
    mix.close()
    here = ReplayMix(config, model, OBS, torch.device("cpu"), fp16=False)
    assert here.missing() == (
        f"replay_mix: no converted 1v1 replays: {shown(gone)} is not here. Copy the replays there (or point "
        "replay_mix.replays at where they are)."
    )
    # One folder here without replays and one gone: each named once.
    empty = tmp_path / "empty"
    empty.mkdir()
    both = settings(tmp_path, dataclasses.replace(MIX, replays=(empty, gone)))
    two = ReplayMix(both, model, OBS, torch.device("cpu"), fp16=False)
    assert two.missing().startswith(
        f"replay_mix: no converted 1v1 replays in {shown(empty)}, and {shown(gone)} is not here. Copy"
    )
    inside = replays.ROOT / "runs" / "replays" / "kickoffs-1v1"
    assert shown(inside) == str(Path(replays.ROOT.name, "runs", "replays", "kickoffs-1v1"))


@NEEDS_MESHES
def test_a_run_whose_replay_folder_is_gone_says_so_at_every_save_and_in_its_report(tmp_path: Path) -> None:
    from nitro.health import TrainingProblem

    config = settings(tmp_path, dataclasses.replace(MIX, replays=(tmp_path / "gone",)))
    folder = _folder(tmp_path, "moved")
    run = _run(config, folder)
    assert run.missing_notes == [], "'warn' looks for the replays at the first iteration, as before"
    assert run.iteration()["replay_mix_steps"] == ""
    run.checkpoint()
    run.iteration()
    run.checkpoint(continue_after=False)
    told = [w for w in run.recent_warnings if "gone is not here; training goes on" in w]
    assert len(told) == 3, "at the first iteration and again at each save"
    run.write_report()
    report = folder.report.read_text(encoding="utf-8")
    assert "replay_mix is on in the settings but copies nothing: " in report and "gone is not here" in report
    _closed(run)
    # With 'stop', training refuses to start; what only plays (reward-check, a copy's save) is not refused.
    stopping = dataclasses.replace(config, run=dataclasses.replace(config.run, missing_data="stop"))
    resumed = _run(stopping, folder, folder.load_latest())
    with pytest.raises(TrainingProblem) as refused:
        resumed.run(threading.Event())
    assert "no converted 1v1 replays: " in str(refused.value) and "gone is not here." in str(refused.value)
    assert "training goes on" not in str(refused.value)
    assert resumed.iterations == 2 and resumed.replay_mix is not None and not resumed.replay_mix.off
    _closed(resumed)
    text = SETTINGS.replace("[run]\n", '[run]\nmissing_data = "stop"\n') + (
        f'[paths]\ncollision_meshes = "{MESHES.as_posix()}"\n'
        f'[replay_mix]\nenabled = true\nreplays = ["{(tmp_path / "gone").as_posix()}"]\n'
    )
    assert reward_check.check(folder, text, 320, say=lambda line: None)["agent_steps"] >= 320


def test_replays_that_do_not_fit_the_bot_are_said_and_left_alone(tmp_path: Path, store: Path) -> None:
    config = settings(tmp_path)
    model = build_networks(config, OBS + 1, _core.num_actions())
    mix = ReplayMix(config, model, OBS + 1, torch.device("cpu"), fp16=False)
    try:
        stats = mix.learn(update(4))
        assert stats is not None and stats.steps == 0 and mix.learn(update(4)) is None
        assert mix.off and "none of the 3 converted replays could be used" in mix.off
        assert f"3 observations of {OBS} numbers, where the bot's have {OBS + 1}" in mix.off
    finally:
        mix.close()


def test_the_settings(tmp_path: Path) -> None:
    default = NitroConfig()
    assert not default.replay_mix.enabled and not default.replay_mix.active
    assert default.replay_mix.share == 0.2 and default.replay_mix.lr < default.ppo.policy_lr
    assert not ReplayMixConfig(enabled=True, share=0.0).active
    text = SETTINGS + "[replay_mix]\nenabled = true\nlr = 1e-3\nreplays = 'mine'\n"
    parsed = parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    assert parsed.replay_mix.replays == (tmp_path / "mine",)
    assert any("replay_mix.lr" in note for note in parsed.advice()), "a rate not below training's is said"
    assert not any(
        "replay_mix" in note for note in settings(tmp_path, ReplayMixConfig(enabled=True)).advice()
    )


def test_the_console_report_shows_the_copying() -> None:
    row = {"agent_steps": 1, "iteration": 1, "replay_mix_steps": 2, "replay_mix_loss": 1.234}
    row |= {"replay_mix_agreement": 0.5, "replay_mix_wait_seconds": 0.0}
    lines = console.iteration_report(row, console.Facts(name="b", running_seconds=1, steps_per_second=1))
    assert any("Replays" in line and "loss 1.234" in line and "50.0%" in line for line in lines)
    plain = console.iteration_report({"agent_steps": 1}, console.Facts("b", 1, 1))
    assert not any("Replays" in line for line in plain)


def _run(config: NitroConfig, folder: RunFolder, state: dict[str, Any] | None = None) -> TrainingRun:
    return TrainingRun(config, folder, state)


def _closed(run: TrainingRun) -> None:
    run.metrics.close()
    if run.replay_mix is not None:
        run.replay_mix.close()


def _folder(tmp_path: Path, name: str) -> RunFolder:
    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    return RunFolder.create(tmp_path / "runs", name, file)


@NEEDS_MESHES
def test_off_or_a_share_of_0_trains_bit_for_bit_as_before(tmp_path: Path, store: Path) -> None:
    weights = []
    for name, mix in [
        ("before", ReplayMixConfig()),
        ("off", dataclasses.replace(MIX, enabled=False)),
        ("none", dataclasses.replace(MIX, share=0.0)),
        ("on", MIX),
    ]:
        run = _run(settings(tmp_path, mix), _folder(tmp_path, name))
        run.iteration()
        row = run.iteration()
        assert ("replay_mix_steps" in row) == (name == "on")
        weights.append({k: v.clone() for k, v in run.model.state_dict().items()})
        path = run.checkpoint(continue_after=False)
        assert ("replay_mix" in torch.load(path / "state.pt", weights_only=True)) == (name == "on")
        _closed(run)
    for other in weights[1:3]:
        for key, value in weights[0].items():
            assert torch.equal(value, other[key]), f"{key} differs"
    assert any(not torch.equal(value, weights[3][key]) for key, value in weights[0].items())


@NEEDS_MESHES
def test_a_run_with_copy_steps_resumes_exactly_and_an_older_save_takes_them_up(
    tmp_path: Path, store: Path
) -> None:
    config = settings(tmp_path)
    straight = _run(config, _folder(tmp_path, "straight"))
    for _ in range(2):
        straight.iteration()
    straight.checkpoint()
    for _ in range(2):
        row = straight.iteration()
    assert row["replay_mix_steps"] == 4 and row["replay_mix_loss"] > 0
    _closed(straight)
    folder = _folder(tmp_path, "stopped")
    stopped = _run(config, folder)
    for _ in range(2):
        stopped.iteration()
    stopped.checkpoint(continue_after=False)
    _closed(stopped)
    resumed = _run(config, folder, folder.load_latest())
    for _ in range(2):
        resumed.iteration()
    _closed(resumed)
    for name, weights in straight.model.state_dict().items():
        assert torch.equal(weights, resumed.model.state_dict()[name]), f"{name} differs after resuming"

    # A bot trained without it takes it up part-way: its PPO optimizer and steps carry on as they were.
    old = _folder(tmp_path, "old")
    first = _run(settings(tmp_path, ReplayMixConfig()), old)
    first.iteration()
    first.checkpoint(continue_after=False)
    _closed(first)
    state = old.load_latest()
    assert state is not None and "replay_mix" not in state
    later = _run(config, old, state)
    assert later.replay_mix_state is None, "the state lives in the ReplayMix, not twice"
    saved_ppo = state["learner"]["optimizer"]["state"]
    for index, moments in later.trainer.learner.optimizer.state_dict()["state"].items():
        assert all(torch.equal(v, saved_ppo[index][k]) for k, v in moments.items()), "PPO's optimizer kept"
    assert later.agent_steps == first.agent_steps
    row = later.iteration()
    assert row["replay_mix_steps"] == 4 and later.agent_steps == 2 * first.agent_steps
    later.checkpoint(continue_after=False)
    _closed(later)
    assert old.load_latest()["replay_mix"]["steps"] == 4  # type: ignore[index]
    # Switched off again, its copy steps' state is kept for the next time it is switched on.
    off = _run(settings(tmp_path, dataclasses.replace(MIX, enabled=False)), old, old.load_latest())
    assert off.replay_mix_state is not None, "kept to be saved again while off"
    off.iteration()
    off.checkpoint(continue_after=False)
    _closed(off)
    assert old.load_latest()["replay_mix"]["steps"] == 4  # type: ignore[index]


@NEEDS_MESHES
def test_a_value_warm_up_holds_the_copy_steps_too(tmp_path: Path, store: Path) -> None:
    config = settings(tmp_path)
    warm = dataclasses.replace(config.pretrain, value_warmup_steps=10**9)
    run = _run(dataclasses.replace(config, pretrain=warm), _folder(tmp_path, "warm"))
    run.from_replays = True  # as a bot copied from replays first, whose critic learns alone at first
    policy = {n: p.detach().clone() for n, p in run.model.named_parameters() if not n.startswith("value.")}
    row = run.iteration()
    _closed(run)
    assert row["replay_mix_steps"] == 0 and row["replay_mix_loss"] == ""
    for name, weights in policy.items():
        assert torch.equal(weights, dict(run.model.named_parameters())[name]), f"{name} moved"


def test_the_dashboard_shows_the_block_and_a_trained_bot_can_switch_it_on(tmp_path: Path) -> None:
    from nitro.dashboard import describe
    from nitro.dashboard.api import Api
    from test_dashboard import TINY, items, make_bot

    config = parse_toml(TINY, NitroConfig, file=tmp_path / "t.toml", base_dir=tmp_path)
    sections = describe.describe(config, TINY, locked=True)
    block = next(s for s in sections if s["path"] == "replay_mix")
    assert block["label"] and block["help"] and not block["note"], "it may change while the bot trains"
    found = items([block])
    assert set(found) == {f"replay_mix.{f.name}" for f in dataclasses.fields(ReplayMixConfig)}
    assert not any(item["fixed"] for item in found.values()) and all(item["more"] for item in found.values())
    assert found["replay_mix.enabled"]["label"] == "Keep copying replays while training"
    runs = tmp_path / "runs"
    folder = make_bot(runs, "trained", saves=2)
    api = Api(Path(__file__).parents[1], runs)
    assert api.stage("trained", "replay_mix.enabled", True) == {"pending": {"replay_mix.enabled": True}}
    assert api.stage("trained", "replay_mix.lr", "0.00005")["pending"]["replay_mix.lr"] == 5e-5
    assert api.apply("trained") == {"applied": 2, "restarting": False}
    written = folder.load_settings().replay_mix
    assert written.enabled and written.lr == 5e-5 and written.share == 0.2


def test_out_of_memory_halves_the_copy_steps_keeps_it_in_the_saves_and_never_stops_the_run(
    tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    cpu = torch.device("cpu")
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, cpu, fp16=False)
    step = mix._step
    sizes: list[int] = []
    failures = iter([True, False, False, False])

    def short_of_memory(obs: torch.Tensor, *rest: torch.Tensor) -> Any:
        sizes.append(len(obs))
        if next(failures, False):
            model.logits(obs).sum().backward()  # type: ignore[no-untyped-call]  # a gradient left half-made
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return step(obs, *rest)

    monkeypatch.setattr(mix, "_step", short_of_memory)
    try:
        done = mix.learn(update(3))
        assert done is not None and done.steps == 3, "the step cut short is taken again"
        assert sizes == [64, 32, 32, 32] and mix.minibatch == 32
        assert len(mix.notices) == 1 and "replay_mix.minibatch 64 -> 32" in mix.notices[0]
        assert all(p.grad is None for p in model.parameters()), "no gradient is left for PPO's step"
        saved = mix.state_dict()
        assert ReplayMix(config, model, OBS, cpu, fp16=False, state=saved).minibatch == 32, "kept"
        chosen = dataclasses.replace(config, replay_mix=dataclasses.replace(MIX, minibatch=48))
        assert ReplayMix(chosen, model, OBS, cpu, fp16=False, state=saved).minibatch == 48, "a new choice"
        monkeypatch.setattr(mix, "_step", lambda *_: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError()))
        stopped = mix.learn(update(3))
        assert stopped is not None and stopped.steps == 0 and mix.off, "below 16 decisions it stops copying"
        assert "training goes on without copying replays" in mix.notices[-1]
        assert mix.learn(update(3)) is None
    finally:
        mix.close()


def test_replays_of_a_size_that_cannot_fit_are_never_made_ready_and_failed_ones_once(
    tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nitro import replay_mix

    for name in ("t0", "t1", "t2", "t3", "t4"):  # in a 2v2 folder: an unpadded 1v1 bot cannot use them
        target = replays.stored(name, 2)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(replays.stored("r0", 1).read_bytes())
    replays.stored("broken", 1).write_bytes(b"not a replay")
    made: list[str] = []
    prepare = replay_mix._prepare

    def counted(job: replays.Job) -> replays.Prepared:
        made.append(job.kept.stem)
        return prepare(job)

    monkeypatch.setattr(replay_mix, "_prepare", counted)
    config = settings(tmp_path, dataclasses.replace(MIX, replays=(store,)))
    model = build_networks(config, OBS, _core.num_actions())
    mix = ReplayMix(config, model, OBS, torch.device("cpu"), fp16=False)
    try:
        for _ in range(30):
            assert mix.learn(update(4)) is not None
        assert len(mix.files) == 9 and mix.next > 3 * 9, "several cycles of the 9 files"
        assert not any(stem.startswith("t") for stem in made), "2v2 replays are never made ready"
        assert sum(stem.startswith("broken") for stem in made) == 1, "a failed replay is made ready once"
        assert len(mix._failed) == 2 and {len(f) for f in mix._failed.values()} == {5, 1}
        assert any("2v2 replays, where the bot's observations fit 1v1" in why for why in mix._failed)
    finally:
        mix.close()


def _touched_store(tmp_path: Path) -> Path:
    """A folder of one converted 1v1 replay whose ball is first touched 60 rows after its kickoff;
    with the `store` fixture (no disk space needed)."""
    replicar_file(tmp_path / "touch.parquet")
    arrays, players = replays.read_replicar(tmp_path / "touch.parquet")
    arrays["ball_position"][60:, 0] = 100.0
    arrays["ball_velocity"][60:, 0] = 500.0
    lean_file(tmp_path / "touch.lean", arrays, players)
    folder = tmp_path / "kickoffs"
    assert replays.keep((tmp_path / "touch.lean").read_bytes(), "k0", 1, store=folder)
    return folder


def test_kickoff_only_seconds_keeps_only_the_kickoffs_and_off_changes_nothing(
    tmp_path: Path, store: Path
) -> None:
    from nitro import pretrain, replay_mix

    folder = _touched_store(tmp_path)
    off = settings(tmp_path, dataclasses.replace(MIX, replays=(folder,)))
    on = settings(tmp_path, dataclasses.replace(MIX, replays=(folder,), kickoff_only_seconds=0.25))
    assert replay_mix.sample_settings(off) == pretrain.copy_settings(off, replicar_rows=True), "as before"
    assert '"kickoff_only_seconds": 0.25' in replay_mix.sample_settings(on)
    file = replays.stored("k0", 1, store=folder)

    def made(config: NitroConfig) -> replays.Prepared:
        chosen = replay_mix.sample_settings(config)
        ready = replays.prepare(replays.Job(0, file, (), Path(), Path(), Path(), file, 1, chosen, 0.1))
        assert ready.usable, ready.problem
        return ready

    every, kickoff = made(off), made(on)
    # The window runs from the kickoff (the first row) to 0.25 s (30 ticks) after the touch at row 60: the
    # first decisions only, each the same sample as without the window.
    assert every.outside_window == 0 and 0 < kickoff.count < every.count
    assert kickoff.count + kickoff.outside_window == every.count
    for name in ("obs", "mask", "label_index", "label_share"):
        assert (getattr(kickoff, name) == getattr(every, name)[: kickoff.count]).all(), name
    # The copy steps take the window, and report what it kept of the replays taken in.
    model = build_networks(on, OBS, _core.num_actions())
    mix = ReplayMix(on, model, OBS, torch.device("cpu"), fp16=False)
    try:
        stats = mix.learn(update(4))
        assert stats is not None and stats.steps == 4
        assert stats.rows_kept is not None and stats.rows_left_out is not None
        times, rest = divmod(stats.rows_kept, kickoff.count)  # the one replay, taken in once a cycle
        assert times > 0 and rest == 0 and stats.rows_left_out == times * kickoff.outside_window
        row = replay_mix.columns(stats)
        assert row["replay_mix_rows_kept"] == stats.rows_kept
    finally:
        mix.close()
    assert set(replay_mix.columns(dataclasses.replace(stats, rows_kept=None))) == set(replay_mix.COLUMNS)
    report = {"agent_steps": 1, "replay_mix_steps": 4, "replay_mix_loss": 1.0, "replay_mix_agreement": 0.5}
    report |= {"replay_mix_rows_kept": 10, "replay_mix_rows_left_out": 30}
    lines = console.iteration_report(report, console.Facts("b", 1, 1))
    assert any("kickoffs only: 25% of the new replays' decisions kept" in line for line in lines)


def test_kickoff_only_seconds_is_checked(tmp_path: Path) -> None:
    assert NitroConfig().replay_mix.kickoff_only_seconds == 0.0, "off by default"
    for value, fine in (("3", True), ("0", True), ("30", True), ("-1", False), ("31", False)):
        text = SETTINGS + f"[replay_mix]\nenabled = true\nkickoff_only_seconds = {value}\n"
        if fine:
            parsed = parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
            assert parsed.replay_mix.kickoff_only_seconds == float(value)
        else:
            with pytest.raises(ConfigError, match="kickoff_only_seconds"):
                parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)


@NEEDS_MESHES
def test_a_run_copying_only_kickoffs_writes_what_the_window_kept(tmp_path: Path, store: Path) -> None:
    folder = _touched_store(tmp_path)
    mix = dataclasses.replace(MIX, replays=(folder,), kickoff_only_seconds=3.0)
    run = _run(settings(tmp_path, mix), _folder(tmp_path, "kickoffs"))
    try:
        run.iteration()
        row = run.iteration()
        assert {"replay_mix_rows_kept", "replay_mix_rows_left_out"} <= set(run.metrics.columns)
        assert "replay_mix_rows_kept" in row
    finally:
        _closed(run)
    plain = _run(settings(tmp_path, dataclasses.replace(MIX, replays=(folder,))), _folder(tmp_path, "all"))
    try:
        assert "replay_mix_rows_kept" not in plain.metrics.columns, "no window, no columns"
    finally:
        _closed(plain)
