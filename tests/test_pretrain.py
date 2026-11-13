"""`nitro pretrain`: recordings become samples in the core, a network learns to choose like
them, and the result starts a run and refuses to be resumed as one. `[pretrain]` in the settings
does it all when a run starts. Meshes are needed only for the runs at the end."""

from __future__ import annotations

import dataclasses
import os
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import __version__, _core, pretrain, replays
from nitro.cli import main
from nitro.config import load_config
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PretrainConfig, StartFromConfig
from nitro.envs import core_settings
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import RunFolder
from nitro.training import TrainingRun
from nitro.transfer import apply, find_source

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

PHYS = 332
CAR = 588
SETTINGS = '[run]\ndevice = "cpu"\n[model]\npolicy = [32]\nvalue = [32]\n'


def phys(frame: int, pos: tuple[float, float, float], vel: tuple[float, float, float] = (0, 0, 0)) -> bytes:
    body = struct.pack("<I3f9f3f3f", frame, *pos, 1, 0, 0, 0, 1, 0, 0, 0, 1, *vel, 0, 0, 0)
    return body + bytes(PHYS - len(body))


def car(frame: int, pos: tuple[float, float, float], steer: float, moving: bool) -> bytes:
    record = bytearray(CAR)
    record[:PHYS] = phys(frame, pos, (0, 900, 0) if moving else (0, 0, 0))
    record[332] = 1  # on the ground
    record[346] = 1  # a jump is available
    struct.pack_into("<f", record, 360, 0.33)  # boost, 0 to 1
    struct.pack_into("<5f", record, 368, 1.0, steer, 0.0, steer, 0.0)  # throttle, steer, pitch, yaw, roll
    return bytes(record)


def recording(path: Path, ticks: int = 400, mirrored: bool = False) -> None:
    """A 1v1 that starts at a kickoff: Blue drives straight, Orange steers left (right when mirrored)."""
    out = bytearray(struct.pack("<4sBIII6fI", b"RLPR", 0, 6, 28, 2, *([0.0] * 6), ticks))
    for t in range(ticks):
        moving = t > 10
        for x, y, steer in (
            (-2048.0, -2560.0 + 7.5 * t * moving, 0.0),
            (2048.0, 2560.0, 1.0 if mirrored else -1.0),
        ):
            out += struct.pack("<I", CAR) + car(t, (x, y, 17.0), steer, moving)
        out += struct.pack("<I", PHYS) + phys(t, (0.0, 0.0, 93.15))
    path.write_bytes(bytes(out))


def settings(tmp_path: Path) -> tuple[Path, NitroConfig]:
    file = tmp_path / "bc.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    return file, parse_toml(SETTINGS, NitroConfig, file=file, base_dir=tmp_path)


def test_a_recording_becomes_samples_labelled_with_the_controls(tmp_path: Path) -> None:
    _, config = settings(tmp_path)
    recording(tmp_path / "a.rlpr")
    samples = pretrain.load([tmp_path / "a.rlpr"], config)
    obs_size, *_ = _core.observation_layout(core_settings(config))
    assert samples.obs.shape == (len(samples), obs_size) and len(samples) % 2 == 0
    steer = torch.tensor([_core.action_controls(int(a))[1] for a in samples.label.argmax(1)])
    assert torch.equal(steer.reshape(-1, 2), torch.tensor([[0.0, -1.0]]).expand(len(samples) // 2, 2))
    assert torch.allclose(samples.label.sum(1), torch.ones(len(samples)))
    with pytest.raises(pretrain.PretrainError, match="observations of"):
        bigger = parse_toml(
            SETTINGS + "[env]\nteam_size = 2\n", NitroConfig, file=Path("x"), base_dir=tmp_path
        )
        pretrain.load([tmp_path / "a.rlpr"], bigger)


def test_cloning_learns_the_choices_and_starts_a_run(tmp_path: Path) -> None:
    file, config = settings(tmp_path)
    for name, mirrored in (("a", False), ("b", False), ("c", True)):
        recording(tmp_path / f"{name}.rlpr", mirrored=mirrored)
    samples = pretrain.load(pretrain.recordings([tmp_path]), config)
    train_set, held = pretrain.split(samples, 0.34, seed=1)
    assert set(train_set.recording.tolist()).isdisjoint(held.recording.tolist()), "whole recordings held back"
    model, history = pretrain.train(
        config,
        train_set,
        held,
        epochs=40,
        batch=64,
        lr=3e-3,
        device=torch.device("cpu"),
        seed=0,
    )
    assert history[-1].train_loss < history[0].train_loss
    _, agreement = pretrain.measure(model, train_set, torch.device("cpu"))
    assert agreement > 0.95

    folder = RunFolder.create(tmp_path / "runs", "cloned", file)
    pretrain.save(folder, model, config, {"samples": len(samples)})
    source = find_source(StartFromConfig(run="cloned"), tmp_path / "runs")
    assert source is not None and source.state["pretrained"] == {"samples": len(samples)}
    assert source.state["nitro_version"] == __version__
    rl = parse_toml(
        SETTINGS + '[start_from]\nrun = "cloned"\nparts = "policy"\n',
        NitroConfig,
        file=file,
        base_dir=tmp_path,
    )
    fresh = Learner(
        ActorCritic(samples.obs.shape[1], _core.num_actions(), rl.model),
        rl.ppo,
        0.99,
        torch.device("cpu"),
        False,
    )
    apply(source, fresh, rl)
    with torch.no_grad():
        assert torch.equal(fresh.model.logits(held.obs), model.logits(held.obs))
    assert main(["resume", "cloned", "--runs", str(tmp_path / "runs")]) == 1, (
        "a cloned bot is not a run to resume"
    )


def test_nothing_to_learn_from_is_said_plainly(tmp_path: Path) -> None:
    with pytest.raises(pretrain.PretrainError, match="no recordings"):
        pretrain.recordings([tmp_path])
    with pytest.raises(pretrain.PretrainError, match="does not exist"):
        pretrain.recordings([tmp_path / "missing.rlpr"])


def test_a_recording_that_does_not_fit_is_left_out_and_never_counted(tmp_path: Path) -> None:
    """The left-out file must not shift which samples belong to which recording."""
    _, config = settings(tmp_path)
    recording(tmp_path / "a.rlpr")
    recording(tmp_path / "b.rlpr")
    (tmp_path / "broken.rlpr").write_bytes(b"RLPR" + bytes(60))
    notes: list[str] = []
    samples = pretrain.load(pretrain.recordings([tmp_path]), config, notes=notes)
    assert [f.name for f in samples.files] == ["a.rlpr", "b.rlpr"] and len(notes) == 1
    train_set, held = pretrain.split(samples, 0.5, seed=3)
    assert len(train_set) and len(held), "one recording learns, one is held back"
    everything, nothing = pretrain.split(samples, 0.0, seed=3)
    assert len(nothing) == 0 and len(everything) == len(samples)
    with pytest.raises(pretrain.PretrainError, match="nothing left to learn"):
        pretrain.train(
            config,
            nothing,
            everything,
            epochs=1,
            batch=8,
            lr=1e-3,
            device=torch.device("cpu"),
            seed=0,
        )


RUN = (
    SETTINGS
    + """[env]
num_arenas = 4
threads = 2
[ppo]
steps_per_iteration = 160
minibatches = 2
[pretrain]
enabled = true
passes = 3
value_warmup_steps = 1_000_000_000
"""
)


def run_settings(tmp_path: Path, extra: str = "") -> Path:
    file = tmp_path / "run.toml"
    file.write_text(f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n' + RUN + extra, "utf-8")
    return file


def fake_conversion(tmp_path: Path, calls: list[tuple[Any, ...]], found: int = 5) -> Callable[..., Any]:
    """Stands in for converting replays (tests/test_replays.py tests the real one): three 1v1 recordings."""

    def convert(
        sources: list[Path], wanted: int, limit: int, meshes: Path, out: Path, say: Any = None
    ) -> replays.Report:
        calls.append((sources, wanted, limit, meshes))
        folder = tmp_path / "converted"
        folder.mkdir(exist_ok=True)
        files = [folder / f"{name}.rlpr" for name in "abc"][: min(found, 3)]
        for file in files:
            recording(file)
        return replays.Report(looked_in=sources, found=found, kept=len(files), recordings=files)

    return convert


def test_pretrain_settings_check_what_they_cannot_go_with(tmp_path: Path) -> None:
    file = run_settings(tmp_path)
    config = load_config(file)
    assert config.pretrain == PretrainConfig(enabled=True, passes=3, value_warmup_steps=1_000_000_000)
    assert config.pretrain.replays == (), "left out: the game's own folder"
    named = load_config(run_settings(tmp_path, 'replays = "my replays"\n'))
    assert named.pretrain.replays == ((tmp_path / "my replays").resolve(),), "relative to the settings file"
    absolute = (tmp_path / "elsewhere").resolve()  # an absolute folder on any system (a drive on Windows)
    several = load_config(run_settings(tmp_path, f'replays = ["a", "{absolute.as_posix()}"]\n'))
    assert several.pretrain.replays == ((tmp_path / "a").resolve(), absolute), "or a list of folders"

    def problems(**changes: Any) -> list[str]:
        return [name for name, _ in dataclasses.replace(config, **changes).problems()]

    assert problems() == []
    assert problems(start_from=StartFromConfig(run="other")) == ["pretrain.enabled"]
    assert problems(model=ModelConfig(shared=(32,))) == []  # The warm-up holds a shared trunk too
    assert main(["train", str(file), "--from", "other", "--runs", str(tmp_path / "runs")]) == 1
    assert not (tmp_path / "runs").exists(), "refused before a run folder is made"


def test_no_replays_or_none_of_the_size_is_said_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(run_settings(tmp_path))
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "run.toml")
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(replays, "convert_replays", fake_conversion(tmp_path, calls, found=0))
    with pytest.raises(pretrain.PretrainError, match=r"no replays .* save some in the game"):
        pretrain.start_source(config, folder, print)
    assert calls[0][1:3] == (1, 30) and calls[0][0] == replays.game_folders(replays.documents())

    def no_1v1(sources: list[Path], *args: Any, **kwargs: Any) -> replays.Report:
        return replays.Report(looked_in=sources, found=7, failed=["x"])

    monkeypatch.setattr(replays, "convert_replays", no_1v1)
    with pytest.raises(pretrain.PretrainError, match=r"none of the 7 replays .* is a 1v1 game; 1 could not"):
        pretrain.start_source(config, folder, print)

    def cut_short(sources: list[Path], *args: Any, **kwargs: Any) -> replays.Report:
        return replays.Report(looked_in=sources, found=700, converted=150, stopped_early=True)

    monkeypatch.setattr(replays, "convert_replays", cut_short)
    with pytest.raises(pretrain.PretrainError, match=r"no 1v1 game yet .* stopped after converting 150"):
        pretrain.start_source(config, folder, print)
    assert (
        pretrain.start_source(dataclasses.replace(config, pretrain=PretrainConfig()), folder, print) is None
    )


@needs_meshes
def test_pretrain_copies_the_replays_once_and_the_run_starts_from_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`nitro train` with [pretrain]: the replays are converted and copied, the copy is kept in the
    run folder and saved as the run's first save, at 0 steps, without training; the next start trains
    from its policy and holds it during the value warm-up, a resumed run knows where it started, and a restart
    before the first save reuses the copy."""
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(replays, "convert_replays", fake_conversion(tmp_path, calls))
    file, runs = run_settings(tmp_path), tmp_path / "runs"
    assert main(["train", str(file), "--name", "copied", "--runs", str(runs), "--steps", "1"]) == 0
    printed = capsys.readouterr().out
    assert "3 1v1 replays ready (of 5" in printed and "  pass 3: first choice agrees" in printed
    assert "Training has not started" in printed and "Training 'copied'" not in printed
    assert len(calls) == 1 and calls[0][3] == MESHES.resolve()
    folder = RunFolder(runs / "copied")
    assert [p.name for p in folder.complete_checkpoints()] == [f"{0:012d}"], "the copy alone, at 0 steps"
    copy = torch.load(folder.root / pretrain.COPY / "state.pt", weights_only=True)
    assert copy["pretrained"]["replays"] == 3 and len(copy["pretrained"]["epochs"]) == 3
    assert main(["resume", "copied", "--runs", str(runs), "--steps", "1"]) == 0
    printed = capsys.readouterr().out
    assert "Training 'copied'" in printed and len(calls) == 1, "the next start trains, copying nothing"
    state = folder.load_latest()
    assert state is not None and state["agent_steps"] > 0
    assert (
        state is not None and state["started_from"] == pretrain.COPY_LABEL and state["started_from_replays"]
    )
    policy = {k: v for k, v in state["learner"]["model"].items() if k.startswith("policy.")}
    assert all(torch.equal(v, copy["learner"]["model"][k]) for k, v in policy.items()), "held in the warm-up"

    config = folder.load_settings(state)
    resumed = TrainingRun(config, folder, state)
    resumed.iteration()
    resumed.metrics.close()
    assert resumed.from_replays and resumed.trainer.learner.policy_frozen
    assert len(calls) == 1

    # A restart before the first save starts from the same copy without converting or copying again; a
    # changed [pretrain] makes it again.
    fresh = RunFolder.create(runs, "restarted", file)
    (fresh.root / pretrain.COPY).mkdir()
    (fresh.root / pretrain.COPY / "state.pt").write_bytes(
        (folder.root / pretrain.COPY / "state.pt").read_bytes()
    )
    again = TrainingRun(config, fresh)
    again.metrics.close()
    assert len(calls) == 1 and again.started_from == pretrain.COPY_LABEL
    heard: list[str] = []
    changed = dataclasses.replace(config, pretrain=dataclasses.replace(config.pretrain, passes=1))
    pretrain.start_source(changed, fresh, heard.append)
    assert len(calls) == 2 and heard[0].startswith("The [pretrain] settings or the networks changed")
    slower = dataclasses.replace(changed, env=dataclasses.replace(changed.env, tick_skip=4, action_delay=3))
    pretrain.start_source(slower, fresh, heard.append)
    assert len(calls) == 3, "the labels depend on the tick skip, so the copy is made again"
    said = len(heard)
    pretrain.start_source(slower, fresh, heard.append)
    assert len(calls) == 3 and len(heard) == said, "the same settings: the copy is kept, quietly"


def test_the_copys_core_settings_carry_what_a_recordings_rows_hold(tmp_path: Path) -> None:
    """The copy's core settings are the run's game and actions plus what a recording's rows hold; the
    label and sample rules are fixed, so no setting chooses them."""
    import json

    def config_of(extra: str = "") -> NitroConfig:
        text = '[run]\ndevice = "cpu"\n[pretrain]\nenabled = true\n' + extra
        return parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)

    config = config_of()
    core = json.loads(core_settings(config))
    for rows in (False, True):
        settings = json.loads(pretrain.copy_settings(config, replicar_rows=rows))
        assert settings == {**core, "replicar_rows": rows}
    for extra in ('label_weights = "weighted"\n', 'samples = "corrected"\n'):
        key = extra.split(" ")[0]
        with pytest.raises(ConfigError) as refused:
            config_of(extra)
        assert f"'{key}' is not a known setting" in str(refused.value), str(refused.value)


def test_load_reads_a_recordings_rows_by_where_it_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In the project's replays folder, where only the converters write, a recording's rows hold the
    controls applied next (replicar's), also in a folder there linked to another disk; elsewhere the controls
    that produced each tick (the bot's own and RocketSim's recordings); `rows` (nitro pretrain --rows) says
    otherwise. nitro pretrain says and notes both, and how many it read each way."""
    import json

    file, config = settings(tmp_path)
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    inside, outside = tmp_path / "replays" / "1v1" / "a.rlpr", tmp_path / "b.rlpr"
    inside.parent.mkdir(parents=True)
    for path in (inside, outside):
        recording(path)
    assert pretrain.made_from_replays(inside) and not pretrain.made_from_replays(outside)
    seen: list[bool] = []
    demonstrations = _core.demonstrations

    def spy(path: str, settings_json: str, softness: float) -> Any:
        seen.append(json.loads(settings_json)["replicar_rows"])
        return demonstrations(path, settings_json, softness)

    monkeypatch.setattr(_core, "demonstrations", spy)
    for rows, read in (("auto", [True, False]), ("replicar", [True, True]), ("recorded", [False, False])):
        seen.clear()
        loaded = pretrain.load([inside, outside], config, rows=rows)
        assert seen == read, rows
        assert loaded.read_as == {"replicar": read.count(True), "recorded": read.count(False)}
    assert main(["pretrain", str(file), str(outside), "--name", "c", "--runs", str(tmp_path / "runs"),
                 "--epochs", "1", "--rows", "replicar"]) == 0  # fmt: skip
    assert seen[-1] is True
    details = json.loads((tmp_path / "runs" / "c" / "pretrain.json").read_text(encoding="utf-8"))
    assert details["rows"] == "replicar"
    assert details["read_as"] == {"replicar": 1, "recorded": 0}
    assert main(["pretrain", str(file), str(inside), str(outside), "--name", "d", "--runs",
                 str(tmp_path / "runs"), "--epochs", "1"]) == 0  # fmt: skip
    said = capsys.readouterr().out
    assert "Read 1 recording(s) as made from replays with replicar and 1 as recorded" in said
    details = json.loads((tmp_path / "runs" / "d" / "pretrain.json").read_text(encoding="utf-8"))
    assert (details["rows"], details["read_as"]) == ("auto", {"replicar": 1, "recorded": 1})

    # A folder of the replays folder linked to another disk: listed under it, so replicar's.
    elsewhere = tmp_path / "big disk" / "2v2"
    elsewhere.mkdir(parents=True)
    try:
        (tmp_path / "replays" / "2v2").symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("links cannot be made here")
    recording(elsewhere / "c.rlpr")
    assert pretrain.made_from_replays(tmp_path / "replays" / "2v2" / "c.rlpr")
    assert not pretrain.made_from_replays(elsewhere / "c.rlpr")
    assert not pretrain.made_from_replays(tmp_path / "replays" / ".." / "b.rlpr")
