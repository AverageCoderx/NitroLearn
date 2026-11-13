"""Short real training runs on the Rocket League environments: determinism, exact resume, no splicing."""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import training
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig, StartFromConfig
from nitro.runs import RunFolder
from nitro.surgery import widen_run, widened_config
from nitro.training import (
    EVEN_START_COLUMNS,
    KICKOFF_COLUMNS,
    TIME_COLUMNS,
    TrainingRun,
    build,
    configure_determinism,
    memory_note,
)

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
pytestmark = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

SETTINGS = """
[run]
seed = 7
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
[[state_setters]]
name = "random"
"""

NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
# (device, precision): the eager CPU path and both CUDA-graph paths.
PATHS = [
    pytest.param("cpu", "fp32", id="cpu-fp32"),
    pytest.param("cuda", "fp32", id="cuda-fp32", marks=NEEDS_GPU),
    pytest.param("cuda", "fp16", id="cuda-fp16", marks=NEEDS_GPU),
]


@pytest.fixture(autouse=True)
def restore_determinism() -> Iterator[None]:
    yield
    configure_determinism(False)


def config(device: str, seed: int = 7, precision: str = "fp32") -> NitroConfig:
    parsed = parse_toml(SETTINGS, NitroConfig, file=Path("test.toml"), base_dir=Path.cwd())
    run = dataclasses.replace(parsed.run, device=device, seed=seed, precision=precision)  # type: ignore[arg-type]
    return dataclasses.replace(
        parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES), run=run
    )


def run(device: str, seed: int = 7, precision: str = "fp32") -> list[dict[str, torch.Tensor]]:
    trainer = build(config(device, seed, precision)).trainer
    snapshots = []
    for _ in range(2):
        report = trainer.iteration()
        r = trainer.rollout
        snapshots.append(
            {
                "obs": r.obs.cpu().clone(),
                "actions": r.actions.cpu().clone(),
                "log_probs": r.log_probs.cpu().clone(),
                "rewards": r.rewards.cpu().clone(),
                "losses": torch.tensor(
                    [report.learn.policy_loss, report.learn.value_loss, report.learn.entropy]
                ),
            }
        )
    return snapshots


@pytest.mark.parametrize(("device", "precision"), PATHS)
def test_deterministic_mode_repeats_runs_exactly(device: str, precision: str) -> None:
    first = run(device, precision=precision)
    second = run(device, precision=precision)
    for a, b in zip(first, second, strict=True):
        for key in a:
            assert torch.equal(a[key], b[key]), f"{key} differs between identical runs"
    other = run(device, seed=8, precision=precision)
    assert not torch.equal(first[0]["obs"], other[0]["obs"]), "a different seed must give different games"


WALL_CLOCK = ("steps_per_second", *TIME_COLUMNS)  # timings differ from run to run


@pytest.mark.parametrize(("device", "precision"), PATHS)
def test_a_resumed_run_continues_exactly_as_if_it_never_stopped(
    tmp_path: Path, device: str, precision: str
) -> None:
    settings, file = config(device, precision=precision), tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")

    straight = TrainingRun(settings, RunFolder.create(tmp_path, "straight", file))
    for _ in range(2):
        straight.iteration()
    straight.checkpoint()
    assert straight.warming_up, "the first iteration after a save is not judged for speed"
    for _ in range(2):
        straight.iteration()
    assert not straight.warming_up
    straight.metrics.close()

    folder = RunFolder.create(tmp_path, "stopped", file)
    stopped = TrainingRun(settings, folder)
    for _ in range(2):
        stopped.iteration()
    stopped.checkpoint(continue_after=False)
    stopped.iteration()  # lost: the run is killed before its next save
    stopped.metrics.close()
    resumed = TrainingRun(settings, folder, folder.load_latest())
    for _ in range(2):
        resumed.iteration()
    resumed.metrics.close()

    for name, weights in straight.model.state_dict().items():
        assert torch.equal(weights, resumed.model.state_dict()[name]), f"{name} differs after resuming"

    def rows(run: RunFolder) -> list[dict[str, str]]:
        with run.metrics.open(encoding="utf-8") as metrics:
            return [{k: v for k, v in row.items() if k not in WALL_CLOCK} for row in csv.DictReader(metrics)]

    assert rows(straight.folder) == rows(folder)
    assert set(KICKOFF_COLUMNS) <= set(rows(folder)[0]), "a kickoff setter is listed"


def test_each_rewards_pay_is_logged_and_a_switched_off_reward_is_not(tmp_path: Path) -> None:
    """metrics.csv has what each reward paid (after zero-sum and weight) and its size, for the
    dashboard's reward shares; they add up to the reward per step. A reward switched off has no column."""
    text = SETTINGS + (
        '[[rewards]]\nname = "touch_ball"\nweight = 5\n'
        '[[rewards]]\nname = "velocity_ball_to_goal"\nweight = 2\nzero_sum = { team_spirit = 1 }\n'
        '[[rewards]]\nname = "air"\nweight = 3\nenabled = false\n'
        '[[rewards]]\nname = "velocity_player_to_ball"\nweight = 1\n'
    )
    file = tmp_path / "settings.toml"
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, "paid", file)
    training = TrainingRun(settings, folder)
    row = training.iteration()
    training.metrics.close()
    labels = ["touch_ball", "velocity_ball_to_goal", "velocity_player_to_ball"]
    assert not {"raw_air", "paid_air", "paid_abs_air"} & set(row)
    assert row["paid_touch_ball"] == pytest.approx(5 * row["raw_touch_ball"])
    assert row["paid_velocity_player_to_ball"] == pytest.approx(row["raw_velocity_player_to_ball"])
    assert sum(row[f"paid_{label}"] for label in labels) == pytest.approx(row["reward_per_step"])
    for label in labels:
        assert row[f"paid_abs_{label}"] >= abs(row[f"paid_{label}"]) - 1e-9
    # Zero-sum in 1v1: what one car gains the other loses, so the pay cancels while its size does not.
    assert abs(row["paid_velocity_ball_to_goal"]) < 1e-6 < row["paid_abs_velocity_ball_to_goal"]
    # Before zero-sum: what each car earned itself, by size.
    assert row["own_abs_velocity_player_to_ball"] == pytest.approx(row["paid_abs_velocity_player_to_ball"])
    assert row["own_abs_velocity_ball_to_goal"] > 0
    # How often each fired before its weight: a share of the steps; speed to the ball nearly always.
    assert all(0 <= row[f"fired_{label}"] <= 1 for label in labels)
    assert (row["fired_touch_ball"] > 0) == (row["raw_touch_ball"] > 0)
    assert row["fired_velocity_player_to_ball"] > 0.9
    with folder.metrics.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert {f"paid_{label}" for label in labels} | {f"paid_abs_{label}" for label in labels} <= set(header)


def test_a_reward_of_ones_own_trains_like_a_built_in_one(tmp_path: Path) -> None:
    """The example reward compiled in from Rust is paid and logged like any other."""
    text = SETTINGS + (
        '[[rewards]]\nname = "example_ball_height"\nweight = 2\npower = 2\n'
        '[[rewards]]\nname = "touch_ball"\nweight = 1\n'
    )
    file = tmp_path / "settings.toml"
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    training = TrainingRun(settings, RunFolder.create(tmp_path, "own", file))
    row = training.iteration()
    training.metrics.close()
    assert 0 < row["raw_example_ball_height"] < 1, "the ball is above the ground and below the ceiling"
    assert row["paid_example_ball_height"] == pytest.approx(2 * row["raw_example_ball_height"])


def test_a_padded_run_moves_from_1v1_to_3v3_on_a_resume(tmp_path: Path) -> None:
    """With env.obs_team_size = 3 the observation keeps its size, so a 1v1 run resumes as 3v3 with the
    same networks and trains on."""
    text = SETTINGS.replace("[env]\n", "[env]\nobs_team_size = 3\n", 1)
    file = tmp_path / "settings.toml"
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    folder = RunFolder.create(tmp_path, "padded", file)
    first = TrainingRun(settings, folder)
    first.iteration()
    first.checkpoint(continue_after=False)
    first.metrics.close()
    size = first.envs.groups[0].obs.shape[1]
    # The run's own copy of the settings is what a resume reads.
    folder.settings.write_text(text.replace("[env]\n", "[env]\nteam_size = 3\n", 1), encoding="utf-8")
    state = folder.load_latest()
    bigger = folder.load_settings(state)
    assert bigger.env.team_size == 3
    bigger = dataclasses.replace(bigger, paths=dataclasses.replace(bigger.paths, collision_meshes=MESHES))
    resumed = TrainingRun(bigger, folder, state)
    assert resumed.envs.groups[0].obs.shape[1] == size
    assert resumed.envs.groups[0].obs.shape[0] == 3 * first.envs.groups[0].obs.shape[0]  # 6 cars, not 2
    row = resumed.iteration()
    assert all(torch.isfinite(torch.tensor(float(row[key]))) for key in ("entropy", "value_loss"))
    resumed.metrics.close()


def _settings(text: str, file: Path) -> NitroConfig:
    file.write_text(text, encoding="utf-8")
    parsed = parse_toml(text, NitroConfig, file=file, base_dir=file.parent)
    return dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))


def _meshes(config: NitroConfig) -> NitroConfig:
    return dataclasses.replace(config, paths=dataclasses.replace(config.paths, collision_meshes=MESHES))


def _saved_run(tmp_path: Path, name: str) -> tuple[RunFolder, TrainingRun]:
    settings = _settings(SETTINGS, tmp_path / f"{name}.toml")
    folder = RunFolder.create(tmp_path, name, tmp_path / f"{name}.toml")
    run = TrainingRun(settings, folder)
    run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    return folder, run


def test_a_resumed_runs_log_numbers_its_iterations_as_its_metrics_do(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """train.log goes on from the save's iteration count, as metrics.csv and the console report do."""
    folder, first = _saved_run(tmp_path, "counted")
    loaded = folder.load_latest()
    assert loaded is not None
    resumed = TrainingRun(_meshes(folder.load_settings(loaded)), folder, loaded)
    caplog.clear()  # the first run's own lines, when an earlier test left the logger at INFO
    with caplog.at_level(logging.INFO, logger="nitro"):
        row = resumed.iteration()
    resumed.metrics.close()
    assert row["iteration"] == first.iterations + 1 == 2
    assert "iteration 2:" in caplog.text and "iteration 1:" not in caplog.text, caplog.text


def test_a_save_with_another_number_of_actions_is_refused_plainly(tmp_path: Path) -> None:
    """A save whose policy chooses among more actions than the table has (made by another version) is
    refused with a plain message on a resume, as a run to start from, and for export."""
    from nitro.export import ExportError, export_run
    from nitro.transfer import TransferError

    folder, _ = _saved_run(tmp_path, "wide")
    newest = folder.complete_checkpoints()[-1] / "state.pt"
    state = torch.load(newest, map_location="cpu", weights_only=True)
    model = state["learner"]["model"]
    last = max(int(k.split(".")[1]) for k in model if k.startswith("policy.") and k.endswith(".weight"))
    for part in ("weight", "bias"):
        key = f"policy.{last}.{part}"
        model[key] = torch.cat([model[key], torch.zeros(120, *model[key].shape[1:])])
    torch.save(state, newest)
    said = "chooses among 210 actions, but this version of NitroLearn has 90: it was made by another version"
    whole = f"{said} and cannot be used here"

    loaded = folder.load_latest()
    assert loaded is not None
    with pytest.raises(ValueError, match=said) as resumed:
        TrainingRun(_meshes(folder.load_settings(loaded)), folder, loaded)
    with pytest.raises(ExportError, match=said) as exported:
        export_run(folder, tmp_path / "bot", None, build=False)
    start = StartFromConfig(run="wide", parts="policy")
    file = tmp_path / "from.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    settings = dataclasses.replace(_settings(SETTINGS, file), start_from=start)
    with pytest.raises(TransferError, match=said) as started:
        TrainingRun(settings, RunFolder.create(tmp_path, "from", file))
    # Plain words only: the whole message, nothing about how such a save was made.
    assert str(resumed.value) == f"This save {whole}."
    assert str(exported.value) == f"'wide' {whole}"
    assert str(started.value).startswith("start_from: wide at ") and str(started.value).endswith(whole)


def test_a_new_run_starts_from_another_runs_policy_and_records_where_it_started(tmp_path: Path) -> None:
    """A run started from another run's save takes its policy and records where it started; the
    source run is left as it was."""
    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    first = TrainingRun(config("cpu"), RunFolder.create(tmp_path, "first", file))
    first.iteration()
    first.checkpoint(continue_after=False)
    first.metrics.close()
    trained = {k: v.clone() for k, v in first.model.state_dict().items()}
    saves = sorted((tmp_path / "first" / "checkpoints").iterdir())

    settings = dataclasses.replace(config("cpu"), start_from=StartFromConfig(run="first", parts="policy"))
    second = TrainingRun(settings, RunFolder.create(tmp_path, "second", file))
    assert second.started_from == f"first at {first.agent_steps:,} agent steps"
    policy = {k: v for k, v in second.model.state_dict().items() if k.startswith("policy.")}
    assert all(torch.equal(v.cpu(), trained[k].cpu()) for k, v in policy.items())
    second.iteration()
    path = second.checkpoint(continue_after=False)
    second.metrics.close()
    saved = torch.load(path / "state.pt", weights_only=True)
    assert saved["started_from"] == second.started_from
    resumed = TrainingRun(settings, RunFolder(tmp_path / "second"), saved)
    row = resumed.iteration()
    resumed.metrics.close()
    assert float(row["entropy"]) > 0
    assert sorted((tmp_path / "first" / "checkpoints").iterdir()) == saves, "the source is left as it was"


def test_a_run_started_from_one_at_another_tick_skip_says_what_changes_meaning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The save's own settings give its tick skip; a run at 4 started from one at 8 logs lambda's
    same-seconds value, one at the same tick skip does not."""
    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    eight = config("cpu")
    eight = dataclasses.replace(eight, env=dataclasses.replace(eight.env, tick_skip=8))
    first = TrainingRun(eight, RunFolder.create(tmp_path, "first", file))
    first.iteration()
    first.checkpoint(continue_after=False)
    first.metrics.close()
    settings = dataclasses.replace(eight, start_from=StartFromConfig(run="first"))
    quicker = dataclasses.replace(settings, env=dataclasses.replace(settings.env, tick_skip=4))
    with caplog.at_level(logging.WARNING, logger="nitro"):
        started = TrainingRun(quicker, RunFolder.create(tmp_path, "quicker", file))
        started.metrics.close()
        assert "the same seconds is 0.9747" in caplog.text
        trained = first.model.state_dict()
        assert all(torch.equal(v.cpu(), trained[k].cpu()) for k, v in started.model.state_dict().items())
        assert "ppo.steps_per_iteration 320" in caplog.text
        caplog.clear()
        TrainingRun(settings, RunFolder.create(tmp_path, "same", file)).metrics.close()
        assert "the same seconds" not in caplog.text


@pytest.mark.parametrize(("device", "precision"), PATHS)
def test_a_widened_run_resumes_and_trains(tmp_path: Path, device: str, precision: str) -> None:
    """`nitro surgery` then `nitro resume`: the widened save loads into a real run on this device,
    which plays as the old one did and trains on, with Adam's fused kernel on a graphics card."""
    settings, file = config(device, precision=precision), tmp_path / "settings.toml"
    # The run's own settings file says how it trains, as a real one does (the surgery checks it).
    file.write_text(
        SETTINGS.replace("[run]\n", f'[run]\ndevice = "{device}"\nprecision = "{precision}"\n', 1),
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path, "small", file)
    original = TrainingRun(settings, folder)
    original.iteration()
    original.checkpoint(continue_after=False)
    original.metrics.close()

    report = widen_run(tmp_path, "small", 2)
    wide_folder = RunFolder(tmp_path / report.target)
    state = wide_folder.load_latest()
    assert state is not None
    wide_settings = dataclasses.replace(settings, model=widened_config(settings.model, 2))
    assert wide_folder.load_settings(state).model == wide_settings.model
    resumed = TrainingRun(wide_settings, wide_folder, state)
    obs = resumed.envs.groups[0].obs.to(resumed.trainer.device)
    with torch.no_grad():
        old_logits, wide_logits = original.model(obs)[0], resumed.model(obs)[0]
    torch.testing.assert_close(wide_logits, old_logits, atol=1e-4, rtol=1e-4)
    if device == "cuda":
        assert all(group["fused"] for group in resumed.trainer.learner.optimizer.param_groups)
    for _ in range(2):
        row = resumed.iteration()
        assert all(torch.isfinite(torch.tensor(float(row[key]))) for key in ("entropy", "value_loss"))
    resumed.metrics.close()
    assert resumed.agent_steps > original.agent_steps
    # The two copies of each unit have learnt apart (with equal shares they would stay identical for good).
    first = next(m for m in resumed.model.policy if isinstance(m, torch.nn.Linear)).weight.detach()
    copies = first.reshape(-1, 2, first.shape[1])
    assert float((copies[:, 0] - copies[:, 1]).abs().max()) > 0.0


def test_a_full_disk_stops_the_run_with_a_plain_message_and_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The save at the end cannot be written (no space left), and neither can the report after it: the
    run stops with one plain message naming the last good save, and leaves no half-written save."""
    import errno
    import signal
    import sys

    from nitro import cli, metrics

    settings, file = config("cpu"), tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path, "small", file)
    first = TrainingRun(settings, folder)
    first.iteration()
    good = first.checkpoint(continue_after=False)
    first.metrics.close()
    state = folder.load_latest()
    assert state is not None

    def full(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    closed: list[bool] = []
    close = metrics.MetricsLog.close

    def closing(log: metrics.MetricsLog) -> None:
        closed.append(True)
        close(log)

    monkeypatch.setattr(torch, "save", full)
    monkeypatch.setattr(metrics.MetricsLog, "write_report", full)
    monkeypatch.setattr(metrics.MetricsLog, "close", closing)
    monkeypatch.setattr(logging, "raiseExceptions", logging.raiseExceptions)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    logger = logging.getLogger("nitro")
    handlers, previous = list(logger.handlers), signal.getsignal(signal.SIGINT)
    try:
        code = cli._run_until_stopped(settings, folder, state, int(state["agent_steps"]) + 1, False)
    finally:
        for handler in logger.handlers[len(handlers) :]:
            logger.removeHandler(handler)
            handler.close()
        signal.signal(signal.SIGINT, previous)
    out, err = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in out + err and "metrics.py" not in out + err
    assert f"Training stopped: the disk is full (the disk that holds {folder.root})." in err
    assert (
        f"The last good save is at {int(state['agent_steps']):,} agent steps ({good}); free some space on "
        "that disk, then start again with: nitro resume small" in err
    )
    assert [p.name for p in folder.checkpoints.iterdir()] == [good.name], "no half-written save left"
    assert closed, "the run's files are closed although the report could not be written"


def test_a_change_applied_in_the_dashboard_is_said_as_applied_not_as_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """After the dashboard's Apply the run starts again with the changes the user just made: they are said
    as applied. Resumed any other way, a change since the save stays a warning."""
    settings, file = config("cpu"), tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path, "small", file)
    first = TrainingRun(settings, folder)
    first.iteration()
    first.checkpoint(continue_after=False)
    first.metrics.close()
    state = folder.load_latest()
    assert state is not None
    changed = dataclasses.replace(settings, ppo=dataclasses.replace(settings.ppo, clip_range=0.15))
    before = settings.ppo.clip_range
    capsys.readouterr()
    with caplog.at_level(logging.WARNING, logger="nitro"):
        TrainingRun(changed, folder, state, applied=True).metrics.close()
    assert f"Applied: ppo.clip_range changed from {before} to 0.15." in capsys.readouterr().out
    assert "changed since the save" not in caplog.text
    with caplog.at_level(logging.WARNING, logger="nitro"):
        TrainingRun(changed, folder, state).metrics.close()
    assert f"Setting changed since the save: ppo.clip_range: {before} -> 0.15" in caplog.text
    assert "Applied:" not in capsys.readouterr().out


def test_learner_never_sees_two_episodes_spliced_together() -> None:
    settings = config("cpu")
    trainer = build(
        dataclasses.replace(settings, run=dataclasses.replace(settings.run, deterministic=False))
    ).trainer
    previous: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    endings = 0
    for _ in range(4):
        trainer.collect()
        r = trainer.rollout
        ended = r.terminal | r.truncated
        endings += int(ended.sum())
        if previous is not None:
            last_episode, last_step, last_ended = previous
            first = (r.episode[0], r.episode_step[0])
            # Across the iteration boundary: the same episode one step later, or a fresh episode.
            assert torch.equal(first[0] == last_episode, ~last_ended)
            assert torch.equal(first[1], torch.where(last_ended, 0, last_step + 1))
        same = r.episode[1:] == r.episode[:-1]
        assert torch.equal(same, ~ended[:-1]), "an episode changed without ending, or ended without changing"
        assert torch.equal(r.episode_step[1:], torch.where(ended[:-1], 0, r.episode_step[:-1] + 1))
        previous = (r.episode[-1].clone(), r.episode_step[-1].clone(), ended[-1].clone())
    assert endings > 10


def test_a_run_saves_every_checkpoint_steps_when_set_else_by_time() -> None:
    """run.checkpoint_steps: a save whenever a multiple of it is passed, time aside."""
    from types import SimpleNamespace

    from nitro.config.schema import RunConfig

    run = SimpleNamespace(config=SimpleNamespace(run=RunConfig(checkpoint_steps=200)), saved_at_steps=-1)
    for steps, since, due in ((150, 1e9, False), (205, 0.0, True)):
        run.agent_steps = steps
        assert TrainingRun._save_due(run, since) is due  # type: ignore[arg-type]
    run.saved_at_steps = 205
    for steps, due in ((390, False), (401, True)):
        run.agent_steps = steps
        assert TrainingRun._save_due(run, 0.0) is due  # type: ignore[arg-type]
    timed = SimpleNamespace(config=SimpleNamespace(run=RunConfig(checkpoint_minutes=1)), agent_steps=0)
    timed.saved_at_steps = -1
    assert TrainingRun._save_due(timed, 61.0) and not TrainingRun._save_due(timed, 59.0)  # type: ignore[arg-type]


def test_the_wall_clock_speed_counts_the_time_between_iterations(tmp_path: Path) -> None:
    """wall_steps_per_second is the agent steps since the previous row over the wall clock, so time
    spent between iterations lowers it (steps_per_second leaves that time out); the first row has none."""
    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    run = TrainingRun(config("cpu"), RunFolder.create(tmp_path, "wall", file))
    first = run.iteration()
    time.sleep(1.0)  # the run's own bookkeeping, a save, ...
    second = run.iteration()
    third = run.iteration()
    run.metrics.close()
    per_iteration = run.agent_steps // 3
    assert first["wall_steps_per_second"] == "" and "wall_steps_per_second" in TIME_COLUMNS
    assert 0 < second["wall_steps_per_second"] <= per_iteration / 1.0
    assert second["wall_steps_per_second"] < second["steps_per_second"]  # however fast the machine
    assert third["wall_steps_per_second"] > 0
    with run.folder.metrics.open(encoding="utf-8") as metrics:
        written = [row["wall_steps_per_second"] for row in csv.DictReader(metrics)]
    assert written == ["", str(second["wall_steps_per_second"]), str(third["wall_steps_per_second"])]


# Self-play on, past versions drawn in at saves, and scripted (uneven) starts. Random play scores
# almost only from saves, so two save blocks: one is counted as an even start below to fill every new column.
UNEVEN = """
[run]
seed = 7
deterministic = true
device = "cpu"
[env]
num_arenas = 8
threads = 4
[terminals]
no_touch_seconds = 4
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 1600
minibatches = 2
[selfplay]
enabled = true
past_share = 0.5
active_opponents = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "aerial_ball"
[[state_setters]]
name = "save"
weight = 2
[[state_setters]]
name = "save"
weight = 2
"""


def _same(a: Any, b: Any, where: str) -> None:
    """Equal to the bit: tensors by value and type, containers item by item, NaN equal to NaN."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for key in a:
            _same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), where


def test_the_even_start_columns_change_no_weight_rating_or_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same short run with the three new columns and without them (main's path: no columns,
    record() without `even`) saves the same to the bit (weights, optimizer, ratings, games, past versions),
    writes the same games.csv and the same rows, and its header only gains the three after matchmaking."""
    from nitro import training

    file = tmp_path / "settings.toml"
    file.write_text(UNEVEN, encoding="utf-8")
    parsed = parse_toml(UNEVEN, NitroConfig, file=file, base_dir=tmp_path)
    settings = dataclasses.replace(
        parsed,
        paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES),
        selfplay=dataclasses.replace(
            parsed.selfplay, snapshot_steps=1600
        ),  # every iteration; below the minimum
    )
    assert training.even_starts(settings.state_setters) == [True, False, False, False]

    def played(name: str, even: list[bool] | None) -> tuple[RunFolder, list[dict[str, Any]]]:
        monkeypatch.setattr(training, "even_starts", lambda setters: even)
        folder = RunFolder.create(tmp_path, name, file)
        run = TrainingRun(settings, folder)
        rows = []
        for number in range(6):
            rows.append(run.iteration())
            if number in (0, 2):
                run.checkpoint()  # past versions drawn in
        run.checkpoint(continue_after=False)
        run.metrics.close()
        run.selfplay.close()
        return folder, rows

    new, new_rows = played("columns", [True, False, True, False])
    old, old_rows = played("before", None)
    _same(new.load_latest(), old.load_latest(), "save")
    games = [(folder.root / "pool" / "games.csv").read_bytes() for folder in (new, old)]
    assert games[0] == games[1]
    pools = [sorted((folder.root / "pool").glob("*.pt")) for folder in (new, old)]
    assert [p.name for p in pools[0]] == [p.name for p in pools[1]] and pools[0]
    for a, b in zip(*pools, strict=True):
        _same(torch.load(a, weights_only=True), torch.load(b, weights_only=True), a.name)
    headers = []
    for folder in (new, old):
        with folder.metrics.open(encoding="utf-8") as metrics:
            headers.append(next(csv.reader(metrics)))
    at = headers[1].index("matchmaking") + 1
    assert headers[0] == headers[1][:at] + EVEN_START_COLUMNS + headers[1][at:]
    for number, (a, b) in enumerate(zip(new_rows, old_rows, strict=True)):
        kept = {k: v for k, v in a.items() if k not in EVEN_START_COLUMNS and k not in WALL_CLOCK}
        _same(kept, {k: v for k, v in b.items() if k not in WALL_CLOCK}, f"row {number}")
    # Not an empty check: games from even and uneven starts against past versions, and Blue's own edge.
    even_games = sum(row["pool_games_even"] for row in new_rows)
    assert 0 < even_games < games[0].count(b"\n") - 1
    assert any(row["pool_goal_share_even"] != "" for row in new_rows)
    assert any(row["start_edge"] != "" for row in new_rows)


def test_a_nearly_full_graphics_card_is_said_once_with_what_to_lower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gb = 2**30
    assert memory_note(7 * gb, 8 * gb) is None, "7 of 8 GB is below 90%"
    said = memory_note(int(7.6 * gb), 8 * gb)
    assert said is not None and "7.6 of 8.0 GB" in said
    for fix in ("env.num_arenas", "ppo.microbatch", "[model]", "Windows"):
        assert fix in said, fix
    assert memory_note(1, 0) is None, "no total known: nothing said"
    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    run = TrainingRun(config("cpu"), RunFolder.create(tmp_path, "full", file))
    assert training.card_memory(run.trainer.device) is None, "the processor has no card memory"
    asked: list[torch.device] = []

    def full_card(device: torch.device) -> tuple[int, int]:
        asked.append(device)
        return int(7.6 * gb), 8 * gb  # a card with 8 GB, as a too-big bot leaves it

    monkeypatch.setattr(training, "card_memory", full_card)
    with caplog.at_level(logging.WARNING, logger="nitro"):
        run.iteration()
        run.iteration()
    run.metrics.close()
    assert len(asked) == 1, "checked after the first iteration only"
    assert [r.message for r in caplog.records].count(said) == 1 and said in run.recent_warnings
