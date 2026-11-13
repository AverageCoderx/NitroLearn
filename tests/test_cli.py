from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from nitro import __version__
from nitro.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_check_settings_accepts_a_valid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = tmp_path / "bot.toml"
    file.write_text(f"[paths]\ncollision_meshes = {json.dumps(str(MESHES))}\n", encoding="utf-8")
    assert main(["check-settings", str(file)]) == 0
    assert "no problems found" in capsys.readouterr().out


def test_check_settings_reports_missing_meshes_as_doctor_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run cannot start without the arena meshes, so the check names the folder and how to fix it."""
    file = tmp_path / "bot.toml"
    file.write_text('[run]\nseed = 1\n[paths]\ncollision_meshes = "meshes"\n', encoding="utf-8")
    assert main(["check-settings", str(file)]) == 1
    out, err = capsys.readouterr()
    assert "no problems found" not in out and "Networks:" in out
    assert f"1 problem in {file}:" in err and "line 4, paths.collision_meshes:" in err
    assert f"{tmp_path / 'meshes'} does not exist" in err and "hint: " in err and "'soccar' subfolder" in err


def test_check_settings_explains_mistakes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = tmp_path / "bot.toml"
    file.write_text('[paths]\ncollision_mesh = "meshes"\n', encoding="utf-8")
    assert main(["check-settings", str(file)]) == 1
    error = capsys.readouterr().err
    assert "line 2, paths.collision_mesh" in error
    assert "did you mean 'collision_meshes'?" in error


def test_doctor_reports_a_bad_settings_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = tmp_path / "bot.toml"
    file.write_text("[paths]\ncollision_meshes = 5\n", encoding="utf-8")
    assert main(["doctor", str(file)]) == 1
    assert "must be a folder or file path" in capsys.readouterr().err


def test_export_writes_a_bot_folder_from_the_newest_save(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import torch

    from nitro.learner import Learner
    from nitro.model import ActorCritic
    from nitro.runs import RunFolder, fixed_settings

    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n', encoding="utf-8")
    runs = tmp_path / "runs"
    assert main(["export", "ghost", "--runs", str(runs), "--no-build"]) == 1
    assert "no run called 'ghost'" in capsys.readouterr().err

    folder = RunFolder.create(runs, "my-bot", settings)
    assert main(["export", "my-bot", "--runs", str(runs), "--no-build"]) == 1
    assert "no save yet" in capsys.readouterr().err

    config = folder.load_settings()
    model = ActorCritic(148, 90, config.model)
    state = {
        "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": 500,
        "iterations": 1,
        "settings": fixed_settings(config),
    }
    folder.save_checkpoint(state, 500, keep=5, milestone_steps=0)
    out = tmp_path / "bots" / "my-bot"
    assert main(["export", "my-bot", "--runs", str(runs), "--out", str(out), "--no-build"]) == 0
    printed = capsys.readouterr().out
    assert "policy.nwt" in printed and "--no-build" in printed
    assert (out / "bot.toml").is_file() and (out / "policy.nwt").is_file()


def test_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        main(["fly"])
    assert info.value.code == 2


def test_a_run_the_games_stop_ends_with_the_resume_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core stops the games during a run (e.g. the graphics card's physics): no traceback, the
    reason and how to go on from the last save, as for a TrainingProblem."""
    import logging
    import signal

    from nitro import _core, cli, training
    from nitro.runs import RunFolder

    class Stops:
        device_note = "cuda (FP32)"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, stop: object, until_steps: int | None = None) -> None:
            raise _core.EnvError("the physics of group 1 failed: a game ran into a problem")

    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    monkeypatch.setattr(training, "TrainingRun", Stops)
    logger = logging.getLogger("nitro")
    handlers, previous = list(logger.handlers), signal.getsignal(signal.SIGINT)
    try:
        assert cli._run_until_stopped(folder.load_settings(), folder, None, None, False) == 1
    finally:
        for handler in logger.handlers[len(handlers) :]:
            logger.removeHandler(handler)
            handler.close()
        signal.signal(signal.SIGINT, previous)
    err = capsys.readouterr().err
    assert "Training stopped: the physics of group 1 failed" in err
    resume = f'nitro resume my-bot --runs "{tmp_path / "runs"}"'
    assert f"No save yet. Fix it in the settings file it was started from ({settings.resolve()})" in err
    assert f"then start it again with: {resume}. It has no save yet, so it reads" in err, "a fresh run"
    assert "Training stopped: the physics of group 1" in folder.log.read_text(encoding="utf-8")
    # With a save, the line says to go on from it.
    (folder.checkpoints / f"{1000:012d}").mkdir(parents=True)
    (folder.checkpoints / f"{1000:012d}" / "state.pt").write_bytes(b"")
    try:
        assert cli._run_until_stopped(folder.load_settings(), folder, None, None, False) == 1
    finally:
        for handler in logger.handlers[len(handlers) :]:
            logger.removeHandler(handler)
            handler.close()
        signal.signal(signal.SIGINT, previous)
    assert f"To continue from the last save: {resume}" in capsys.readouterr().err


def test_a_file_that_cannot_be_written_stops_a_run_plainly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk (or another failed write) during a run: a plain message and how to go on, no traceback."""
    import errno
    import logging
    import signal
    import sys

    from nitro import cli, training
    from nitro.runs import RunFolder

    failure: list[OSError] = []

    class Fails:
        device_note = "cpu (FP32)"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, stop: object, until_steps: int | None = None) -> None:
            raise failure[0]

    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    monkeypatch.setattr(training, "TrainingRun", Fails)
    monkeypatch.setattr(logging, "raiseExceptions", logging.raiseExceptions)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    logger = logging.getLogger("nitro")
    handlers, previous = list(logger.handlers), signal.getsignal(signal.SIGINT)

    def stopped(error: OSError) -> str:
        failure[:] = [error]
        try:
            assert cli._run_until_stopped(folder.load_settings(), folder, None, None, False) == 1
        finally:
            for handler in logger.handlers[len(handlers) :]:
                logger.removeHandler(handler)
                handler.close()
            signal.signal(signal.SIGINT, previous)
        return capsys.readouterr().err

    resume = f'nitro resume my-bot --runs "{tmp_path / "runs"}"'
    err = stopped(OSError(errno.ENOSPC, "No space left on device"))
    assert f"Training stopped: the disk is full (the disk that holds {folder.root})." in err
    assert f"Nothing was saved yet; free some space on that disk, then start again with: {resume}" in err
    save = folder.checkpoints / f"{1000:012d}"
    save.mkdir(parents=True)
    (save / "state.pt").write_bytes(b"")
    err = stopped(PermissionError(errno.EACCES, "Permission denied", str(folder.report)))
    assert f"Training stopped: a file could not be written ({folder.report}): Permission denied." in err
    assert f"The last good save is at 1,000 agent steps ({save}); fix that, then start again with" in err
    assert "Traceback" not in err


def test_resume_goes_on_from_the_save_before_one_that_does_not_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newest save cut short: a plain warning naming it, and the run goes on from the one before; with no
    save that reads, a plain message and no traceback."""
    from nitro import cli
    from nitro.runs import RunFolder, fixed_settings

    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n', encoding="utf-8")
    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "my-bot", settings)
    fixed = fixed_settings(folder.load_settings())
    for steps in (300, 700):
        folder.save_checkpoint({"agent_steps": steps, "settings": fixed}, steps, keep=5, milestone_steps=0)
    newest = folder.checkpoints / f"{700:012d}" / "state.pt"
    newest.write_bytes(newest.read_bytes()[:100])
    resumed: list[Any] = []
    monkeypatch.setattr(cli, "_run", lambda config, folder, state, *rest: resumed.append(state) or 0)
    assert main(["resume", "my-bot", "--runs", str(runs)]) == 0
    assert resumed[0]["agent_steps"] == 300
    err = capsys.readouterr().err
    assert f"WARNING: The save of 'my-bot' at 700 agent steps ({newest}) cannot be read" in err
    assert "It was left where it is; 'my-bot' goes on from the save before, at 300 agent steps." in err
    (folder.checkpoints / f"{300:012d}" / "state.pt").write_bytes(b"")
    assert main(["resume", "my-bot", "--runs", str(runs)]) == 1
    err = capsys.readouterr().err
    assert "2 saves of 'my-bot'" in err and "has no other save" in err and "Traceback" not in err


def test_a_run_started_by_the_dashboard_says_nothing_of_ctrl_c(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard's Console tab shows the run's output, where Ctrl+C means nothing: a run it started
    (NITRO_JOB set) leaves the Ctrl+C hints out; one started in a terminal keeps them."""
    import logging
    import signal
    import sys

    from nitro import cli, training
    from nitro.dashboard.jobs import JOB_TOKEN
    from nitro.runs import RunFolder

    class Stopped:
        device_note = "cpu (FP32)"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, stop: object, until_steps: int | None = None) -> None:
            signal.raise_signal(signal.SIGINT)  # Ctrl+C, or the dashboard's Stop

    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    monkeypatch.setattr(training, "TrainingRun", Stopped)
    monkeypatch.setattr(logging, "raiseExceptions", logging.raiseExceptions)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    logger = logging.getLogger("nitro")
    handlers, previous = list(logger.handlers), signal.getsignal(signal.SIGINT)

    def output() -> str:
        try:
            assert cli._run_until_stopped(folder.load_settings(), folder, None, None, False) == 0
        finally:
            for handler in logger.handlers[len(handlers) :]:
                logger.removeHandler(handler)
                handler.close()
            signal.signal(signal.SIGINT, previous)
        return capsys.readouterr().out

    monkeypatch.delenv(JOB_TOKEN, raising=False)
    terminal = output()
    assert f"Training 'my-bot'. Progress: {folder.report} (open it in a browser). Ctrl+C stops." in terminal
    assert "Stopping after this iteration and saving. Press Ctrl+C again to quit at once" in terminal
    monkeypatch.setenv(JOB_TOKEN, "")  # as the dashboard's spawn sets it, a restart after Apply too
    dashboard = output()
    assert "Training 'my-bot'.\n" in dashboard and "Stopping after this iteration and saving.\n" in dashboard
    assert "Ctrl+C" not in dashboard


def test_a_trainer_started_again_after_apply_knows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the trainer the one before it started after the dashboard's Apply is given that process to wait
    for: it says its changes as applied."""
    from nitro import cli
    from nitro.dashboard.jobs import AFTER_PID

    monkeypatch.delenv(AFTER_PID, raising=False)
    assert cli._wait_for_previous_trainer() is False
    monkeypatch.setenv(AFTER_PID, str(os.getpid()))
    assert cli._wait_for_previous_trainer(limit=0.0) is True
    assert AFTER_PID not in os.environ, "not passed on to what this trainer starts"


def test_new_makes_a_bot_without_starting_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = tmp_path / "bot.toml"
    file.write_text('[paths]\ncollision_meshes = "meshes"\n', encoding="utf-8")
    runs = tmp_path / "runs"
    assert main(["new", str(file), "--name", "fresh", "--runs", str(runs)]) == 0
    assert "nitro resume fresh" in capsys.readouterr().out
    folder = runs / "fresh"
    assert (folder / "settings.toml").read_text(encoding="utf-8") == file.read_text(encoding="utf-8")
    assert (folder / "run.json").is_file() and not any((folder / "checkpoints").iterdir())
    assert main(["new", str(file), "--name", "fresh", "--runs", str(runs)]) == 1  # never over a bot
    assert "already exists" in capsys.readouterr().err


def test_new_refuses_a_settings_file_with_mistakes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file = tmp_path / "bot.toml"
    file.write_text('[paths]\ncollision_mesh = "meshes"\n', encoding="utf-8")
    assert main(["new", str(file), "--name", "fresh", "--runs", str(tmp_path / "runs")]) == 1
    assert "collision_mesh" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "fresh").exists()


def test_training_output_survives_a_closed_pipe() -> None:
    """`nitro train | tee train.log` and Ctrl+C: tee is gone, and a print must not stop training mid-iteration
    (with several cards the others would wait for it)."""
    from nitro.cli import _Unbroken

    class Gone:
        def write(self, text: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise BrokenPipeError

    stream = _Unbroken(Gone())
    assert stream.write("Stopping after this iteration") == len("Stopping after this iteration")
    stream.flush()
    print("and later prints too", file=stream, flush=True)


def test_the_dashboard_without_a_window_toolkit_says_how_to_get_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux without Qt or GTK for Python: a plain message with the install hint, no traceback."""
    from webview.errors import WebViewException

    from nitro.dashboard import app

    def no_window(*args: object, **kwargs: object) -> int:
        raise WebViewException("You must have either QT or GTK with Python extensions installed")

    monkeypatch.setattr(app, "run", no_window)
    monkeypatch.setattr("sys.platform", "linux")
    assert main(["dashboard", "--runs", str(tmp_path / "runs")]) == 1
    err = capsys.readouterr().err
    assert "The dashboard could not open its window: You must have either QT or GTK" in err
    assert 'pip install "pywebview[qt]"' in err
