"""Run folders, checkpoints, metrics and the report; a killed run resumes cleanly."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from nitro.config import ConfigError, NitroConfig
from nitro.config.loader import parse_toml
from nitro.metrics import MetricsLog
from nitro.runs import RunFolder, UnreadableSave, all_settings, fixed_settings
from nitro.training import arena_seed

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


def settings_file(folder: Path, text: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "settings.toml"
    file.write_text(text, encoding="utf-8")
    return file


def test_arena_seeds_start_at_the_run_seed_and_differ_per_checkpoint() -> None:
    assert arena_seed(7, 0) == 7
    seeds = {arena_seed(7, k) for k in range(1, 1000)}
    assert len(seeds) == 999 and all(0 <= s < 2**63 for s in seeds) and 7 not in seeds
    assert arena_seed(8, 1) != arena_seed(7, 1)


def test_a_run_folder_keeps_the_newest_complete_checkpoints(tmp_path: Path) -> None:
    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    with pytest.raises(FileExistsError, match="nitro resume a"):
        RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    assert folder.load_latest() is None
    for steps in (100, 200, 300):
        folder.save_checkpoint({"agent_steps": steps, "weights": torch.full((2,), steps)}, steps, 2, 0)
    # A save interrupted by a crash or power cut is never picked up.
    (folder.checkpoints / "000000000400.partial").mkdir()
    assert [p.name for p in folder.complete_checkpoints()] == ["000000000200", "000000000300"]
    state = folder.load_latest()
    assert (
        state is not None
        and state["agent_steps"] == 300
        and torch.equal(state["weights"], torch.full((2,), 300))
    )


def test_the_first_checkpoint_of_every_milestone_is_kept(tmp_path: Path) -> None:
    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    for steps in range(100, 701, 100):
        folder.save_checkpoint({"agent_steps": steps}, steps, keep=2, milestone_steps=250)
    assert [int(p.name) for p in folder.complete_checkpoints()] == [100, 300, 500, 600, 700]


def test_a_resumed_run_refuses_settings_its_saved_networks_cannot_follow(tmp_path: Path) -> None:
    original = settings_file(tmp_path, "[model]\npolicy = [64, 64]\n")
    folder = RunFolder.create(tmp_path / "runs", "a", original)
    state = {"settings": fixed_settings(folder.load_settings())}
    folder.settings.write_text("[ppo]\npolicy_lr = 1e-4\n[model]\npolicy = [64, 64]\n", encoding="utf-8")
    assert folder.load_settings(state).ppo.policy_lr == 1e-4
    folder.settings.write_text("[model]\npolicy = [128, 64]\nactivation = 'gelu'\n", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        folder.load_settings(state)
    assert "line 2, model.policy: was [64, 64] when the run was saved" in str(error.value)
    assert 'line 3, model.activation: was "leaky_relu"' in str(error.value)


def test_relative_paths_in_a_runs_settings_mean_what_they_did_next_to_the_original(tmp_path: Path) -> None:
    original = settings_file(tmp_path / "project", '[paths]\ncollision_meshes = "meshes"\n')
    folder = RunFolder.create(tmp_path / "elsewhere" / "runs", None, original)
    assert folder.load_settings().paths.collision_meshes == (tmp_path / "project" / "meshes").resolve()


def test_metrics_drop_the_iterations_a_resumed_run_plays_again(tmp_path: Path) -> None:
    path, columns = tmp_path / "metrics.csv", ["agent_steps", "entropy", "reward_per_step"]
    metrics = MetricsLog(path, columns)
    for steps in (100, 200, 300):
        metrics.write({"agent_steps": steps, "entropy": steps / 1000, "reward_per_step": ""})
    metrics.close()
    with path.open("a", encoding="utf-8") as file:
        file.write("40")  # the start of "400,...", cut short by a kill
    resumed = MetricsLog(path, columns, until=200)
    resumed.write({"agent_steps": 300, "entropy": 0.5, "reward_per_step": 0.1})
    resumed.close()
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(r["agent_steps"], r["entropy"]) for r in rows] == [
        ("100", "0.1"),
        ("200", "0.2"),
        ("300", "0.5"),
    ]
    resumed.write_report(tmp_path / "report.html", "test run", ["a note"], ["a warning"])
    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "test run" in report and "a warning" in report and "<svg" in report


def test_a_column_the_resumed_run_no_longer_has_keeps_its_history(tmp_path: Path) -> None:
    """Removing a reward before resuming must not erase its logged values."""
    path = tmp_path / "metrics.csv"
    first = MetricsLog(path, ["agent_steps", "weight_touch_ball"])
    first.write({"agent_steps": 100, "weight_touch_ball": 2.5})
    first.close()
    resumed = MetricsLog(path, ["agent_steps", "entropy"], until=100)
    resumed.write({"agent_steps": 200, "entropy": 0.5})
    resumed.close()
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(r["agent_steps"], r["weight_touch_ball"], r["entropy"]) for r in rows] == [
        ("100", "2.5", ""),
        ("200", "", "0.5"),
    ]


def test_a_resumed_learner_uses_the_current_learning_rates_not_the_saved_ones() -> None:
    import dataclasses

    from nitro.config import default_config
    from nitro.learner import Learner
    from nitro.model import ActorCritic

    config = default_config(Path.cwd())
    saved = Learner(ActorCritic(16, 5, config.model), config.ppo, 0.99, torch.device("cpu"), fp16=False)
    state = saved.state_dict()
    changed = dataclasses.replace(config.ppo, policy_lr=1e-5, value_lr=2e-5, adam_eps=1e-8)
    resumed = Learner(ActorCritic(16, 5, config.model), changed, 0.99, torch.device("cpu"), fp16=False)
    resumed.load_state_dict(state)
    by_name = dict(zip(resumed.groups, resumed.optimizer.param_groups, strict=True))
    assert (by_name["policy"]["lr"], by_name["value"]["lr"]) == (1e-5, 2e-5)
    assert all(group["eps"] == 1e-8 for group in resumed.optimizer.param_groups)


def test_every_setting_is_saved_and_changes_are_named_on_resume(tmp_path: Path) -> None:
    from nitro.runs import all_settings, changed_settings

    before = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path, "[ppo]\npolicy_lr = 1e-4\n"))
    saved = all_settings(before.load_settings())
    assert saved["ppo.policy_lr"] == 1e-4 and saved["env.num_arenas"] == 1024 and "rewards" in saved
    before.settings.write_text("[ppo]\npolicy_lr = 2e-4\nentropy = 0.01\n", encoding="utf-8")
    changes = changed_settings(saved, all_settings(before.load_settings()))
    assert changes == ["ppo.entropy: 0.035 -> 0.01", "ppo.policy_lr: 0.0001 -> 0.0002"]


def test_a_list_item_added_since_the_save_is_named_field_by_field(tmp_path: Path) -> None:
    from nitro.runs import all_settings, changed_settings

    text = '[[rewards]]\nname = "goal"\n'
    saved = all_settings(
        RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path, text)).load_settings()
    )
    more = text + '[[rewards]]\nname = "touch_ball"\nweight = 0.5\n'
    now = all_settings(
        RunFolder.create(tmp_path / "runs", "b", settings_file(tmp_path, more)).load_settings()
    )
    changes = changed_settings(saved, now)
    assert changes and all(change.startswith("rewards[1].") for change in changes), changes
    assert (
        "rewards[1].name: (none) -> touch_ball" in changes and "rewards[1].weight: (none) -> 0.5" in changes
    )
    assert changed_settings(now, saved) == [
        change.replace("(none) -> ", "") + " -> (none)" for change in changes
    ]


def test_a_save_removes_every_half_written_save_left_by_a_kill(tmp_path: Path) -> None:
    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    for stale in ("000000000100.partial", "000000000700.partial"):
        (folder.checkpoints / stale).mkdir()
        (folder.checkpoints / stale / "state.pt").write_bytes(b"half")
    folder.save_checkpoint({"agent_steps": 300}, 300, keep=5, milestone_steps=0)
    assert sorted(p.name for p in folder.checkpoints.iterdir()) == ["000000000300"]


def test_the_report_always_shows_the_latest_value(tmp_path: Path) -> None:
    from nitro.metrics import _SAMPLE_POINTS

    metrics = MetricsLog(tmp_path / "m.csv", ["agent_steps", "entropy"])
    for i in range(1, 3 * _SAMPLE_POINTS + 2):
        metrics.write({"agent_steps": i, "entropy": i})
    metrics.close()
    sample = metrics.samples["entropy"]
    assert sample.stride > 1 and len(sample.points) <= _SAMPLE_POINTS + 1
    assert sample.points[-1] == (3 * _SAMPLE_POINTS + 1, 3 * _SAMPLE_POINTS + 1)


def _rows(folder: Path) -> list[dict[str, str]]:
    with (folder / "metrics.csv").open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_killed_run_resumes_from_its_last_save(tmp_path: Path) -> None:
    settings = settings_file(
        tmp_path,
        f"[paths]\ncollision_meshes = '{MESHES.resolve()}'\n[run]\ndevice = 'cpu'\n"
        "[env]\nnum_arenas = 8\nthreads = 4\n[model]\npolicy = [32]\nvalue = [32]\n"
        "[ppo]\nsteps_per_iteration = 320\nminibatches = 2\n",
    )
    nitro = [sys.executable, "-m", "nitro"]
    runs = ["--runs", str(tmp_path / "runs")]
    run = tmp_path / "runs" / "k"
    subprocess.run([*nitro, "train", str(settings), "--name", "k", *runs, "--steps", "960"], check=True)
    assert [p.name for p in RunFolder(run).complete_checkpoints()] == ["000000000960"]

    # Killed (no chance to save) once it is past the save. Its target is far away so that a slow poll on a
    # busy machine cannot let it finish and save before the kill.
    resumed = subprocess.Popen([*nitro, "resume", "k", *runs, "--steps", "10000000"])
    deadline = time.monotonic() + 120
    while not any(int(r["agent_steps"]) > 1280 for r in _rows(run)):
        assert resumed.poll() is None and time.monotonic() < deadline
        time.sleep(0.2)
    resumed.kill()
    resumed.wait()

    subprocess.run([*nitro, "resume", "k", *runs, "--steps", "1920"], check=True)
    steps = [int(r["agent_steps"]) for r in _rows(run)]
    assert steps == list(range(320, 1921, 320)), "every iteration logged once, in order"
    assert RunFolder(run).complete_checkpoints()[-1].name == "000000001920"
    assert (run / "report.html").is_file() and "INFO" in (run / "train.log").read_text(encoding="utf-8")


def test_a_checkpoint_can_be_loaded_by_its_step_count(tmp_path: Path) -> None:
    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    for steps in (300, 700):
        folder.save_checkpoint({"agent_steps": steps}, steps, keep=5, milestone_steps=0)
    assert folder.load_checkpoint(300) == {"agent_steps": 300}
    assert folder.load_latest() == {"agent_steps": 700}
    assert folder.load_checkpoint(500) is None


def test_a_save_that_does_not_read_is_passed_over_with_a_plain_warning(tmp_path: Path) -> None:
    """A newest save cut short (a crash, a power cut or a full disk) is said by name and left where it
    is; the run goes on from the save before it. With no save that reads, a plain error says what to do."""
    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    for steps in (300, 700, 900):
        folder.save_checkpoint({"agent_steps": steps}, steps, keep=5, milestone_steps=0)
    newest = folder.checkpoints / "000000000900" / "state.pt"
    newest.write_bytes(newest.read_bytes()[: newest.stat().st_size // 2])  # truncated
    said: list[str] = []
    assert folder.load_latest(said.append) == {"agent_steps": 700}
    assert said == [
        f"The save of 'a' at 900 agent steps ({newest}) cannot be read: it is damaged or cut short (as by a "
        "crash, a power cut or a full disk while it was written). It was left where it is; 'a' goes on from "
        "the save before, at 700 agent steps."
    ]
    assert newest.is_file(), "never deleted"
    (folder.checkpoints / "000000000700" / "state.pt").write_bytes(b"")  # empty: a power cut
    said.clear()
    assert folder.load_latest(said.append) == {"agent_steps": 300}
    assert said[0].startswith(f"2 saves of 'a' in {folder.checkpoints} cannot be read (at 700, 900 agent ")
    assert said[0].endswith(
        "They were left where they are; 'a' goes on from the save before, at 300 agent steps."
    )
    (folder.checkpoints / "000000000300" / "state.pt").write_bytes(b"not a save" * 10)
    with pytest.raises(
        UnreadableSave, match=r"3 saves of 'a'.*has no other save\. To start it again"
    ) as info:
        folder.load_latest(said.append)
    assert "nitro resume a" in str(info.value)
    with pytest.raises(UnreadableSave, match=r"The save of 'a' at 900 agent steps \(.*\) cannot be read"):
        folder.load_checkpoint(900)


def test_a_save_that_cannot_be_written_leaves_no_half_written_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk: the half-written save is removed at once (its space back), the saves before stay."""
    import errno

    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    folder.save_checkpoint({"agent_steps": 300}, 300, keep=5, milestone_steps=0)

    def full(obj: object, file: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(torch, "save", full)
    with pytest.raises(OSError, match="No space left"):
        folder.save_checkpoint({"agent_steps": 700}, 700, keep=5, milestone_steps=0)
    assert sorted(p.name for p in folder.checkpoints.iterdir()) == ["000000000300"]
    assert folder.load_latest() == {"agent_steps": 300}


def test_a_source_given_on_the_command_line_is_kept_for_a_restart(tmp_path: Path) -> None:
    """`nitro train --from` stays with the run folder, so `nitro resume` before the first save
    still starts from that run."""
    original = settings_file(tmp_path, "")
    folder = RunFolder.create(tmp_path / "runs", "a", original, {"run": "my-bot", "at": 5})
    restarted = folder.load_settings()
    assert (restarted.start_from.run, restarted.start_from.at) == ("my-bot", 5)
    plain = RunFolder.create(tmp_path / "runs", "b", original)
    assert plain.load_settings().start_from.run == ""


def test_a_run_without_a_save_reads_its_fixed_settings_file_again(tmp_path: Path) -> None:
    """A run that stopped before its first save (a wrong path, a mistake) is fixed in the file it was started
    from: the next start takes that file again, unless the run's own copy was edited too."""
    file = settings_file(tmp_path / "mine", "[run]\nseed = 1\n")
    folder = RunFolder.create(tmp_path / "runs", "a", file)
    assert folder.started_with() == file.resolve() and folder.refresh_settings() is None
    file.write_text("[run]\nseed = 2\n", encoding="utf-8")
    note = folder.refresh_settings()
    assert note is not None and "it starts with the new settings" in note
    assert folder.settings.read_text(encoding="utf-8") == "[run]\nseed = 2\n"
    assert folder.refresh_settings() is None, "taken once"
    folder.settings.write_text("[run]\nseed = 3\n", encoding="utf-8")
    file.write_text("[run]\nseed = 4\n", encoding="utf-8")
    note = folder.refresh_settings()
    assert note is not None and "the run uses its copy" in note
    assert f"delete the folder {folder.root} and run: nitro train {file.resolve()} --name a" in note
    assert folder.settings.read_text(encoding="utf-8") == "[run]\nseed = 3\n", "an edited copy is kept"
    folder.save_checkpoint({"agent_steps": 5}, 5, keep=1, milestone_steps=0)
    file.write_text("[run]\nseed = 5\n", encoding="utf-8")
    assert folder.refresh_settings() is None, "a run with a save keeps its settings"
    with pytest.raises(FileExistsError, match=r"To continue it, use: nitro resume a"):
        RunFolder.create(tmp_path / "runs", "a", file)


def test_resume_to_fewer_steps_than_the_run_has_does_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nitro.cli import main

    folder = RunFolder.create(tmp_path / "runs", "a", settings_file(tmp_path))
    folder.save_checkpoint({"agent_steps": 500}, 500, keep=1, milestone_steps=0)
    assert main(["resume", "a", "--runs", str(tmp_path / "runs"), "--steps", "100"]) == 0
    said = capsys.readouterr().out
    assert "a already has 500 agent steps, past the 100 asked for: nothing was done." in said


def test_the_hints_to_continue_name_a_runs_folder_other_than_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nitro.runs import resume_command

    monkeypatch.chdir(tmp_path)
    assert resume_command(RunFolder(Path("runs") / "a")) == "nitro resume a"
    assert resume_command(RunFolder(tmp_path / "runs" / "a")) == "nitro resume a"
    elsewhere = tmp_path / "other" / "a"
    assert resume_command(RunFolder(elsewhere)) == f'nitro resume a --runs "{elsewhere.parent}"'
    folder = RunFolder.create(tmp_path / "other", "b", settings_file(tmp_path / "s"))
    with pytest.raises(FileExistsError) as made:
        RunFolder.create(tmp_path / "other", "b", settings_file(tmp_path / "s"))
    assert f'nitro resume b --runs "{folder.root.parent}"' in str(made.value)
    assert "It has no save yet: to start it again from" in str(made.value)


def test_saves_record_paths_relative_to_the_bots_folder_or_nitrolearns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NitroLearn's folder stands beside the test's own, so "elsewhere" is outside it wherever the
    # temporary folder is (tools/env.ps1 puts it inside NitroLearn's).
    project = tmp_path / "NitroLearn"
    monkeypatch.setattr("nitro.runs.PROJECT", project)
    run = tmp_path / "runs" / "bot"
    text = (
        f'[paths]\ncollision_meshes = "{(project / "collision_meshes").as_posix()}"\n'
        f'[pretrain]\nreplays = ["{(run / "settings-files" / "demos").as_posix()}", '
        f'"{(tmp_path / "elsewhere").as_posix()}"]\n'
    )
    config = parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    saved = all_settings(config, run)
    assert saved["paths.collision_meshes"] == "collision_meshes"
    assert saved["pretrain.replays"] == ["settings-files/demos", str(tmp_path / "elsewhere")], "else as is"
    assert all_settings(config)["pretrain.replays"][0] == str(run / "settings-files" / "demos")
