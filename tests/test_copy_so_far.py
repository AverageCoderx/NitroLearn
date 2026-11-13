"""Saves of the copy of replays as it goes (pretrain.save_every): every so
many replays the copy so far becomes the bot's save, which can be watched, exported and have rewards tried on
it; the next start carries on the copy, and the finished copy replaces it. Also the value warm-up's progress
for the dashboard's Overview."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import cli, pretrain
from nitro.cli import main
from nitro.dashboard import bots
from nitro.dashboard.api import Api
from nitro.runs import COPY_SO_FAR, RunFolder
from test_dashboard import MESHES, NEEDS_MESHES, ROOT, TINY, items, make_bot
from test_pretrain_large import SETTINGS, SHELL, large, tools_and_replays


def test_a_save_of_the_copy_so_far_is_marked_found_replaced_and_dropped(tmp_path: Path) -> None:
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    assert folder.copy_so_far() is None, "no save"
    folder.save_checkpoint({"agent_steps": 0}, 0, 3, 0, copy_so_far={"learned": 2, "limit": 4})
    assert folder.copy_so_far() == {"learned": 2, "limit": 4}
    folder.save_checkpoint({"agent_steps": 0}, 0, 3, 0)  # the finished copy, at the same 0 steps
    assert folder.copy_so_far() is None and len(folder.complete_checkpoints()) == 1
    assert not (folder.checkpoints / f"{0:012d}" / COPY_SO_FAR).exists()
    folder.save_checkpoint({"agent_steps": 0}, 0, 3, 0, copy_so_far={"learned": 2, "limit": 4})
    folder.drop_copy_so_far()
    assert folder.complete_checkpoints() == []
    folder.save_checkpoint({"agent_steps": 0}, 0, 3, 0)
    folder.drop_copy_so_far()
    assert len(folder.complete_checkpoints()) == 1, "a save of its own is never dropped"


def test_copy_save_says_why_it_cannot_save(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    make_bot(runs, "fresh")
    assert main(["copy-save", "fresh", "--runs", str(runs)]) == 1
    assert "has not finished a round yet" in capsys.readouterr().err
    own = make_bot(runs, "own", saves=1)
    assert main(["copy-save", "own", "--runs", str(runs)]) == 1
    assert "a save of its own already" in capsys.readouterr().err
    assert own.complete_checkpoints(), "left as it was"


@SHELL
def test_a_copy_made_again_drops_the_save_of_the_old_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d", "e", "f", "g", "h"])
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    asked: list[int] = []

    class Stop(Exception):
        pass

    def saved(learned: int) -> None:  # as the real one marks its save; then stopped
        asked.append(learned)
        folder.save_checkpoint({"agent_steps": 0}, 0, 3, 0, copy_so_far={"learned": learned, "limit": 4})
        raise Stop

    config = large(tmp_path, demos, batch_replays=1, save_every=2)
    with pytest.raises(Stop):
        pretrain.start_source(config, folder, lambda line: None, save_so_far=saved)
    assert asked == [2], "after the round that reached 2 replays"
    assert folder.copy_so_far() == {"learned": 2, "limit": 4}

    seen: list[bool] = []

    def heard(line: str) -> None:
        if "large replay mode: up to" in line:  # the copy made again from its start
            seen.append(bool(folder.complete_checkpoints()))

    other = large(tmp_path, demos, batch_replays=1, keep_batches=1, save_every=0)
    assert pretrain.start_source(other, folder, heard) is not None
    assert seen == [False], "the old copy's save went before the new copy began"


def warmup_bot(runs: Path, name: str, steps: int, row_steps: int | None) -> RunFolder:
    """A bot that started from a copy of replays, with a value warm-up and a metrics row at `row_steps`."""
    folder = make_bot(runs, name, TINY + f"[pretrain]\nvalue_warmup_steps = {steps}\n")
    path = folder.checkpoints / f"{0:012d}"
    path.mkdir()
    state: dict[str, Any] = {
        "agent_steps": 0,
        "started_from": pretrain.COPY_LABEL,
        "started_from_replays": True,
    }
    torch.save(state, path / "state.pt")
    if row_steps is not None:
        folder.metrics.write_text(f"agent_steps,iteration\n{row_steps},1\n", encoding="utf-8")
    return folder


def test_the_value_warm_up_shows_how_far_it_is(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    held = warmup_bot(runs, "held", 1000, 400)
    assert bots.summary(held)["warmup"] == {"path": "pretrain.value_warmup_steps", "steps": 1000, "done": 400}
    done = warmup_bot(runs, "done", 1000, 5000)
    assert bots.summary(done)["warmup"]["done"] == 1000, "at most its steps"
    assert bots.summary(warmup_bot(runs, "none", 0, 400))["warmup"] is None, "no warm-up"
    assert bots.summary(make_bot(runs, "scratch", saves=1))["warmup"] is None, "started from scratch"
    shown = items(Api(ROOT, runs).bot("held")["settings"])["pretrain.value_warmup_steps"]
    assert not shown["fixed"], "it can change while the bot trains"


@SHELL
@NEEDS_MESHES
def test_the_copy_saves_itself_as_it_goes_and_the_next_start_carries_it_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d", "e", "f", "g", "h"])
    file = tmp_path / "run.toml"
    file.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n'
        + SETTINGS
        + "[env]\nnum_arenas = 4\nthreads = 2\n[ppo]\nsteps_per_iteration = 160\nminibatches = 2\n"
        + f'[pretrain]\nenabled = true\nreplays = "{demos.as_posix()}"\nlimit = 4\npasses = 2\n'
        + "batch_replays = 1\nworkers = 2\nminibatch = 32\nsave_every = 2\nvalue_warmup_steps = 100000\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    real = cli._save_so_far

    def save_then_stop(folder: RunFolder, learned: int, say: Any) -> None:
        real(folder, learned, say)  # `nitro copy-save` in a process of its own, as the dashboard runs it
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_save_so_far", save_then_stop)
    assert main(["train", str(file), "--name", "big", "--runs", str(runs)]) == 1
    printed = capsys.readouterr().out
    assert "saved the copy so far (2 replays) as the bot's save" in printed, printed[-2000:]
    folder = RunFolder(runs / "big")
    assert folder.copy_so_far() == {"learned": 2, "limit": 4}
    state = folder.load_latest()
    assert state is not None and state["agent_steps"] == 0 and state["started_from_replays"]
    progress = torch.load(folder.root / pretrain.COPY / pretrain.PROGRESS, weights_only=True)
    assert progress["used"] == 2
    for name, weights in progress["model"].items():
        if not name.startswith("value."):
            assert torch.equal(state["learner"]["model"][name], weights), f"{name}: the copy so far, exactly"

    shown = bots.summary(folder)
    assert shown["copy_so_far"] == {"learned": 2, "limit": 4} and shown["saves"] == 1
    assert shown["copy_first"] and not shown["start"]["done"] and shown["start_word"] == "Copy replays"
    assert shown["warmup"] is None, "no training before the copy is finished"
    settings = items(Api(ROOT, runs).bot("big")["settings"])
    assert not settings["model.policy"]["fixed"], "the copy can still be made again with other networks"

    assert main(["reward-check", "big", "--runs", str(runs), "--steps", "100"]) == 0, "Try these rewards"
    out = tmp_path / "rlbot-bot"
    exported = main(["export", "big", "--runs", str(runs), "--out", str(out), "--no-build"])
    assert exported == 0, "Make RLBot bot"
    assert any(out.iterdir())
    capsys.readouterr()

    monkeypatch.setattr(cli, "_save_so_far", real)
    assert main(["resume", "big", "--runs", str(runs)]) == 0
    printed = capsys.readouterr().out
    assert "has a save of its copy so far only (2 replays): it goes on with its copy" in printed
    assert "carrying on after round 2" in printed and "  round 4: 4 of 4 replays learned" in printed
    assert "saved the copy so far" not in printed, "the last round is saved as the finished copy"
    assert folder.copy_so_far() is None and len(folder.complete_checkpoints()) == 1
    copy = torch.load(folder.root / pretrain.COPY / "state.pt", weights_only=True)
    assert copy["pretrained"]["replays"] == 4
    shown = bots.summary(folder)
    assert shown["start"]["done"] and shown["start_word"] == "Start training from the copy"
    assert shown["warmup"] == {"path": "pretrain.value_warmup_steps", "steps": 100000, "done": 0}

    assert main(["resume", "big", "--runs", str(runs), "--steps", "1"]) == 0
    warm = bots.summary(folder)["warmup"]
    assert 0 < warm["done"] < 100000, "the warm-up counts the steps trained"
