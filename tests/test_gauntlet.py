"""`nitro gauntlet` : scripted benchmark opponents in the simulator, the report card and its file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from nitro.cli import main
from nitro.config import load_config
from nitro.eval import MatchResult, load_side
from nitro.gauntlet import (
    DEFAULT_OPPONENTS,
    CardLine,
    append_report,
    play_scripts,
    run_gauntlet,
)
from nitro.gauntlet_card import newest, read_report, verdict
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import RunFolder, fixed_settings

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")


def _result(won: int, lost: int, draws: int) -> MatchResult:
    return MatchResult(won + lost + draws, won, lost, draws, 10.0, 60.0)


def test_the_card_reads_beats_loses_or_even() -> None:
    lines = [_result(97, 3, 0), _result(10, 90, 5), _result(48, 52, 0), _result(0, 0, 20)]
    entries = [
        CardLine(name, r).entry()
        for name, r in zip(("chaser", "goalie", "kickoff", "aerial"), lines, strict=True)
    ]
    texts = [verdict(e) for e in entries]
    assert texts[0].startswith("beats chaser 97% (92-99)")
    assert texts[1].startswith("loses to goalie 10%")
    assert texts[2].startswith("even with kickoff 48%")
    assert texts[3] == "aerial: nobody scored in 20 points"


def test_the_report_keeps_one_card_per_save_seed_and_games(tmp_path: Path) -> None:
    path = tmp_path / "gauntlet.json"
    assert read_report(path) == []
    card = [CardLine("chaser", _result(5, 1, 0))]
    append_report(path, 200, card, 1, 50)
    append_report(path, 100, card, 1, 50)
    append_report(path, 200, [CardLine("chaser", _result(6, 0, 0))], 1, 50)
    cards = read_report(path)
    assert [c["agent_steps"] for c in cards] == [100, 200]
    assert cards[1]["lines"][0]["won"] == 6
    shown = newest(path)
    assert shown is not None and shown["agent_steps"] == 200
    assert shown["lines"][0]["text"].startswith("beats chaser 100%")


def test_a_card_cut_off_while_written_keeps_the_earlier_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training stopped (or the PC turned off) while the report is written must not lose its history: the
    next card would otherwise start the file again from nothing."""
    path = tmp_path / "gauntlet.json"
    card = [CardLine("chaser", _result(5, 1, 0))]
    append_report(path, 100, card, 1, 50)
    before = path.read_text(encoding="utf-8")
    write = Path.write_text

    def cut_off(self: Path, data: str, *args: object, **kwargs: object) -> int:
        write(self, data[: len(data) // 2], encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", cut_off)
    with pytest.raises(KeyboardInterrupt):
        append_report(path, 200, card, 1, 50)
    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == before
    append_report(path, 200, card, 1, 50)
    assert [c["agent_steps"] for c in read_report(path)] == [100, 200]


def _bot(tmp_path: Path) -> RunFolder:
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nthreads = 2\nnum_arenas = 16\n[model]\npolicy = [8]\nvalue = [8]\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    config = folder.load_settings()
    torch.manual_seed(1)
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
    return folder


@needs_meshes
def test_an_untrained_bot_loses_to_the_chaser_and_the_same_seed_plays_the_same_games(tmp_path: Path) -> None:
    folder = _bot(tmp_path)
    side = load_side(folder, None)
    first = run_gauntlet(side, ("chaser", "idle"), games=32, seed=4)
    again = run_gauntlet(side, ("chaser", "idle"), games=32, seed=4)
    assert [line.entry() for line in first] == [line.entry() for line in again]
    chaser, idle = first[0].result, first[1].result
    assert chaser.orange_goals > chaser.blue_goals, "random play concedes to the chaser"
    assert idle.orange_goals == 0 or idle.orange_goals < chaser.orange_goals


@needs_meshes
def test_the_command_writes_the_card(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _bot(tmp_path)
    code = main(
        ["gauntlet", "my-bot", "--runs", str(tmp_path / "runs"), "--games", "16", "--opponents", "goalie"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "goalie" in out and "Saved to" in out
    cards = read_report(folder.root / "gauntlet.json")
    assert [line["opponent"] for line in cards[0]["lines"]] == ["goalie"]


@needs_meshes
def test_known_answers_between_scripted_players(tmp_path: Path) -> None:
    """Identical scripts split the goals evenly; the chaser beats a player that does nothing."""
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nthreads = 2\nnum_arenas = 32\n",
        encoding="utf-8",
    )
    config = load_config(settings)
    mirror = play_scripts(config, "chaser", "chaser", 256, 5)
    low, high = mirror.interval()
    assert low < 0.5 < high, (mirror.blue_goals, mirror.orange_goals)
    beaten = play_scripts(config, "chaser", "idle", 256, 5)
    assert beaten.interval()[0] > 0.5
    assert DEFAULT_OPPONENTS == ("chaser", "goalie", "kickoff", "aerial")
