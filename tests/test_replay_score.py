"""`nitro score-replay`: a real game scored with a bot's rewards and shown with the reward chart."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nitro import recordings, replay_score, replays
from nitro.cli import main
from nitro.viewer import recording as layouts
from test_pretrain_large import lean_file
from test_replays import random_arrays
from test_watch import NEEDS_MESHES, saved_run, tiny_config

pytestmark = NEEDS_MESHES


def bundled(name: str) -> Path:
    cargo = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "git" / "checkouts"
    copies = sorted(cargo.glob("rocketsim-*/*/rocketsim_test/recordings/*.rlpr.zst"))
    found = [path for path in [*recordings.bundled_recordings(), *copies] if name in path.name]
    if not found:
        pytest.skip("RocketSim's recordings are not here (the core was not built from source)")
    return found[0]


def marks(data: dict[str, object]) -> np.ndarray:
    frames = layouts.unpack(str(data["frames"]), layouts.width(int(str(data["cars"]))))
    return frames[:, 0].astype(int)


def test_a_real_game_is_scored_with_the_bots_rewards_and_shown_with_the_chart() -> None:
    config = tiny_config()
    data = replay_score.score(config, bundled("daizen_2v2"), seconds=300, fps=30)
    assert data["cars"] == 4 and data["fps"] == 30.0 and data["scored"] is True
    rewards = data["rewards"]
    labels = rewards["labels"]
    assert labels == [reward.name for reward in config.rewards]
    steps = len(rewards["frames"])
    pay = layouts.unpack_floats(rewards["pay"]).reshape(steps, len(labels), 4)
    assert steps > 2500 and np.isfinite(pay).all()
    goal = labels.index("goal")
    scored = [s for s in range(steps) if pay[s, goal].any()]
    assert scored, "a real 2v2 holds goals"
    shown = marks(data)
    for s in scored:
        # The scorers are paid, the other side charged, on the step whose last frame shows the goal.
        blue = pay[s, goal, :2].sum() > 0
        assert shown[rewards["frames"][s]] == (2 if blue else 3)
        assert (pay[s, goal, :2] > 0).all() == blue and (pay[s, goal, 2:] > 0).all() == (not blue)
    assert (shown == 1).sum() >= 1 and shown[0] == 1, "each stretch of play starts as a new game"
    assert np.all(np.diff(rewards["frames"]) > 0)
    totals = replay_score.totals(data)
    assert set(totals) == set(labels) and all(len(v) == 4 for v in totals.values())
    again = replay_score.score(config, bundled("daizen_2v2"), seconds=300, fps=30)
    assert again == data, "the same game scores the same"
    shorter = replay_score.score(config, bundled("daizen_2v2"), seconds=30, fps=30)
    assert len(shorter["rewards"]["frames"]) < steps / 5


def test_a_converted_replay_is_scored_from_the_store_and_a_replay_finds_its_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    arrays = random_arrays(3, per_side=1)
    players = [{"index": k, "team": k % 2} for k in range(arrays["car_position"].shape[1])]
    lean = tmp_path / "game.lean"
    lean_file(lean, arrays, players)
    assert replays.keep(lean.read_bytes(), "a game", 1)
    stored = replays.stored("a game", 1)
    data = replay_score.score(tiny_config(), stored, seconds=60)
    assert data["cars"] == 2 and len(data["rewards"]["frames"]) > 0
    replay = tmp_path / "a game.replay"
    replay.write_bytes(b"not read: its conversion is")
    assert replay_score.score(tiny_config(), replay, seconds=60) == data
    other = tmp_path / "other.replay"
    other.write_bytes(b"never converted")
    with pytest.raises(replay_score.ScoreError, match="has not been converted yet"):
        replay_score.score(tiny_config(), other, seconds=60)
    with pytest.raises(replay_score.ScoreError, match="is not a replay"):
        replay_score.score(tiny_config(), lean, seconds=60)
    with pytest.raises(replay_score.ScoreError, match="there is no file"):
        replay_score.score(tiny_config(), tmp_path / "missing.rlpr", seconds=60)


def test_the_command_writes_the_scored_game_and_says_what_each_reward_paid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = saved_run(tmp_path, 500)
    game = bundled("nexto_1v1")
    runs = folder.root.parent
    assert (
        main(["score-replay", "my-bot", str(game), "--runs", str(runs), "--seconds", "60", "--no-open"]) == 0
    )
    said = capsys.readouterr().out
    assert "What my-bot's rewards paid each player" in said and "goal" in said
    page = (folder.root / "replay_score.html").read_text(encoding="utf-8")
    data = layouts.from_page(page)
    assert data["title"] == f"{game.name} scored with my-bot's rewards" and "rewards" in data
    assert layouts.load(folder.root / "replay_score.nlrec")["rewards"] == data["rewards"]
    assert main(["score-replay", "my-bot", str(tmp_path / "x.rlpr"), "--runs", str(runs), "--no-open"]) == 1
    assert "there is no file" in capsys.readouterr().err


def test_the_dashboard_scores_a_chosen_game_and_opens_it_with_the_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from browser import playwright_or_skip
    from nitro.dashboard.api import Api
    from test_dashboard import ROOT
    from test_dashboard_watch import STARTED, Bridge, no_listening, opened, playing_bot
    from test_watch import chromium

    sync_api = playwright_or_skip()
    game = bundled("nexto_1v1")
    folder = playing_bot(tmp_path, 500)
    api = Api(ROOT, tmp_path / "runs")
    monkeypatch.setattr(Api, "choose_game", lambda self: {"game": str(game)})
    assert "there is no file" in api.watch_score("my-bot", str(tmp_path / "none.rlpr"))["error"]
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        no_listening(monkeypatch)
        try:
            tab = opened(browser, Bridge(api), problems)
            tab.click("#main [data-bot=my-bot]")
            tab.click("[data-act=watch]")
            tab.click("[data-act=watch-score]")
            tab.wait_for_function(f"{STARTED} && NV.rewards.state.ready", timeout=180_000)
            assert tab.evaluate("NV.rec.title") == f"{game.name} scored with my-bot's rewards"
            assert tab.locator(".nv-rewards-plot").is_visible()
            tab.keyboard.press("Escape")
            tab.wait_for_function("window.NV === null")
            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-rec=scored]")
            assert "Last scored:" in tab.text_content(".modal")
        finally:
            browser.close()
    assert (folder.root / "replay_score.nlrec").is_file()
    assert api.recordings("my-bot")["scored"]["id"] == "scored"
    assert problems == []
