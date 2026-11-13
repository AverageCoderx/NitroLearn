"""`nitro reward-check`: a run's newest save plays without learning and says what each reward
tried would pay it; nothing of the run changes. The games need the collision meshes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import reward_check
from nitro.cli import main
from nitro.learner import Learner
from nitro.runs import RunFolder

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

REWARDS = '[[rewards]]\nname = "touch_ball"\n\n[[rewards]]\nname = "velocity_player_to_ball"\nweight = 0.5\n'


def settings(tmp_path: Path) -> Path:
    file = tmp_path / "bot.toml"
    file.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[model]\npolicy = [32]\nvalue = [32]\n[env]\nnum_arenas = 4\nthreads = 2\n"
        "[ppo]\nsteps_per_iteration = 160\nminibatches = 2\n\n" + REWARDS,
        encoding="utf-8",
    )
    return file


def copying_settings(tmp_path: Path) -> Path:
    """A bot that copies replays first ([pretrain])."""
    file = tmp_path / "copying.toml"
    text = settings(tmp_path).read_text(encoding="utf-8")
    file.write_text(text + "\n[pretrain]\nenabled = true\n", encoding="utf-8")
    return file


def test_the_result_ranks_the_rewards_by_share_with_a_step_and_a_minute() -> None:
    result = reward_check.summary(
        ["a", "b"], raw=[2.0, 50.0], paid=[1.0, -3.0], size=[1.0, 3.0], agent_steps=100, tick_skip=8
    )
    first, second = result["ranked"]
    assert first["label"] == "b" and first["share"] == 0.75 and second["share"] == 0.25
    assert first["paid_per_step"] == -0.03 and first["raw_per_step"] == 0.5
    assert second["paid_per_minute"] == pytest.approx(0.01 * 900), "900 decisions a minute at tick skip 8"
    assert result["minutes_of_play"] == pytest.approx(100 / 900) and result["finished"]
    lonely = reward_check.summary(["a", "b"], [1, 0], [1, 0], [1, 0], 10, 8)
    assert [(a["label"], a["level"]) for a in lonely["alerts"]] == [("b", "watch")], "b paid nothing"
    drowned = reward_check.summary(["a", "b"], [1, 95], [1, 95], [1, 95], 10, 8)["alerts"]
    assert [(a["label"], a["level"]) for a in drowned] == [("b", "warning")]
    nothing = reward_check.summary(["a"], [0.0], [0.0], [0.0], 0, 8)
    assert nothing["ranked"][0]["share"] == 0.0 and nothing["reward_per_step"] == 0.0
    assert first["fire_rate"] is None, "no counts given: no rate"
    fired = reward_check.summary(["a", "b"], [2.0, 50.0], [1.0, -3.0], [1.0, 3.0], 100, 8, fired=[50, 2])
    by = {item["label"]: item for item in fired["ranked"]}
    assert by["a"]["fire_rate"] == 0.5 and by["b"]["fires_per_minute"] == pytest.approx(0.02 * 900)


@needs_meshes
def test_a_check_plays_the_newest_save_learning_nothing_and_keeps_only_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    file = settings(tmp_path)
    RunFolder.create(runs, "fresh", file)
    assert main(["reward-check", "fresh", "--runs", str(runs)]) == 1
    assert "has no save yet" in capsys.readouterr().err
    assert main(["reward-check", "ghost", "--runs", str(runs)]) == 1

    assert main(["train", str(file), "--name", "bot", "--runs", str(runs), "--steps", "1"]) == 0
    folder = RunFolder(runs / "bot")
    saves = {p: (p / "state.pt").read_bytes() for p in folder.complete_checkpoints()}
    metrics = folder.metrics.read_text(encoding="utf-8")
    capsys.readouterr()

    def no_learning(*args: object, **kwargs: object) -> None:
        raise AssertionError("a reward check learned")

    monkeypatch.setattr(Learner, "learn", no_learning)
    assert main(["reward-check", "bot", "--runs", str(runs), "--steps", "500"]) == 0
    printed = capsys.readouterr().out
    assert "What each reward would have paid 'bot'" in printed and "velocity_player_to_ball" in printed
    kept = sorted((folder.root / reward_check.CHECKS).iterdir())
    assert len(kept) == 1 and kept[0].suffix == ".json", "the result only: the games' folder is gone"
    result = json.loads(kept[0].read_text(encoding="utf-8"))
    assert result["finished"] and result["agent_steps"] >= 500 and result["bot"] == "bot"
    assert {r["label"] for r in result["ranked"]} == {"touch_ball", "velocity_player_to_ball"}
    assert sum(r["share"] for r in result["ranked"]) == pytest.approx(1.0)
    assert result["saved_steps"] == int(folder.complete_checkpoints()[-1].name)
    assert reward_check.newest(folder) == {**result, "file": kept[0].name}
    assert {p: (p / "state.pt").read_bytes() for p in folder.complete_checkpoints()} == saves
    assert folder.metrics.read_text(encoding="utf-8") == metrics, "nothing of the run changed"

    # Other rewards, from a reward set: only they are paid; the run's settings stay as they are.
    trial = tmp_path / "trial.toml"
    trial.write_text('# a set\n[[rewards]]\nname = "face_ball"\nweight = 2.0\n', encoding="utf-8")
    before = folder.settings.read_text(encoding="utf-8")
    out = tmp_path / "result.json"
    command = ["reward-check", "bot", "--runs", str(runs), "--steps", "200", "--rewards", str(trial)]
    assert main([*command, "--out", str(out)]) == 0
    tried = json.loads(out.read_text(encoding="utf-8"))
    assert [r["label"] for r in tried["ranked"]] == ["face_ball"] and tried["ranked"][0]["share"] == 1.0
    assert tried["rewards"] == ["face_ball"]
    assert folder.settings.read_text(encoding="utf-8") == before
    capsys.readouterr()

    trial.write_text("[run]\nseed = 3\n", encoding="utf-8")
    assert main(command) == 1
    assert "lists no [[rewards]] to try" in capsys.readouterr().err
    trial.write_text('[[rewards]]\nname = "no_such_reward"\n', encoding="utf-8")
    assert main(command) == 1
    assert "the rewards tried" in capsys.readouterr().err, "the mistake is named as the rewards tried"
    assert len(list((folder.root / reward_check.CHECKS).iterdir())) == 1, "a refused check keeps nothing"


@needs_meshes
def test_the_dashboard_tries_the_rewards_as_edited_on_the_frozen_bot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import time

    from nitro.dashboard.api import Api

    runs = tmp_path / "runs"
    file = settings(tmp_path)
    RunFolder.create(runs, "to-copy", copying_settings(tmp_path))
    api = Api(Path(__file__).parents[1], runs)
    listed = {b["name"]: b for b in api.home()["bots"]}
    assert listed["to-copy"]["copy_first"] and listed["to-copy"]["status"] == "stopped"
    assert "no save yet" in api.try_rewards("to-copy")["error"]

    assert main(["train", str(file), "--name", "bot", "--runs", str(runs), "--steps", "1"]) == 0
    capsys.readouterr()
    assert not api.bot("bot")["copy_first"]
    entries = [{"from": e["from"], "view": e["view"]} for e in api.rewards("bot")["rewards"]]
    entries[1]["view"]["weight"] = 2.0  # velocity_player_to_ball, waiting for Apply
    api.stage_rewards("bot", entries)
    before = (runs / "bot" / "settings.toml").read_text(encoding="utf-8")
    assert api.try_rewards("bot", 300) == {"started": True}
    assert "is trying rewards" in api.try_rewards("bot", 300)["error"]
    # No training beside a check. Refused either because the check is listed as playing or,
    # once its process has claimed the bot, as running (seen under the full suite's load).
    refused = api.start("bot")
    assert "started" not in refused, refused
    assert any(words in refused.get("error", "") for words in ("is trying rewards", "is already training")), (
        refused
    )
    deadline = time.monotonic() + 300
    while (shown := api.reward_check("bot"))["playing"] and time.monotonic() < deadline:
        # The check may end between the two reads (seen under the full suite's load): then it no longer plays.
        assert api.bot("bot")["checking"] or not api.reward_check("bot")["playing"]
        time.sleep(0.5)
    assert not shown["playing"] and not shown["error"], shown
    ranked = {r["label"]: r for r in shown["result"]["ranked"]}
    chase = ranked["velocity_player_to_ball"]
    assert chase["raw_per_step"] != 0 and chase["paid_per_step"] == pytest.approx(2 * chase["raw_per_step"])
    assert (runs / "bot" / "settings.toml").read_text(encoding="utf-8") == before, "tried, not applied"
    assert api.rewards("bot")["pending"]


def test_the_page_shows_copy_replays_first_and_the_rewards_tried(tmp_path: Path) -> None:

    sync_api = playwright_or_skip()
    from nitro.dashboard.api import Api
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "copier", copying_settings(tmp_path))
    result = reward_check.summary(
        ["touch_ball", "velocity_player_to_ball"], [0.01, 0.2], [0.01, 0.1], [0.01, 0.1], 9000, 8,
        extra={"bot": "copier", "saved_steps": 0, "when": 0},
    )  # fmt: skip
    (folder.root / reward_check.CHECKS).mkdir()
    (folder.root / reward_check.CHECKS / "20261019-120000.json").write_text(json.dumps(result), "utf-8")
    api = Api(Path(__file__).parents[1], runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1400, "height": 900})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click(".botcard")
            tab.wait_for_selector("[data-act=start]:has-text('Copy replays')")
            tab.wait_for_selector("text=copies them into it and saves it, then stops")
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("#rw-try >> text=Try these rewards")
            assert tab.is_disabled("[data-act=rw-try]"), "nothing to play before the copy is saved"
            tab.wait_for_selector("#rw-try .share >> text=91%")
            tab.wait_for_selector("#rw-try >> text=the others would hardly count")
            tab.wait_for_selector("#rw-try >> text=a minute")
            tab.screenshot(path=str(tmp_path / "try.png"), full_page=True)
        finally:
            browser.close()
    assert errors == []
