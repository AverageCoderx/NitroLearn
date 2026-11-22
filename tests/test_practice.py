"""The reward timeline of a watched game (`nitro watch --rewards`) and practice starts (`nitro watch
--moments`, `nitro practice`, the viewer's Practice panel): what the timeline holds adds up to what the
learner is paid, and a saved moment is exactly the state a game then starts in."""

from __future__ import annotations

import csv
import dataclasses
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nitro import practice
from nitro.cli import main
from nitro.config.state_setters import Snapshots
from nitro.envs import EnvPool, probe_config
from nitro.runs import numbered
from nitro.training import build
from nitro.viewer import recording as layouts
from nitro.watch import page, record
from test_viewer import GRAPHICS
from test_watch import NEEDS_MESHES, saved_run, tiny_config

pytestmark = NEEDS_MESHES


def recorded(seconds: float = 4.0, **kept: bool) -> Any:
    config = tiny_config()
    state = {"learner": {"model": build(config).model.state_dict()}}
    return config, state, record(config, state, seconds=seconds, seed=3, fps=30, **kept)


def in_order(pay: np.ndarray) -> np.ndarray:
    """Each car's rewards added up one after another in float32, as the core adds them."""
    total = np.zeros(pay.shape[::2], dtype=np.float32)
    for reward in range(pay.shape[1]):
        total = total + pay[:, reward]
    return total


def test_a_watched_games_reward_timeline_adds_up_to_each_steps_reward_and_changes_nothing() -> None:
    config, state, game = recorded(rewards=True)
    timeline = game.rewards
    assert timeline is not None and game.moments is None
    steps = len(game.decisions)
    assert timeline.pay.shape == (steps, len(timeline.labels), 2) and timeline.total.shape == (steps, 2)
    np.testing.assert_array_equal(in_order(timeline.pay).view(np.uint32), timeline.total.view(np.uint32))
    assert np.abs(timeline.pay).sum() > 0, "a game pays something"
    every = game.tick_skip // game.every
    assert timeline.frames == [d[0] + every for d in game.decisions], "each step ends a decision later"
    plain = record(config, state, seconds=4.0, seed=3, fps=30)
    np.testing.assert_array_equal(plain.frames, game.frames)
    assert plain.decisions == game.decisions and plain.rewards is None
    data = game.data("t")
    assert data["rewards"]["labels"] == timeline.labels and data["rewards"]["frames"] == timeline.frames
    np.testing.assert_array_equal(layouts.unpack_floats(data["rewards"]["pay"]), timeline.pay.ravel())
    assert "rewards" not in plain.data("t") and "moments" not in plain.data("t")


def test_the_pay_kept_per_step_adds_up_to_what_the_runs_log_reports() -> None:
    """The same seed and actions with and without keeping the pay: the same rewards, and the pay kept adds up
    to the core's statistics, which become metrics.csv's paid_ columns and reward_per_step."""
    config = tiny_config()
    config = dataclasses.replace(config, env=dataclasses.replace(config.env, num_arenas=1, threads=1))
    pools = [EnvPool(probe_config(config, 7), pin_memory=False) for _ in range(2)]
    pools[0].keep_reward_parts(0)
    labels = numbered(pools[0].reward_labels)
    sums = np.zeros(len(labels))
    generator = torch.Generator().manual_seed(5)
    for pool in pools:
        pool.take_stats()
    for _ in range(300):
        actions = torch.randint(0, pools[0].num_actions, (2,), generator=generator, dtype=torch.int32)
        rewards = []
        for pool in pools:
            group = pool.groups[0]
            group.actions.copy_(actions)
            pool.step(0)
            pool.wait(0)
            rewards.append(group.reward.clone())
        torch.testing.assert_close(rewards[0], rewards[1], rtol=0, atol=0)
        parts = np.asarray(pools[0].reward_parts(0), dtype=np.float32).reshape(len(labels), 2)
        np.testing.assert_array_equal(in_order(parts[None])[0], rewards[0].numpy())
        sums += parts.astype(np.float64).sum(axis=1)
    assert pools[1].reward_parts(0) == [], "not kept unless asked"
    logged = pools[0].take_stats()
    np.testing.assert_allclose(sums, logged.weighted, rtol=1e-9, atol=1e-9)
    assert sum(logged.weighted) / logged.agent_steps == pytest.approx(sums.sum() / 600)


def start_from(config: Any, csv_file: Path, seed: int = 1) -> list[list[float]]:
    """Every car's moment in an arena whose only start is the moments in `csv_file`."""
    only = (Snapshots(file=csv_file, weight=1.0),)
    env = dataclasses.replace(config.env, num_arenas=1, threads=1)
    pool = EnvPool(
        probe_config(dataclasses.replace(config, state_setters=only, env=env), seed), pin_memory=False
    )
    return pool.moments(0)


@pytest.mark.parametrize("car", [0, 1])
def test_a_moment_kept_from_a_game_is_the_state_a_game_then_starts_in(tmp_path: Path, car: int) -> None:
    config, _, game = recorded(moments=True)
    assert game.moments is not None and game.moments.shape == (len(game.decisions), 2, layouts.MOMENT)
    data = game.data("t")
    chosen = practice.from_recording(data, seconds=2.0, car=car)
    decision = int(np.argmin([abs(d[0] / 30 - 2.0) for d in game.decisions]))
    np.testing.assert_array_equal(np.float32(chosen), game.moments[decision, car])
    file = tmp_path / "start.csv"
    with file.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(practice.COLUMNS)
        writer.writerow(practice.moment_line("practice", chosen))
    blue = start_from(config, file)[0]  # 1v1: the moment goes to the only Blue car
    for name, want, got in zip(layouts.MOMENT_COLUMNS, chosen, blue, strict=True):
        assert np.float32(got) == np.float32(want), name


def test_practice_adds_moments_to_the_runs_starts_and_its_settings_once(tmp_path: Path) -> None:
    folder = saved_run(tmp_path, 500)
    _, _, game = recorded(moments=True)
    data = game.data("t")
    moments = [
        ("practice", practice.from_recording(data, 1.0, 0)),
        ("x", practice.from_recording(data, 3.0, 1)),
    ]
    file = practice.add(folder, moments[:1])
    settings = folder.settings.read_text(encoding="utf-8")
    assert settings.count('name = "snapshots"') == 1 and 'file = "' in settings and "weight = 0.2" in settings
    practice.add(folder, moments[1:])
    assert folder.settings.read_text(encoding="utf-8") == settings, "the start is added once"
    assert practice.read_moments(file.read_text(encoding="utf-8")) == [
        (group, [float(np.float32(v)) for v in values]) for group, values in moments
    ]
    practice.add(folder, [], weight=0.5)
    changed = folder.settings.read_text(encoding="utf-8")
    assert "weight = 0.5" in changed and changed.count('name = "snapshots"') == 1
    config = folder.load_settings()
    snapshots = [s for s in config.state_setters if isinstance(s, Snapshots)]
    assert len(snapshots) == 1 and snapshots[0].weight == 0.5
    assert snapshots[0].file.resolve() == file.resolve(), "the settings name the run's own file"
    blue = start_from(config, file, seed=11)[0]
    assert [np.float32(v) for v in blue] in [[np.float32(v) for v in m] for _, m in moments]

    before = (folder.settings.read_text(encoding="utf-8"), file.read_text(encoding="utf-8"))
    with pytest.raises(practice.PracticeError, match="not a moments file"):
        practice.read_moments("a,b\n1,2\n")
    with pytest.raises(practice.PracticeError, match="not finite"):
        practice.add(folder, [("p", [float("nan")] * layouts.MOMENT)])
    assert (folder.settings.read_text(encoding="utf-8"), file.read_text(encoding="utf-8")) == before


def test_the_commands_record_moments_and_add_one_to_the_bot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = saved_run(tmp_path, 500)
    runs = folder.root.parent
    watch = ["watch", "my-bot", "--runs", str(runs), "--seconds", "2", "--no-open"]
    assert main([*watch, "--rewards", "--moments"]) == 0
    data = layouts.load(folder.root / "watch.nlrec")
    assert "rewards" in data and data["moments"]["columns"] == layouts.MOMENT_COLUMNS
    assert layouts.from_page((folder.root / "watch.html").read_text(encoding="utf-8"))["run"] == "my-bot"
    assert main(["practice", "my-bot", "--runs", str(runs), "--at", "1.0", "--car", "1"]) == 0
    assert "has 1 practice start" in capsys.readouterr().out
    assert practice.count(folder) == 1
    assert main(["practice", "my-bot", str(tmp_path / "missing.csv"), "--runs", str(runs)]) == 1
    assert "Nothing was added" in capsys.readouterr().err
    assert main([*watch, "--live", "--rewards"]) == 1
    assert main([*watch]) == 0
    assert main(["practice", "my-bot", "--runs", str(runs), "--at", "1.0"]) == 1
    assert "keeps no moments" in capsys.readouterr().err
    assert practice.count(folder) == 1


def test_the_viewers_command_finds_the_file_in_the_downloads_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The page saves a moments file as a download and shows `nitro practice BOT "practice-....csv"`, to run
    in NitroLearn's folder: the bare name is looked for in the Downloads folder too."""
    folder = saved_run(tmp_path, 500)
    runs = folder.root.parent
    _, _, game = recorded(moments=True)
    line = practice.moment_line("practice", practice.from_recording(game.data("t"), 1.0, 0))
    downloads = tmp_path / "home" / "Downloads"
    downloads.mkdir(parents=True)
    (downloads / "practice-20250101120000.csv").write_text(
        ",".join(practice.COLUMNS) + "\n" + ",".join(line) + "\n", encoding="utf-8"
    )
    project = tmp_path / "NitroLearn"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert main(["practice", "my-bot", "practice-20250101120000.csv", "--runs", str(runs)]) == 0
    assert "has 1 practice start" in capsys.readouterr().out
    assert main(["practice", "my-bot", "practice-other.csv", "--runs", str(runs)]) == 1
    assert "Nothing was added" in capsys.readouterr().err
    assert practice.count(folder) == 1


def test_the_page_charts_the_rewards_jumps_to_a_spike_and_saves_exact_starts(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    config, _, game = recorded(seconds=6.0, rewards=True, moments=True)
    timeline = game.rewards
    assert timeline is not None and game.moments is not None
    file = tmp_path / "game.html"
    file.write_text(page(game, "a game", run="my-bot"), encoding="utf-8")
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        try:
            browser = play.chromium.launch(args=GRAPHICS)
        except sync_api.Error as error:
            pytest.skip(f"no Chromium to open the page in: {error}")
        try:
            tab = browser.new_page(viewport={"width": 1100, "height": 700}, accept_downloads=True)
            tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
            tab.on("console", lambda said: problems.append(said.text) if said.type == "error" else None)
            tab.goto(file.as_uri())
            tab.wait_for_function(
                "typeof NV !== 'undefined' && NV.frames > 2 && NV.rewards.state.ready && NV.practice.ready",
                timeout=120_000,
            )
            tab.evaluate("NV.player.playing = false")
            # The chart reads what Python wrote, step by step.
            step = 40
            shown = tab.evaluate("(f) => NV.rewards.payAt(f, 0)", timeline.frames[step])
            assert shown == {
                label: float(timeline.pay[step, r, 0]) for r, label in enumerate(timeline.labels)
            }
            names = tab.evaluate("NV.rewards.series.map((s) => s.name)")
            assert names and set(names) <= {*timeline.labels, "Other"}
            assert tab.locator(".nv-rewards-chip").count() == len(names)
            # A click near a spike goes to the start of the biggest step there.
            sizes = np.abs(timeline.pay[:, :, 0].sum(axis=1))
            spike = int(np.argmax(sizes))
            box = tab.locator(".nv-rewards-plot").bounding_box()
            assert box is not None
            last = tab.evaluate("NV.rec.count - 1")
            x = box["x"] + 8 + timeline.frames[spike] / last * (box["width"] - 16)
            tab.mouse.click(x, box["y"] + box["height"] / 2)
            start = timeline.frames[spike - 1] if spike else 0
            assert tab.evaluate("Math.round(NV.player.t * NV.rec.fps)") == start
            assert tab.locator(".nv-rewards-peek").is_visible()
            # Hiding a reward takes it off the chart and the legend says so.
            first = tab.locator(".nv-rewards-chip").first
            first.click()
            assert first.get_attribute("aria-pressed") == "false"
            tab.keyboard.press("r")
            assert tab.locator(".nv-rewards-plot").is_hidden()
            tab.keyboard.press("r")
            assert tab.locator(".nv-rewards-plot").is_visible()

            # Practice this moment: the followed car's exact moment, in a file nitro practice reads.
            tab.evaluate("NV.player.seek(2)")
            tab.click(".nv-practice-open")
            with tab.expect_download() as download:
                tab.get_by_role("button", name="Practice this moment").click()
            saved = Path(download.value.path()).read_text(encoding="utf-8")
            car = tab.evaluate("NV.cams.car")
            decision = int(np.argmin([abs(d[0] / 30 - 2.0) for d in game.decisions]))
            [(group, values)] = practice.read_moments(saved)
            assert group == "practice"
            np.testing.assert_array_equal(np.float32(values), game.moments[decision, car])
            command = tab.locator(".nv-practice-command").inner_text()
            assert re.fullmatch(
                r'(\.\\nitro|\./nitro\.sh) practice my-bot "practice-\d{14}\.csv" --weight 0\.2', command
            )

            # A start placed by hand is exactly the start the arena then gives.
            tab.get_by_role("button", name="Place a start by hand").click()
            tab.evaluate("NV.practice.place({car: [-1200, 900], yaw: 0.7, ball: [300, 2400]})")
            tab.fill("[data-field=carSpeed]", "800")
            tab.fill("[data-field=ballZ]", "400")
            tab.fill("[data-field=ballVz]", "-300")
            with tab.expect_download() as download:
                tab.get_by_role("button", name="Save this start").click()
            edited = Path(download.value.path())
            [(group, placed)] = practice.read_moments(edited.read_text(encoding="utf-8"))
            assert (
                group == "placed"
                and placed[:2] == [-1200.0, 900.0]
                and placed[32:35] == [300.0, 2400.0, 400.0]
            )
            got = start_from(config, edited)[0]
            for name, want, have in zip(layouts.MOMENT_COLUMNS, placed, got, strict=True):
                assert np.float32(have) == np.float32(want), name
        finally:
            browser.close()
    assert problems == []


def test_the_dashboards_viewer_adds_a_moment_straight_to_the_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    from nitro.dashboard.api import Api
    from test_dashboard_watch import Bridge, no_listening, opened, playing_bot
    from test_watch import chromium

    folder = playing_bot(tmp_path, 500)
    state = folder.load_latest()
    assert state is not None
    game = record(folder.load_settings(), state, seconds=4.0, seed=3, fps=30, rewards=True, moments=True)
    assert game.moments is not None
    layouts.save(folder.root / "watch.nlrec", game.data("my-bot at 500 agent steps"))
    api = Api(Path(__file__).resolve().parents[1], folder.root.parent)
    assert "error" in api.practice_add("my-bot", "not,a\nmoments,file\n")
    assert practice.count(folder) == 0, "nothing added from a bad file"
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        no_listening(monkeypatch)
        try:
            tab = opened(browser, Bridge(api), problems)
            tab.click("#main [data-bot=my-bot]")
            tab.click("[data-act=watch]")
            tab.click("[data-act=watch-recorded] >> nth=0")
            tab.wait_for_function("window.NV && NV.frames > 2 && NV.practice.ready", timeout=120_000)
            assert tab.locator(".nv-rewards-plot").is_visible(), "the reward chart shows"
            tab.evaluate("NV.player.playing = false; NV.player.seek(2)")
            tab.click(".nv-practice-open")
            tab.get_by_role("button", name="Practice this moment").click()
            tab.wait_for_function(
                "document.querySelector('.nv-practice-result p').textContent.includes('Added')"
            )
            assert "1 in all" in tab.locator(".nv-practice-result p").inner_text()
            assert tab.locator(".nv-practice-command").inner_text() == ""
        finally:
            browser.close()
    car = 0
    decision = int(np.argmin([abs(d[0] / 30 - 2.0) for d in game.decisions]))
    [(_, values)] = practice.read_moments((folder.root / practice.FILE).read_text(encoding="utf-8"))
    np.testing.assert_array_equal(np.float32(values), game.moments[decision, car])
    assert 'name = "snapshots"' in folder.settings.read_text(encoding="utf-8")
    assert problems == []
