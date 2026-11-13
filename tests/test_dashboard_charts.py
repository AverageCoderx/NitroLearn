"""The Overview's charts: every column training writes has a folder (a new one lands in Other under its name
in words), every reward of a bot gets its charts, long runs are thinned on the Python side, and the page's
folders, Your charts, the Add charts list and each chart's menu work and are kept."""

from __future__ import annotations

import dataclasses
import itertools
import threading
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro import replay_mix
from nitro.config.schema import PPOConfig
from nitro.dashboard import bots, charts
from nitro.dashboard.api import Api
from nitro.learner import signal_columns
from nitro.runs import RunFolder
from nitro.training import (
    EVEN_START_COLUMNS,
    KICKOFF_COLUMNS,
    PLAY_COLUMNS,
    SELFPLAY_COLUMNS,
    SETTER_COLUMNS,
    TIME_COLUMNS,
    TrainingRun,
    size_column_names,
)
from test_dashboard import NEEDS_MESHES, TINY, make_bot
from test_dashboard_bot import opened

REWARDS = ["touch_ball", "goal", "touch_ball_2"]
REWARD_KINDS = ("raw_", "paid_", "paid_abs_", "own_abs_", "fired_")
NETWORK_KINDS = ("grad_norm", "update_norm", "weight_norm", "turn", "growth")
BASE = [
    "agent_steps", "iteration", "steps_per_second", "reward_per_step", "episode_seconds", "entropy",
    "policy_change", "approx_kl", "clip_fraction", "explained_variance", "value_loss", "policy_loss",
    "skipped_updates", "grad_scale", "reward_scale", "policy_lr", "start_value", "start_value_share",
    "update_size",
]  # fmt: skip


def every_column() -> list[str]:
    """Every column training can write, with each optional part switched on."""
    ppo = dataclasses.replace(
        PPOConfig(),
        goal_critic_weight=1.0,
        vdag_beta=0.1,
        advantage_keep=0.5,
        advantage_keep_rest=0.5,
        entropy_target=0.5,
        head_loss_scale="popart",
    )
    nets = ("policy", "value", "shared")
    return [
        *BASE,
        *(f"{kind}{label}" for kind in REWARD_KINDS for label in REWARDS),
        *(f"{kind}_{net}" for kind in NETWORK_KINDS for net in nets),
        *TIME_COLUMNS,
        *PLAY_COLUMNS,
        *size_column_names([1, 2, 3]),
        *signal_columns(ppo),
        *replay_mix.COLUMNS,
        *replay_mix.KICKOFF_COLUMNS,
        *SELFPLAY_COLUMNS,
        *EVEN_START_COLUMNS,
        *(f"{name}_{label}" for label in ("kickoff", "random", "ball_touch") for name in SETTER_COLUMNS),
        *KICKOFF_COLUMNS,
    ]  # fmt: skip


def test_every_column_training_writes_has_a_folder_and_a_new_one_lands_in_other() -> None:
    columns = every_column()
    assert "head_loss_weight" in columns and "advantage_rows" in columns, "every learning signal on"
    found = charts.catalog([*columns, "brand_new_count"], 900.0)
    by_id = {c.id: c for c in found}
    assert [c.id for c in found if c.folder == "other"] == ["brand_new_count"]
    assert by_id["brand_new_count"].title == "Brand new count", "its name in words"
    assert not {"agent_steps", "iteration"} & set(by_id), "the axis and the count are no charts"
    assert set(by_id) == set(columns[2:]) | {f"share_{label}" for label in REWARDS} | {"brand_new_count"}
    for label in REWARDS:  # every reward: what it gives, its share, how often it pays, its details
        folders = {
            by_id[f"{kind}{label}"].folder for kind in ("raw_", "share_", "fired_", "paid_", "own_abs_")
        }
        assert folders == {"rewards", "shares", "fires", "reward_details"}
    assert by_id["raw_touch_ball_2"].title == "Touch ball 2"
    assert by_id["fired_goal"].scale == 900.0 and by_id["fired_goal"].unit == "/min"
    assert (
        by_id["2v2_goals_per_minute"].title == "2v2 goals" and by_id["2v2_goals_per_minute"].folder == "sizes"
    )
    assert by_id["goal_share_ball_touch"].title == "Ball touch starts: goals"
    assert by_id["grad_norm_shared"].title == "Gradient size: shared trunk"
    order = [c.folder for c in found]
    assert order == sorted(order, key=charts.FOLDER_IDS.index), "in the folders' order"
    for chart in found:
        assert chart.title and chart.about and chart.folder in charts.FOLDER_IDS, chart.id
    colours = [colour for _, _, colour, _ in charts.FOLDERS]
    assert set(colours) == set(range(8)), "seven colours and the neutral one, no more"


def test_long_runs_are_thinned_and_every_kept_row_is_read_right(tmp_path: Path) -> None:
    """A long file read at once and one read as it grows keep the same kinds of rows: thinned older ones
    (their numbers right) and every one of the newest."""
    rows = 30_000
    header = "agent_steps,iteration,steps_per_second,x,y\n"
    lines = [f"{(i + 1) * 100},{i + 1},1000,{i},{'' if i % 7 == 0 else i * 2}\n" for i in range(rows)]
    whole = make_bot(tmp_path / "runs", "whole")
    whole.metrics.write_text(header + "".join(lines), encoding="utf-8")
    growing = make_bot(tmp_path / "runs", "growing")
    growing.metrics.write_text(header, encoding="utf-8")
    for start in range(0, rows, 2_500):
        with growing.metrics.open("a", encoding="utf-8") as out:
            out.write("".join(lines[start : start + 2_500]))
        bots.metrics(growing)
    for folder in (whole, growing):
        m = bots.metrics(folder)
        assert m.count == rows and "x" not in m.columns, "x and y are kept for the charts only"
        kept = m.rows_from(0)
        assert kept == sorted(set(kept)) and kept[-1] == rows - 1
        assert len(kept) <= bots._THIN_ROWS + 2 * bots._TAIL_ROWS
        assert kept[-bots._TAIL_ROWS :] == list(range(rows - bots._TAIL_ROWS, rows)), "every newest row"
        assert all(m.value("x", r) == r for r in kept)
        assert all(m.value("y", r) == (None if r % 7 == 0 else r * 2) for r in kept), "empty cells stay empty"
        assert m.rows_from(rows - 10) == list(range(rows - 10, rows))
    view = charts.view(whole, 0, ["x", "steps_per_second"], points=50)
    xs = view["series"]["x"]
    assert len(xs) == 50 and xs[-1] == [rows * 100, rows - 1], "50 points, the newest as it is"
    assert all(a[0] < b[0] for a, b in itertools.pairwise(xs)), "in order of steps"
    assert view["series"]["steps_per_second"][0][1] == 1000


def test_the_window_reading_a_growing_file_from_several_calls_at_once_counts_each_row_once(
    tmp_path: Path,
) -> None:
    """The window's calls run in threads of their own (the charts' while the page refreshes): two reading on
    from the same place used to add the new rows twice."""
    folder = make_bot(tmp_path / "runs", "racer")
    folder.metrics.write_text("agent_steps,iteration,steps_per_second,x\n", encoding="utf-8")
    rows = 0
    for _ in range(30):
        with folder.metrics.open("a", encoding="utf-8") as out:
            out.write("".join(f"{(rows + i + 1) * 10},{rows + i + 1},5,{i}\n" for i in range(300)))
        rows += 300
        gate = threading.Barrier(6)

        def read(gate: threading.Barrier = gate) -> None:
            gate.wait()
            bots.metrics(folder)

        readers = [threading.Thread(target=read) for _ in range(6)]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        assert bots.metrics(folder).count == rows


def test_a_file_written_anew_with_as_many_rows_shows_its_own_numbers(tmp_path: Path) -> None:
    """A resume writes metrics.csv anew (rows past its save dropped, then new ones): with as many rows as
    before, the charts used to keep the old file's points."""
    folder = make_bot(tmp_path / "runs", "racer")
    header = "agent_steps,iteration,steps_per_second,x\n"
    folder.metrics.write_text(
        header + "".join(f"{i * 10},{i},5,{i}\n" for i in range(1, 6)), encoding="utf-8"
    )
    assert charts.view(folder, 0, ["x"])["series"]["x"][-1] == [50, 5]
    again = folder.metrics.with_name("again.csv")
    again.write_text(header + "".join(f"{i * 10},{i},5,{-i}\n" for i in range(1, 6)), encoding="utf-8")
    again.replace(folder.metrics)
    assert charts.view(folder, 0, ["x"])["series"]["x"] == [[i * 10, -i] for i in range(1, 6)]


def test_a_column_written_now_and_then_shows_its_newest_number_as_it_is(tmp_path: Path) -> None:
    """A column blank in most rows (the flip resets' height, only when there were some): its "now" is the
    newest row that holds it, not an average with the rows before."""
    folder = make_bot(tmp_path / "runs", "racer")
    lines = [f"{i * 10},{i},5,{i if i % 50 == 0 else ''}\n" for i in range(1, 1011)]
    folder.metrics.write_text("agent_steps,iteration,steps_per_second,h\n" + "".join(lines), encoding="utf-8")
    points = charts.view(folder, 0, ["h"], points=10)["series"]["h"]
    assert points[-1] == [10_000, 1000]
    assert all(a[0] < b[0] for a, b in itertools.pairwise(points))


@NEEDS_MESHES
def test_a_real_training_run_is_charted_whole(tmp_path: Path) -> None:
    """A short CPU training with several rewards, two ways of starting and self-play: every column it
    writes has its folder, and every reward of the bot its charts."""
    settings = TINY.replace("num_arenas = 2", "num_arenas = 4").replace(
        '[[state_setters]]\nname = "kickoff"\n',
        '[[state_setters]]\nname = "kickoff"\n\n[[state_setters]]\nname = "random"\n\n'
        '[[rewards]]\nname = "touch_ball"\nweight = 1\n\n[[rewards]]\nname = "goal"\nweight = 10\n\n'
        '[[rewards]]\nname = "velocity_player_to_ball"\nweight = 0.5\n\n'
        "[selfplay]\nenabled = true\npast_share = 0.5\nactive_opponents = 1\n",
    )
    folder = make_bot(tmp_path / "runs", "real", settings)
    TrainingRun(folder.load_settings(), folder).run(threading.Event(), until_steps=960)
    header = bots.metrics(folder).header
    labels = charts.reward_labels(header)
    assert labels == ["touch_ball", "goal", "velocity_player_to_ball"]
    api = Api(tmp_path / "project", tmp_path / "runs")
    out = api.bot_charts("real", "all")
    by_id = {c["id"]: c for c in out["charts"]}
    assert not [c for c in out["charts"] if c["folder"] == "other"], "every column has its folder"
    assert set(header) - {"agent_steps", "iteration"} <= set(by_id)
    for label in labels:
        for kind, folder_id in (("raw_", "rewards"), ("share_", "shares"), ("fired_", "fires")):
            assert by_id[kind + label]["folder"] == folder_id
    assert {"rating_mu", "goal_share_kickoff", "touches_per_minute"} <= set(by_id)
    assert out["default"][:3] == ["reward_per_step", "steps_per_second", "touches_per_minute"]
    assert "rating_mu" in out["default"], "self-play's rating is in the starting set"
    assert out["rows"] == bots.metrics(folder).count >= 3
    assert all(out["series"][i] for i in ("reward_per_step", "entropy", "share_goal"))
    shares = [out["series"][f"share_{label}"][-1][1] for label in labels]
    assert sum(shares) == pytest.approx(100, abs=0.01), "the shares of the newest round add up"


def chart_metrics(folder: RunFolder, rows: int = 40) -> None:
    header = (
        "agent_steps,iteration,steps_per_second,reward_per_step,raw_goal,paid_abs_goal,touches_per_minute,"
        "entropy"
    )
    lines = [
        f"{i * 1000},{i},500,{0.01 * i},{0.001 * i},{0.5},{2 + i * 0.1},{1 - i * 0.01}"
        for i in range(1, rows)
    ]
    folder.metrics.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def colour_of(tab: Any, selector: str) -> str:
    return tab.locator(selector).first.evaluate("el => getComputedStyle(el).borderTopColor")


def test_charts_are_added_moved_removed_and_kept(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    chart_metrics(make_bot(runs, "racer", saves=1))
    api = Api(tmp_path / "project", runs)
    errors: list[str] = []
    yours = "[data-folder=yours] .chart"
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1366, "height": 768})
            open_bot(tab, "racer")
            tab.wait_for_selector(f"{yours} canvas")
            pinned = tab.eval_on_selector_all(yours, "els => els.map(e => e.dataset.chartId)")
            assert pinned == ["reward_per_step", "steps_per_second", "touches_per_minute", "entropy"]
            assert tab.locator("[data-folder=rewards] .chart").count() == 0, "a closed folder draws nothing"

            # A folder opens (and is kept open), its charts in its colour; one is pinned with its pin.
            tab.click("[data-chfold=rewards]")
            tab.wait_for_selector("[data-folder=rewards] .chart[data-chart-id=raw_goal] canvas")
            blue = tab.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--c1').trim()")
            assert colour_of(tab, "[data-folder=rewards] .chart") == rgb(blue), "the folder's colour"
            assert colour_of(tab, f"{yours}[data-chart-id=steps_per_second]") != rgb(blue)
            tab.click("[data-folder=rewards] [data-chpin=raw_goal]")
            tab.wait_for_selector(f"{yours}[data-chart-id=raw_goal]")
            assert api.home()["prefs"]["charts"]["pinned"][-1] == "raw_goal"
            assert api.home()["prefs"]["charts"]["open"] == ["yours", "rewards"]

            # The Add charts list: searched, a click adds one and another click takes it out.
            tab.click("[data-act=ch-add]")
            tab.fill("#ch-search", "share")
            tab.wait_for_selector("#ch-pick-share_goal")
            assert tab.locator(".ch-pick").count() == 1, "only what matches"
            tab.click("#ch-pick-share_goal")
            tab.wait_for_selector("#ch-pick-share_goal[aria-pressed=true]")
            tab.fill("#ch-search", "randomness")
            tab.click("#ch-pick-entropy")
            tab.wait_for_selector("#ch-pick-entropy[aria-pressed=false]")
            tab.click(".modal [data-close]")
            tab.wait_for_selector(f"{yours}[data-chart-id=share_goal] canvas")
            assert tab.locator(f"{yours}[data-chart-id=entropy]").count() == 0

            # The chart's menu moves it to the start, from the keyboard too.
            tab.click(f"{yours}[data-chart-id=share_goal] [data-chmenu]")
            tab.click(".ch-menu [data-by='-999']")
            tab.wait_for_function(f"document.querySelector('{yours}').dataset.chartId === 'share_goal'")
            tab.keyboard.press("Escape")
            tab.wait_for_selector(".ch-menu", state="detached")
            canvas = f"{yours}[data-chart-id=reward_per_step] canvas"
            tab.focus(canvas)
            tab.keyboard.press("Home")
            read = f"{yours}[data-chart-id=reward_per_step] [data-read]"
            assert tab.text_content(read) == "0.0100 at 1.0K steps"
            tab.keyboard.press("ArrowRight")
            assert tab.text_content(read) == "0.0200 at 2.0K steps"
            tab.click(f"{yours}[data-chart-id=raw_goal] [data-chmenu]")
            tab.click(".ch-menu [data-chpin=raw_goal]")
            tab.wait_for_selector(f"{yours}[data-chart-id=raw_goal]", state="detached")
            tab.click("[data-seg=chsmooth] [data-v='1']")
            tab.click("[data-chfold=yours]")
            tab.wait_for_selector(yours, state="detached")
            tab.close()

            # All of it is kept: a new window shows the same.
            again = opened(browser, api, errors, viewport={"width": 900, "height": 640})
            open_bot(again, "racer")
            again.wait_for_selector("[data-folder=rewards] .chart canvas")
            assert again.locator(yours).count() == 0, "Your charts stays closed"
            again.click("[data-chfold=yours]")
            again.wait_for_selector(f"{yours} canvas")
            pinned = again.eval_on_selector_all(yours, "els => els.map(e => e.dataset.chartId)")
            assert pinned == ["share_goal", "reward_per_step", "steps_per_second", "touches_per_minute"]
            assert again.get_attribute("[data-seg=chsmooth] [data-v='1']", "aria-pressed") == "true"
            assert again.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (
                "no sideways scroll"
            )
            inside = """els => els.every(e => {
                const b = e.getBoundingClientRect();
                return b.left >= 0 && b.right <= innerWidth;
            })"""
            assert again.eval_on_selector_all(".chart, .ch-head", inside)
            again.click("[data-chfold=speed]")
            again.wait_for_selector("[data-folder=speed] .chart canvas")
            again.click("[data-folder=yours] [data-act=ch-add]")
            again.click(".modal [data-chdefault]")
            again.click(".modal [data-close]")
            pinned = again.eval_on_selector_all(yours, "els => els.map(e => e.dataset.chartId)")
            assert pinned == ["reward_per_step", "steps_per_second", "touches_per_minute", "entropy"], "reset"
        finally:
            browser.close()
    assert errors == []


def rgb(hex_colour: str) -> str:
    value = hex_colour.lstrip("#")
    return "rgb({}, {}, {})".format(*(int(value[i : i + 2], 16) for i in (0, 2, 4)))


def some_metrics(folder: RunFolder, columns: list[str], rows: int) -> None:
    """`rows` rows of 1000 steps each: reward_per_step a hundredth of the row's number, every other column
    a thousandth."""
    header = ["agent_steps", "iteration", "steps_per_second", "reward_per_step", *columns]
    lines = [
        ",".join(map(str, [i * 1000, i, 500, 0.01 * i] + [0.001 * i] * len(columns))) for i in range(1, rows)
    ]
    folder.metrics.write_text(",".join(header) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def test_charts_keep_other_bots_pins_the_point_read_and_a_choice_made_while_the_garage_refreshes(
    tmp_path: Path,
) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    goals = make_bot(runs, "goals", saves=1)
    some_metrics(goals, ["entropy", "raw_goal", "paid_abs_goal"], 600)
    rated = make_bot(runs, "rated", saves=1)
    some_metrics(rated, ["entropy", "rating_mu", "raw_demo", "paid_abs_demo"], 40)
    api = Api(tmp_path / "project", runs)
    errors: list[str] = []
    yours = "[data-folder=yours] .chart"

    def pinned(tab: Any) -> list[str]:
        return list(tab.eval_on_selector_all(yours, "els => els.map(e => e.dataset.chartId)"))

    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1366, "height": 768})
            open_bot(tab, "goals")
            tab.wait_for_selector(f"{yours} canvas")
            tab.click("[data-chfold=rewards]")
            tab.click("[data-folder=rewards] [data-chpin=raw_goal]")
            tab.wait_for_selector(f"{yours}[data-chart-id=raw_goal]")

            # Moving a chart on a bot without the goal reward keeps the goal's chart for the bot that has it.
            open_bot(tab, "rated")
            tab.wait_for_selector(f"{yours}[data-chart-id=entropy] canvas")
            tab.click(f"{yours}[data-chart-id=entropy] [data-chmenu]")
            tab.click(".ch-menu [data-by='-999']")
            tab.wait_for_function(f"document.querySelector('{yours}').dataset.chartId === 'entropy'")
            assert api.home()["prefs"]["charts"]["pinned"] == [
                "entropy", "reward_per_step", "steps_per_second", "raw_goal"
            ]  # fmt: skip

            # Back to the starting set: each bot's own (the rated one's has its rating).
            tab.click("[data-folder=yours] [data-act=ch-add]")
            tab.click(".modal [data-chdefault]")
            tab.click(".modal [data-close]")
            assert pinned(tab) == ["reward_per_step", "steps_per_second", "rating_mu", "entropy"]
            open_bot(tab, "goals")
            tab.wait_for_selector(f"{yours} canvas")
            assert pinned(tab) == ["reward_per_step", "steps_per_second", "entropy"]

            # A pin made while a Garage refresh is on its way stays when that refresh's older answer comes.
            tab.evaluate("""() => {
                const real = window.__testApi;
                window.__testApi = new Proxy({}, {get: (t, m) => m !== "home" ? real[m] : async (...a) => {
                    const out = await real.home(...a);
                    await new Promise(r => setTimeout(r, 700));
                    return out;
                }});
                window.__homeDone = loadHome();
            }""")
            tab.wait_for_timeout(200)
            tab.click("[data-folder=rewards] [data-chpin=raw_goal]")
            tab.wait_for_selector(f"{yours}[data-chart-id=raw_goal]")
            tab.evaluate("window.__homeDone.then(() => TAB_REFRESH.overview())")
            tab.wait_for_timeout(700)
            tab.click("[data-chfold=shares]")
            tab.click("[data-folder=shares] [data-chpin=share_goal]")
            tab.wait_for_selector(f"{yours}[data-chart-id=share_goal]")
            assert pinned(tab)[-2:] == ["raw_goal", "share_goal"]
            assert api.home()["prefs"]["charts"]["pinned"][-2:] == ["raw_goal", "share_goal"]

            # The point read stays on its step when new numbers come in (the points move along).
            canvas = f"{yours}[data-chart-id=reward_per_step] canvas"
            read = f"{yours}[data-chart-id=reward_per_step] [data-read]"
            tab.locator(canvas).scroll_into_view_if_needed()
            box = tab.locator(canvas).bounding_box()
            tab.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + 20)
            step = tab.evaluate("S.chHover.step")
            with goals.metrics.open("a", encoding="utf-8") as out:
                out.write("".join(f"{i * 1000},{i},500,{0.01 * i},1,1,1\n" for i in range(600, 700)))
            tab.evaluate("TAB_REFRESH.overview()")
            tab.wait_for_function("chPoints('reward_per_step').slice(-1)[0][0] === 699000")
            at = tab.evaluate("""(() => { const pts = chPoints('reward_per_step'), i = S.chHover.index;
                return [pts[i][0], chValue(pts[i][1], '') + ' at ' + steps(pts[i][0]) + ' steps']; })()""")
            assert abs(at[0] - step) < 3000 and tab.text_content(read) == at[1]
        finally:
            browser.close()
    assert errors == []
