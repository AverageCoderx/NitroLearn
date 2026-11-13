"""A bot's page: its sectors and "next" board in each state, the number-and-colours dialog, the charts' ranges
and pointer reading, the Console's command line, and the Tools: each command's plan, the rule of one heavy job
at a time, what is read from a command's output, and real runs on a tiny bot whose results reach the page."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.dashboard import app, bots, tools
from nitro.dashboard.api import Api
from nitro.dashboard.jobs import CONSOLE, Job
from nitro.runs import RunFolder
from test_dashboard import NEEDS_MESHES, ROOT, TINY, make_bot, wait_for

BOT_TOML = """[settings]
name = "Shown Name"
run_command = "nitro_bot.exe policy.nwt"
run_command_linux = "./nitro_bot policy.nwt"

[details]
description = "Trained with NitroLearn: run {name}, 1,000 agent steps"
"""


def opened(browser: Any, api: Api, errors: list[str], source: str = "", **page_options: Any) -> Any:
    """The dashboard's page (or `source`) in a new tab, talking to `api`."""
    tab = browser.new_page(**page_options)
    tab.on("pageerror", lambda e: errors.append(str(e)))
    tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
    tab.add_init_script("window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});")
    tab.goto("about:blank")
    tab.set_content(source or app.page(api))
    return tab


def train(folder: RunFolder) -> None:
    """Marks the bot as training: a marker naming a live Python process (this one)."""
    marker = {"pid": os.getpid(), "started": time.time() - 100}
    (folder.root / bots.RUNNING).write_text(json.dumps(marker), encoding="utf-8")


def minutes_of_metrics(folder: RunFolder, minutes: int) -> None:
    """A metrics file whose every row is one minute of training, the ball touches climbing."""
    rows = [f"{i * 60_000},{i},1000,{0.01 * i:.2f},{10 + 0.25 * i}" for i in range(1, minutes + 1)]
    header = "agent_steps,iteration,steps_per_second,reward_per_step,touches_per_minute\n"
    folder.metrics.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def rlbot_bot(project: Path, name: str) -> None:
    out = project / "bots" / name
    out.mkdir(parents=True)
    (out / "bot.toml").write_text(BOT_TOML.format(name=name), encoding="utf-8")
    (out / "policy.nwt").write_bytes(b"")


def real_saves(folder: RunFolder, steps: tuple[int, ...] = (1000, 2000)) -> None:
    """Saves a command can load: untrained networks of the bot's sizes."""
    import torch

    from nitro.learner import Learner
    from nitro.model import ActorCritic
    from nitro.runs import all_settings, fixed_settings

    config = folder.load_settings()
    for seed, agent_steps in enumerate(steps):
        torch.manual_seed(seed)
        model = ActorCritic(148, 90, config.model)
        state = {
            "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.empty(0, dtype=torch.uint8),
            "checkpoint": 1,
            "agent_steps": agent_steps,
            "iterations": 1,
            "settings": fixed_settings(config),
            "all_settings": all_settings(config),
        }
        folder.save_checkpoint(state, agent_steps, keep=5, milestone_steps=0)


def test_the_sectors_facts_come_from_the_bots_folder(tmp_path: Path) -> None:
    project, runs = tmp_path / "project", tmp_path / "runs"
    folder = make_bot(runs, "racer", saves=2)
    api = Api(project, runs)
    facts = api.bot_tools("racer")
    assert facts["recorded"] is None and facts["rlbot"] is None and facts["trend"] is None
    assert facts["saves"] == [2000, 1000] and facts["jobs"] == {} and facts["heavy"] is None
    assert facts["network"] == {"shared": [], "policy": [16], "value": [16], "problem": ""}

    (folder.root / "watch.nlrec").write_bytes(b"")
    rlbot_bot(project, "racer")
    facts = api.bot_tools("racer")
    assert facts["recorded"]["at"] > 0, "a recorded game: the recording `nitro watch` leaves in the folder"
    assert facts["rlbot"]["name"] == "Shown Name" and facts["rlbot"]["steps"] == 1000
    assert facts["rlbot"]["folder"] == str(project / "bots" / "racer") and not facts["rlbot"]["program"]
    (project / "bots" / "racer" / "nitro_bot").write_bytes(b"")
    assert api.bot_tools("racer")["rlbot"]["program"], "the bot program its bot.toml names is there"
    assert "no bot called" in api.bot_tools("ghost")["error"]


def test_the_trend_and_the_ranges_count_training_time(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer", saves=1)
    api = Api(tmp_path / "project", runs)
    minutes_of_metrics(folder, 40)
    assert api.bot_tools("racer")["trend"] is None, "under 70 minutes: nothing to compare with an hour ago"
    short = api.bot_charts("racer", "hour")
    assert short["seconds"] == pytest.approx(39 * 60), "all there is, and how much that is"

    minutes_of_metrics(folder, 80)
    trend = api.bot_tools("racer")["trend"]
    assert trend["now"] == pytest.approx(10 + 0.25 * 75) and trend["before"] == pytest.approx(
        10 + 0.25 * 14.5
    )
    hour, ten = api.bot_charts("racer", "hour"), api.bot_charts("racer", "minutes")
    assert hour["seconds"] == pytest.approx(3600) and ten["seconds"] == pytest.approx(600)
    touches = hour["series"]["touches_per_minute"]
    assert len(touches) == 61 and touches[0][0] == 20 * 60_000 and touches[-1][0] == 80 * 60_000
    assert len(ten["series"]["reward_per_step"]) == 11
    everything = api.bot_charts("racer", "all", ["reward_per_step"])
    assert everything["span"] == "all" and len(everything["series"]["reward_per_step"]) == 80
    assert list(everything["series"]) == ["reward_per_step"], "only the charts asked for"


def test_the_sectors_and_the_next_step_for_each_state(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    project, runs = tmp_path / "project", tmp_path / "runs"
    make_bot(runs, "fresh")
    training = make_bot(runs, "training", saves=1)
    train(training)
    first_minutes = make_bot(runs, "first-minutes")
    train(first_minutes)
    make_bot(runs, "pending", saves=1)
    make_bot(runs, "broken").settings.write_text('[run]\nseed = "three"\n', encoding="utf-8")
    recorded = make_bot(runs, "recorded", saves=1)
    (recorded.root / "watch.html").write_text("a game", encoding="utf-8")
    exported = make_bot(runs, "exported", saves=1)
    (exported.root / "watch.html").write_text("a game", encoding="utf-8")
    rlbot_bot(project, "exported")
    steady = make_bot(runs, "steady", saves=1)
    (steady.root / "watch.html").write_text("a game", encoding="utf-8")
    rlbot_bot(project, "steady")
    minutes_of_metrics(steady, 80)
    train(steady)
    api = Api(project, runs)
    api._rlbot_ready = lambda: True  # type: ignore[method-assign]
    assert api.stage("pending", "ppo.policy_lr", "0.0002")["pending"]
    errors: list[str] = []

    def shown(tab: Any, name: str) -> tuple[list[str | None], str, str]:
        open_bot(tab, name)
        tab.wait_for_selector(f"#bothead h1:has-text('{name}')")
        states = [tab.get_attribute(f"[data-sector='{n}']", "data-state") for n in (1, 2, 3, 4)]
        button = tab.locator("#bothead .board .btn")
        return states, tab.text_content("#bothead .board p"), button.text_content() if button.count() else ""

    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            states, text, button = shown(tab, "fresh")
            assert states == ["done", "todo", "todo", "todo"] and "has not trained yet" in text
            assert button == "Start training from scratch"
            assert tab.get_attribute("#bothead .board .btn", "data-act") == "start"
            assert "not started yet" in tab.text_content("[data-sector='2']")
            assert tab.text_content("#bothead .fact") == "No save yet"

            states, text, button = shown(tab, "training")
            assert states == ["done", "now", "todo", "todo"] and button == "Watch it play"
            assert text == "Watch a match to see what it has learned."
            assert tab.text_content("#bothead .tag") == "Training"
            assert tab.locator("#bothead .actions [data-act=stop]").count() == 1

            states, text, button = shown(tab, "first-minutes")
            assert states[1] == "now" and "first save comes 15 minutes into training by default" in text
            assert button == "", "nothing to press yet"

            states, text, button = shown(tab, "pending")
            assert button == "Apply your 1 change" and text == "1 change waits until you apply it."
            tab.click("#bothead .board .btn")
            tab.wait_for_selector(".modal [data-act=apply-go]")
            tab.click(".modal [data-close]")

            states, text, button = shown(tab, "broken")
            assert button == "Fix its settings" and "mistake" in text
            assert tab.locator("#bothead .board.waits").count() == 1, "hazard-marked: it waits for the user"
            tab.click("#bothead .board .btn")
            tab.wait_for_selector("[data-tab=settings][aria-selected=true]")

            states, text, button = shown(tab, "recorded")
            assert states == ["done", "stopped", "done", "todo"] and button == "Make RLBot bot"
            assert "Sector 2 · stopped" in tab.text_content("[data-sector='2']")
            assert text == "Make its RLBot bot to play it in a real match."
            assert "last recorded just now" in tab.text_content("[data-sector='3']")

            states, text, button = shown(tab, "exported")
            assert states == ["done", "stopped", "done", "done"] and button == "Carry on training"
            assert "holds 1.0K steps" in tab.text_content("[data-sector='4']")
            assert "It is stopped at 1.0K steps" in text

            states, text, button = shown(tab, "steady")
            assert states == ["done", "now", "done", "done"] and button == "", (
                "a quiet fact, nothing to press"
            )
            assert (
                text == "It touches the ball 28.8 times a minute now, up from 13.6 an hour of training ago."
            )
            assert tab.text_content("#bothead .fact") == "1,000 steps a second"

            figures = tab.locator("#bothead .figures dt").all_text_contents()
            assert figures[:5] == ["Game size", "Network", "Observation", "Tick skip", "Start"]
            assert {"Agent steps", "Speed", "This session", "Saves"} <= set(figures)
            assert tab.text_content("[data-start-chip]") == "Started from scratch"

            api._rlbot_ready = lambda: False  # type: ignore[method-assign]
            tab.close()
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            states, text, button = shown(tab, "exported")
            assert states[3] == "now", "made, but there is no RLBot to play it in yet"
            assert "RLBot is not found yet" in tab.text_content("[data-sector='4']")
            assert button == "Open RLBot" and "needs RLBot v5" in text
        finally:
            browser.close()
    assert errors == []


def test_the_number_and_colours_are_changed_in_a_dialog_with_a_preview(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "racer")
    api = Api(tmp_path / "project", runs)
    assert api.check_livery("7", "#C8321F", "#F3F1EC")["livery"]["ink"] == "#F3F1EC"
    assert "too alike" in api.check_livery("7", "#C8321F", "#C8321F")["error"]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            open_bot(tab, "racer")
            before = api.bot("racer")["livery"]
            tab.click("#bothead [data-act=livery]")
            assert tab.text_content("#lv-show .plate") == before["number"]
            tab.fill("#lv-number", "7")
            tab.wait_for_function("document.querySelector('#lv-show .plate').textContent === '7'")
            tab.fill("#lv-main", "#22B8A6")
            tab.fill("#lv-contrast", "#22B8A6")
            tab.wait_for_selector("#lv-error >> text=too alike")
            tab.click("[data-act=livery-save]")
            tab.wait_for_selector("#lv-error >> text=too alike")
            assert api.bot("racer")["livery"] == before, "refused: nothing is kept"

            tab.click(".swatch >> nth=1")  # the ready-made red on chalk
            tab.wait_for_function("document.querySelector('#lv-error').textContent === ''")
            assert tab.input_value("#lv-main") == "#C8321F" and tab.input_value("#lv-contrast") == "#F3F1EC"
            plate = tab.locator("#lv-show .plate")
            assert plate.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(200, 50, 31)"
            tab.click("[data-act=livery-save]")
            tab.wait_for_selector("#modal", state="hidden")
            kept = {"number": "07", "main": "#C8321F", "contrast": "#F3F1EC", "ink": "#F3F1EC"}
            assert api.bot("racer")["livery"] == kept
            assert tab.text_content("#bothead .plate") == "07"
        finally:
            browser.close()
    assert errors == []


def test_pointing_at_a_chart_reads_its_value_and_the_range_switches(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer", saves=1)
    minutes_of_metrics(folder, 80)
    make_bot(runs, "fresh")
    api = Api(tmp_path / "project", runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "racer")
            tab.wait_for_selector("canvas[data-chart=reward_per_step]")
            titles = tab.locator(".chart h3").all_text_contents()
            assert titles == ["All rewards together", "Training speed", "Ball touches"], "Your charts"
            card = ".chart[data-chart-id=reward_per_step]"
            assert tab.text_content(f"{card} .note").startswith("What the bot is paid per step")
            tab.wait_for_function(f"document.querySelector('{card} .last').textContent === '0.800'")
            read = f"{card} [data-read]"
            assert tab.text_content(read) == "agent steps"
            box = tab.locator("canvas[data-chart=reward_per_step]").bounding_box()
            tab.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            tab.wait_for_function(f"document.querySelector('{read}').textContent !== 'agent steps'")
            value, steps = re.fullmatch(r"([\d.]+) at ([\d.]+)M steps", tab.text_content(read)).groups()
            assert 1.8 <= float(steps) <= 2.6 and float(value) == pytest.approx(float(steps) / 6, abs=0.01)
            tab.mouse.move(box["x"] + box["width"] - 2, box["y"] + 10)
            tab.wait_for_function(f"document.querySelector('{read}').textContent === '0.800 at 4.8M steps'")
            tab.mouse.move(5, 5)
            tab.wait_for_function(f"document.querySelector('{read}').textContent === 'agent steps'")

            first = f"{card} .axis span >> nth=0"
            assert tab.text_content(first) == "60.0K"
            tab.click("[data-seg=span] [data-v=hour]")
            tab.wait_for_function("document.querySelector('.chart .axis span').textContent === '1.2M'")
            tab.click("[data-seg=span] [data-v=minutes]")
            tab.wait_for_function("document.querySelector('.chart .axis span').textContent === '4.2M'")
            assert tab.get_attribute("[data-seg=span] [data-v=minutes]", "aria-pressed") == "true"
            tab.click("[data-seg=span] [data-v=all]")
            tab.wait_for_function("document.querySelector('.chart .axis span').textContent === '60.0K'")

            open_bot(tab, "fresh")
            tab.wait_for_selector("#telemetry .empty >> text=Nothing to chart yet")
            assert tab.locator("#ov-tools .tool").count() >= 10, "the tools are there before any training"
        finally:
            browser.close()
    assert errors == []


def test_the_console_names_its_command_and_copies_its_text(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer")
    lines = [
        "$ nitro resume racer   (2026-01-02 10:00:00)",
        "first",
        "",
        "$ nitro eval racer   (2026-01-02 11:00:00)",
    ]
    text = "\n" + "\n".join(lines) + "\nlast\n"
    (folder.root / CONSOLE).write_bytes(text.encode())
    api = Api(tmp_path / "project", runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            keep = "() => { window.copied = document.activeElement.value; return true; }"
            tab.evaluate(f"window.copied = null; document.execCommand = {keep}")
            open_bot(tab, "racer")
            tab.click("[data-tab=console]")
            tab.wait_for_selector("#consoleFrom >> text=nitro eval racer")
            assert "started 2026-01-02 11:00:00" in tab.text_content("#consoleFrom")
            commands = tab.locator("#consoleText .cmd").all_text_contents()
            assert commands == [
                "$ nitro resume racer   (2026-01-02 10:00:00)",
                "$ nitro eval racer   (2026-01-02 11:00:00)",
            ]
            assert tab.text_content("#consoleText") == text
            tab.click("[data-act=console-copy]")
            tab.wait_for_selector(".toast >> text=Copied the console's text.")
            copied = tab.evaluate("window.copied")
            assert copied is None or copied == text, "through the clipboard, or through a selection"
        finally:
            browser.close()
    assert errors == []


def test_each_tools_command_is_worked_out_from_its_options(tmp_path: Path) -> None:
    project, runs = tmp_path / "project", tmp_path / "runs"
    (project / "configs" / "presets").mkdir(parents=True)
    (project / "configs" / "presets" / "starter.toml").write_text(TINY, encoding="utf-8")
    make_bot(runs, "racer", saves=3)
    make_bot(runs, "rival", TINY.replace("policy = [16]", "policy = [32]"), saves=1)
    make_bot(runs, "fresh")
    make_bot(runs, "seer", TINY.replace("[model]\n", '[model]\nencoder = "attention"\n'), saves=1)
    api = Api(project, runs)
    runs_flag = f"--runs {runs}"

    def command(bot: str, kind: str, /, **options: Any) -> str:
        out = api.tool_command(bot, kind, options)
        return str(out.get("command") or out["error"])

    # nitro eval
    assert command("racer", "eval") == f"nitro eval racer {runs_flag} --games 512 --both-ways --log"
    assert command("racer", "eval", save="2000", points=128, as_bot=True) == (
        f"nitro eval racer {runs_flag} --games 128 --vs-at 2000 --as-bot"
    )
    assert command("racer", "eval", opponent="rival").endswith("--games 512 --vs rival --both-ways --log")
    sides = api.tool_command("racer", "eval", {})
    assert (
        sides["first"] == "racer's newest save (3.0K steps)" and sides["second"] == "its save at 1.0K steps"
    )
    assert "pick one of racer's earlier saves" in command("racer", "eval", save=3000)
    assert "only one save" in command("rival", "eval")
    assert "no save yet" in command("fresh", "eval") and "no save yet" in command(
        "racer", "eval", opponent="fresh"
    )
    assert "between 8 and 100,000" in command("racer", "eval", points=2)
    # nitro skills
    assert command("racer", "skills") == f"nitro skills racer {runs_flag} --games 512"
    assert command("racer", "skills", save=1000, situations=128, as_bot=True) == (
        f"nitro skills racer {runs_flag} --games 128 --at 1000 --temperature 0"
    )
    assert "no save at 5,000 steps" in command("racer", "skills", save=5000)
    # nitro surgery
    assert command("racer", "surgery") == f"nitro surgery racer --widen 2 --as racer-x2 {runs_flag}"
    assert command("racer", "surgery", widen=3, new_name="big").startswith(
        "nitro surgery racer --widen 3 --as big"
    )
    assert "already a bot called rival" in command("racer", "surgery", new_name="rival")
    assert "only plain networks can be widened" in command("seer", "surgery")
    assert "no save yet" in command("fresh", "surgery") and "between 2 and 8" in command(
        "racer", "surgery", widen=1
    )
    assert api.bot_tools("seer")["network"]["problem"], "the page says so before the tool is opened"
    # nitro ab: a bot's own settings go in as a copy that reads the same from anywhere
    copy = runs / ".dashboard" / "compare"
    on_task = command("racer", "ab", a="bot:racer", b="bot:rival", on="ball_touch")
    assert on_task == (
        f"nitro ab {copy / 'racer.toml'} {copy / 'rival.toml'} --task ball_touch --seeds 10 --folder ab "
        "--name racer-vs-rival-ball_touch"
    )
    by_speed = command("racer", "ab", a="bot:racer", b="starter", on="speed", repeats=4, name="quick")
    assert by_speed == (
        f"nitro ab {copy / 'racer.toml'} configs/presets/starter.toml --speed --repeats 4 --folder ab "
        "--name quick"
    )
    assert "--with-rewards" in command(
        "racer", "ab", a="bot:racer", b="bot:rival", on="shot", with_rewards=True
    )
    assert "two different settings files" in command("racer", "ab", a="bot:racer", b="bot:racer", on="speed")
    assert "pick a settings file" in command("racer", "ab", a="bot:racer", on="speed")
    assert "pick what to compare them on" in command("racer", "ab", a="bot:racer", b="bot:rival", on="chess")
    assert "no settings file called" in command(
        "racer", "ab", a="bot:racer", b="file:/etc/passwd", on="speed"
    )
    assert "name uses letters" in command("racer", "ab", a="bot:racer", b="bot:rival", on="speed", name="a b")
    assert not copy.exists(), "asking for a command writes nothing"
    # nitro check-settings, export, pack
    assert command("racer", "settings") == f"nitro check-settings {copy / 'racer.toml'}"
    assert command("racer", "settings", file="starter") == "nitro check-settings configs/presets/starter.toml"
    assert command("racer", "export") == f"nitro export racer {runs_flag}"
    assert (
        command("racer", "export", bot_name="Sky Rocket")
        == f'nitro export racer {runs_flag} --bot-name "Sky Rocket"'
    )
    assert "without quotes" in command("racer", "export", bot_name='Sky "Rocket"')
    assert "no save yet" in command("fresh", "export")
    assert api.tool_command("racer", "export", {})["folder"] == str(project / "bots" / "racer")
    packed = command("racer", "pack", all_saves=True, meshes=False)
    assert re.fullmatch(
        rf"nitro pack racer {re.escape(runs_flag)} --to packs/racer-[\d-]+\.nitropack\.zip"
        + r" --all-saves --no-meshes",
        packed,
    )
    assert "no tool called" in command("racer", "polish")

    usual = Api(project, project / "runs")
    make_bot(project / "runs", "racer", saves=2)
    assert usual.tool_command("racer", "skills", {})["command"] == "nitro skills racer --games 512", (
        "the usual runs folder is not named"
    )

    choices = api.tool_choices("racer")
    assert [b["value"] for b in choices["bots"]] == ["rival", "seer"], "the other bots with a save"
    ids = [f["id"] for f in choices["files"]]
    assert ids == ["bot:fresh", "bot:racer", "bot:rival", "bot:seer", "starter"]
    assert not any(f["broken"] for f in choices["files"]) and "path" not in choices["files"][0]
    wrong = project / "my-settings" / "wrong.toml"
    wrong.parent.mkdir()
    wrong.write_text('[run]\nseed = "three"\n', encoding="utf-8")
    listed = api.tool_choices("racer")["files"][-1]
    assert listed["broken"] and listed["id"] == f"file:{wrong}" and listed["group"] == "my-settings"
    assert command("racer", "settings", file=listed["id"]) == "nitro check-settings my-settings/wrong.toml"
    assert "run.seed" in command("racer", "ab", a="bot:racer", b=listed["id"], on="speed"), (
        "said before it starts"
    )
    tasks = {t["name"]: t["about"] for t in choices["tasks"]}
    assert (
        tasks["ball_touch"] == "touch a ball at rest" and "kickoff" in tasks and choices["comparisons"] == []
    )


def test_a_bots_settings_go_to_a_command_reading_the_same_from_anywhere(tmp_path: Path) -> None:
    from nitro.config import load_config

    home = tmp_path / "home"
    (home / "meshes").mkdir(parents=True)
    settings = home / "racer.toml"
    settings.write_text('[paths]\ncollision_meshes = "meshes"  # beside this file\n[run]\nseed = 5\n')
    folder = RunFolder.create(tmp_path / "runs", "racer", settings)
    copy = tmp_path / "elsewhere" / "racer.toml"
    copy.parent.mkdir()
    copy.write_text(tools._standalone(folder), encoding="utf-8")
    assert "# beside this file" in copy.read_text() and "seed = 5" in copy.read_text()
    assert load_config(copy) == folder.load_settings(), "the folder it names is written in full"
    assert load_config(copy).paths.collision_meshes == (home / "meshes").resolve()


def test_one_heavy_job_at_a_time_and_never_beside_the_bots_training(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    racer = make_bot(runs, "racer", saves=2)
    rival = make_bot(runs, "rival", saves=2)
    api = Api(tmp_path / "project", runs)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        job = Job("racer", "skills", [], racer.root / CONSOLE, sleeper)
        api._tool_runs()[("racer", "skills")] = tools._Run(job, "nitro skills racer", 0, {})
        busy = "racer is busy with a skill check: wait until it is done, or stop it"
        assert api.tool_command("racer", "eval", {})["blocked"] == busy + " in Tools"
        assert api.tool_start("racer", "eval", {})["error"] == busy + " in Tools"
        assert api.start("racer")["error"] == busy + " in Tools", "nor its training beside it"
        assert api.try_rewards("racer")["error"] == busy + " in Tools"
        other = api.tool_command("rival", "surgery", {})["blocked"]
        assert other == busy + " (one heavy job at a time)", "nor another bot's heavy tool"
        assert api.tool_command("rival", "settings", {})["blocked"] == "", "a light one may run beside it"
        assert api.tool_command("rival", "export", {})["blocked"] == ""
        facts = api.bot_tools("rival")
        assert facts["heavy"] == {"bot": "racer", "kind": "skills", "what": "a skill check"}
        assert api.bot_tools("racer")["jobs"]["skills"]["state"] == "running"
        assert api.tool_state("racer", "skills")["state"] == "running"

        assert api.tool_stop("racer", "skills") == {"stopped": True}
        sleeper.wait()
        assert api.tool_state("racer", "skills")["state"] == "stopped", (
            "stopped from the window: not a failure"
        )
        assert api.tool_stop("racer", "skills")["error"] == "it is not running"
        assert api.tool_command("racer", "eval", {})["blocked"] == "" and api._busy("racer") == ""
    finally:
        sleeper.kill()
        sleeper.wait()

    train(rival)
    for kind, why in (
        ("eval", "a match beside training"),
        ("skills", "a check beside training"),
        ("surgery", "newest save"),
    ):
        blocked = api.tool_command("rival", kind, {})["blocked"]
        assert blocked.startswith("rival is training: stop it first") and why in blocked
    assert (
        "stop it first, so the pack holds its newest save" in api.tool_command("rival", "pack", {})["blocked"]
    )
    options = {"a": "bot:racer", "b": "bot:rival", "on": "speed"}
    assert api.tool_command("racer", "ab", options)["blocked"] == (
        "A speed comparison needs a quiet PC: stop rival first"
    )
    assert api.tool_command("racer", "ab", {**options, "on": "ball_touch"})["blocked"] == "", "a task may"
    assert api.tool_state("racer", "eval") == {"state": "none"}


EVAL_BOTH = """Playing 512 points each way: racer at 2,000 agent steps as Blue, then as Orange, against \
racer at 1,000 agent steps, from the same starts...
racer at 2,000 agent steps vs racer at 1,000 agent steps, both ways round from the same starts: 512 points \
as Blue, 512 as Orange.
  As Blue it scored 210 and let in 120; as Orange it scored 192 and let in 131; 371 ended without a goal \
(the 60 s limit).
  Its share of the goals: as Blue 64%, as Orange 59%; both ways 62% (95% interval 58% to 65%).
  Reading: The first is clearly better: even allowing for luck, it takes more than half the goals.
Added to runs/racer/evals.csv.
"""
EVAL_ONE = """Playing 64 points: racer at 2,000 agent steps (Blue) vs rival at 9,000 agent steps (Orange)...
Picking actions: Blue likeliest (as the RLBot bot), Orange likeliest (as the RLBot bot).
racer at 2,000 agent steps (Blue) vs rival at 9,000 agent steps (Orange): 64 points from kickoff.
  Blue scored 12, Orange 30; 22 ended without a goal (the 60 s limit or no touch for 10 s). \
A point lasted 31.0 s on average.
  Blue's share of the goals: 29% (95% interval 17% to 44%).
  Reading: Orange is clearly better: even allowing for luck, it takes more than half the goals.
"""
SKILLS = """Scoring racer at 2,000 agent steps on 4 skills, 512 situations each...
ball_touch    time_to_touch 1.9s (bar 3s; under the bar: has the skill); touch_rate 0.98
boost_pickup  boost_collected 6 (bar 40; below the bar: not yet)
aerial_touch  time_to_touch 6s (bar 5.6s; over the bar: not yet); touch_rate 0
shot          goal_rate 0.6 (bar 0.45; above the bar: has the skill)
Saved to runs/racer/skills.csv. The bar is what a fresh bot trained on that one skill alone reaches; it says \
whether the skill is there, not how good the bot is.
"""
AB = """A: a.toml
B: b.toml
A, seed 0 done (1 of 20); about 96 min left.
B, seed 0 done (2 of 20); about 90 min left.
Measured: time_to_touch (lower is better).
Verdict: B IS BETTER - B's results are clearly apart from A's, by more than the noise and the smallest \
difference that matters.
  Smallest difference that matters: 0.1.
"""
CHECK_BAD = """2 problems in my-settings/bad.toml:
  - line 2, run.seed: must be a whole number (like 8), but is 'three'
  - line 4, ppo.epochz: 'epochz' is not a known setting
      hint: did you mean 'epochs'?
iteration 4 of another command
"""


def test_results_are_read_from_what_the_commands_print() -> None:
    first, second = "racer's newest save (2.0K steps)", "its save at 1.0K steps"
    both = tools.read_eval(EVAL_BOTH, first, second)
    assert both is not None and (both["scored"], both["let_in"], both["no_goal"]) == (402, 251, 371)
    assert (both["points"], both["share"], both["low"], both["high"]) == (1024, 62, 58, 65)
    assert both["winner"] == "first"
    assert (
        both["headline"] == "Racer's newest save (2.0K steps) is clearly better than its save at 1.0K steps."
    )
    assert "scored 402 goals and let in 251: 62% of the goals, over 1,024 points" in both["lines"][0]
    assert "between 58% and 65%" in both["lines"][1] and "not luck" in both["lines"][1]
    one = tools.read_eval(EVAL_ONE, "racer's newest save (2.0K steps)", "rival's save at 9.0K steps")
    assert one is not None and one["winner"] == "second" and (one["scored"], one["let_in"]) == (12, 30)
    assert one["headline"].startswith("Rival's save at 9.0K steps is clearly better")
    even = EVAL_ONE.replace(
        "Orange is clearly better: even", "No clear difference with this many points: the"
    )
    neither = tools.read_eval(even, first, second)
    assert neither is not None and neither["winner"] == "neither" and "even split" in neither["lines"][1]
    nobody = "  Blue scored 0, Orange 0; 8 ended without a goal (the 3 s limit).\n"
    nobody += "  Blue's share of the goals: -.\n"
    nobody += "  Reading: Nobody scored: nothing to compare; raise --games.\n"
    empty = tools.read_eval(nobody, first, second)
    assert empty is not None and empty["winner"] == "nobody" and empty["share"] is None
    assert tools.read_eval("Traceback (most recent call last):", first, second) is None

    skills = tools.read_skills(SKILLS)
    assert skills is not None and skills["headline"] == "It has 2 of the 4 skills."
    assert [(r["skill"], r["value"], r["unit"], r["bar"], r["has"]) for r in skills["rows"]] == [
        ("Touch a ball at rest", "1.9", "s", "3", True),
        ("Collect boost", "6", "", "40", False),
        ("Touch a ball in the air", "6", "s", "5.6", False),
        ("Score in an empty net", "0.6", "", "0.45", True),
    ]
    # Each of the four words is read: above and under have the skill, below and over not yet.
    for side, has in (("above", True), ("under", True), ("below", False), ("over", False)):
        line = f"ball_touch    time_to_touch 1.9s (bar 3s; {side} the bar: has it or not)\n"
        found = tools.read_skills(line)
        assert found is not None and found["rows"][0]["has"] is has, side
    assert skills["rows"][0]["lower_is_better"] and not skills["rows"][1]["lower_is_better"]
    assert skills["who"] == "racer at 2,000 agent steps" and skills["situations"] == 512
    assert skills["note"].startswith("The bar is what a fresh bot") and tools.read_skills("nothing") is None

    verdict = tools.read_verdict(AB, "racer", "rival")
    assert verdict is not None and verdict["headline"] == "rival did clearly better than racer."
    assert (
        verdict["verdict"] == "B is better"
        and verdict["better"] == "b"
        and "clearly apart" in verdict["meaning"]
    )
    same = tools.read_verdict("Verdict: NO MEASURABLE DIFFERENCE - the difference is within the noise.\n")
    assert same is not None and same["headline"] == "No measurable difference between the two."
    assert tools.read_verdict("A: a.toml\n") is None
    assert tools.read_progress(AB) == {"done": 2, "of": 20, "minutes_left": 90}
    assert tools.read_progress("A: a.toml\n") is None

    bad = tools.read_check(CHECK_BAD)
    assert not bad["fine"] and bad["headline"] == "2 mistakes to fix." and len(bad["problems"]) == 3
    assert bad["problems"][2] == "hint: did you mean 'epochs'?", "its own lines only, not another command's"
    fine = tools.read_check(
        "x.toml: no problems found.\nNetworks: 5,227 parameters.\nNote: tick skip 8 is rare.\n"
    )
    assert (
        fine["fine"]
        and fine["networks"].startswith("Networks: 5,227")
        and fine["notes"] == ["tick skip 8 is rare."]
    )
    assert tools.read_check("")["headline"] == "The check stopped before it was done."

    built = tools.read_export(
        "Exported to bots/racer:\n  policy.nwt\nAdd this folder to RLBot v5.\n", "bots/racer"
    )
    assert built == {"folder": "bots/racer", "written": True, "program": True, "missing": ""}
    unbuilt = tools.read_export(
        "Exported to bots/racer:\nThe bot program is missing: cargo not found\n", "bots/racer"
    )
    assert unbuilt["written"] and not unbuilt["program"] and unbuilt["missing"] == "cargo not found"


def finished(api: Api, name: str, kind: str, seconds: float) -> dict[str, Any]:
    wait_for(lambda: api.tool_state(name, kind)["state"] != "running", seconds, f"the {kind} tool to end")
    state: dict[str, Any] = api.tool_state(name, kind)
    return state


def test_a_settings_check_runs_and_its_result_reaches_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer")
    api = Api(ROOT, runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "racer")
            tab.click("#ov-tools [data-tool=settings]")
            tab.wait_for_selector("#tool-command >> text=nitro check-settings")
            assert "racer's settings" in tab.text_content("#opt-file option:checked")
            tab.click("[data-act=tool-go]")
            tab.wait_for_selector("#tool-result .verdict >> text=No problems found.", timeout=120_000)
            assert "Networks: " in tab.text_content("#tool-result")
            assert "nitro check-settings" in tab.text_content("#tool-command")
            # Run again with the dialog closed: a note on the bot's page says when it is done.
            tab.click("[data-act=tool-again]")
            tab.wait_for_selector("[data-act=tool-go]:not([disabled])")
            tab.click("[data-act=tool-go]")
            tab.click("#tool-foot [data-close]")
            done = "#bothead .banner >> text=Check a settings file is done: No problems found."
            tab.wait_for_selector(done, timeout=120_000)
            tab.wait_for_selector(".toast >> text=Check a settings file of racer is done.")
            tab.wait_for_selector("#ov-tools [data-tool=settings] .tag >> text=Result")
            tab.click("#bothead .banner [data-tool=settings]")
            tab.wait_for_selector("#tool-result .verdict >> text=No problems found.")
            assert tab.locator("#bothead .banner [data-tool=settings]").count() == 0, (
                "looked at: the note goes"
            )
            assert tab.locator("#ov-tools [data-tool=settings] .tag").count() == 0
            tab.click("#tool-foot [data-close]")
            tab.click("[data-tab=console]")
            tab.wait_for_selector("#consoleText .cmd >> text=nitro check-settings")
            assert "no problems found." in tab.text_content("#consoleText")
            tab.click("[data-tab=tools]")
            assert tab.locator("#tools-page .tool").count() >= 10
        finally:
            browser.close()
    assert errors == []
    state = api.tool_state("racer", "settings")
    assert state["state"] == "done" and state["result"]["fine"] and state["result"]["problems"] == []
    assert (runs / ".dashboard" / "compare" / "racer.toml").is_file(), "the copy the command read"

    folder.settings.write_text(TINY.replace("seed = 3", 'seed = "three"'), encoding="utf-8")
    assert api.tool_start("racer", "settings", {})["started"], "a file with a mistake is checked as it is"
    state = finished(api, "racer", "settings", 120)
    assert state["state"] == "failed" and state["result"]["headline"] == "1 mistake to fix."
    assert "run.seed: must be a whole number" in state["result"]["problems"][0]


@NEEDS_MESHES
def test_the_newest_save_plays_an_older_one_and_the_result_reaches_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer", TINY.replace("no_touch_seconds = 2", "max_episode_seconds = 3"))
    real_saves(folder)
    make_bot(runs, "rival", saves=1)
    api = Api(ROOT, runs)
    started = api.tool_start("racer", "eval", {"points": 8})
    assert started["started"] and started["second"] == "its save at 1.0K steps"
    assert "busy with a test against older saves" in api.start("racer")["error"]
    assert "one heavy job at a time" in api.tool_command("rival", "surgery", {})["blocked"]
    state = finished(api, "racer", "eval", 300)
    assert state["state"] == "done", state.get("why")
    result = state["result"]
    assert result["points"] == 16 and result["scored"] + result["let_in"] + result["no_goal"] == 16
    assert result["winner"] in ("first", "second", "neither", "nobody") and result["headline"]
    assert (folder.root / "evals.csv").is_file(), "the command kept the match in the bot's folder too"
    console = (folder.root / CONSOLE).read_text(encoding="utf-8")
    assert "$ nitro eval racer" in console and "Reading: " in console
    assert api.bot_tools("racer")["jobs"]["eval"]["state"] == "done" and api._busy("racer") == ""
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "racer")
            assert tab.text_content("#ov-tools [data-tool=eval] .tag") == "Result"
            tab.click("#ov-tools [data-tool=eval]")
            tab.wait_for_selector("#tool-result .verdict")
            assert tab.text_content("#tool-result .verdict") == result["headline"]
            assert result["reading"] in tab.text_content("#tool-result")
            tab.click("[data-act=tool-again]")
            tab.wait_for_selector("#opt-opponent")
            assert tab.locator("#opt-save option").all_text_contents() == ["Its oldest save (1,000 steps)"]
            tab.wait_for_selector("#tool-command >> text=--games 512 --both-ways --log")
            tab.click("[data-seg=opt-points] [data-v='128']")
            tab.wait_for_selector("#tool-command >> text=--games 128 --both-ways --log")
        finally:
            browser.close()
    assert errors == []


@NEEDS_MESHES
def test_a_save_is_scored_on_the_practice_skills(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer")
    real_saves(folder, (1000,))
    api = Api(ROOT, runs)
    assert api.tool_start("racer", "skills", {"situations": 2})["started"]
    state = finished(api, "racer", "skills", 600)
    assert state["state"] == "done", state.get("why")
    rows = state["result"]["rows"]
    assert [row["task"] for row in rows] == [
        "ball_touch",
        "recovery",
        "boost_pickup",
        "shot",
        "save",
        "aerial_touch",
    ]
    assert all(float(row["value"]) >= 0 and isinstance(row["has"], bool) for row in rows)
    assert state["result"]["situations"] == 2 and state["result"]["who"] == "racer at 1,000 agent steps"
    assert (folder.root / "skills.csv").is_file()


def test_the_tools_are_commands_for_the_search_box_when_the_page_has_one(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "racer", saves=2)
    rival = make_bot(runs, "rival", saves=1)
    api = Api(tmp_path / "project", runs)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job = Job("rival", "skills", [], rival.root / CONSOLE, sleeper)
    api._tool_runs()[("rival", "skills")] = tools._Run(job, "nitro skills rival", 0, {})
    source = app.page(api)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, source)
            tab.wait_for_function("COMMANDS.length > 0")
            tab.wait_for_function("COMMANDS.some(c => c.id === 'tool-eval')")
            listed = tab.evaluate("COMMANDS.map(c => [c.id, c.label, c.command, c.when ? c.when() : true])")
            assert ["tool-eval", "Test against older saves", "nitro eval", False] in listed, "no bot is open"
            assert {
                "tool-skills",
                "tool-ab",
                "tool-surgery",
                "tool-settings",
            } <= {entry[0] for entry in listed}
            assert {"export", "pack"} <= {entry[0] for entry in listed}, "listed once, under the search's own"
            open_bot(tab, "racer")
            tab.wait_for_selector("#ov-tools .tool")
            assert tab.evaluate("COMMANDS.find(c => c.id === 'tool-surgery').when()")
            assert not tab.evaluate("COMMANDS.find(c => c.id === 'wandb').when()"), "it has no W&B page"
            tab.evaluate("COMMANDS.find(c => c.id === 'tool-surgery').run()")
            tab.wait_for_selector("#tool-command >> text=nitro surgery racer --widen 2 --as racer-x2")
            assert "racer itself is not changed" in tab.text_content("#tool-body")
            tab.click("[data-seg=opt-widen] [data-v='3']")
            tab.wait_for_selector("#tool-command >> text=--widen 3 --as racer-x3")
            assert "16 becomes 48" in re.sub(r"\s+", " ", tab.text_content("#tool-body .sizes"))
            tab.click("#opt-new_name")
            tab.keyboard.type("big-one")
            tab.wait_for_selector("#tool-command >> text=--widen 3 --as big-one")
            assert tab.evaluate("document.activeElement.id") == "opt-new_name", "typing is never cut short"
            assert tab.input_value("#opt-new_name") == "big-one" and "big-one" in tab.text_content(
                ".tool-about"
            )
            tab.fill("#opt-new_name", "")
            tab.wait_for_selector("#tool-command >> text=--widen 3 --as racer-x3")
            # Another bot's heavy tool is running: the page says who waits for what, and Start stays off.
            waits = "rival is busy with a skill check: wait until it is done, or stop it (one heavy job at a "
            waits += "time)."
            assert tab.text_content("#tool-cannot") == waits and tab.is_disabled("[data-act=tool-go]")
            tab.click("#tool-foot [data-close]")
            assert "rival is busy with a skill check" in tab.text_content("#ov-tools [data-heavy]")
        finally:
            browser.close()
            sleeper.kill()
            sleeper.wait()
    assert errors == []


def test_the_bots_page_fits_a_small_window(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    project, runs = tmp_path / "project", tmp_path / "runs"
    folder = make_bot(runs, "a-bot-with-quite-a-long-name", saves=2)
    minutes_of_metrics(folder, 80)
    train(folder)
    make_bot(runs, "rival", saves=1)
    api = Api(project, runs)
    assert api.stage("a-bot-with-quite-a-long-name", "ppo.policy_lr", "0.0002")["pending"]
    errors: list[str] = []
    fits = "document.documentElement.scrollWidth <= window.innerWidth"
    inside = """sel => [...document.querySelectorAll(sel)].every(el => {
        const box = el.getBoundingClientRect();
        return box.width > 0 && box.left >= 0 && box.right <= window.innerWidth + 0.5;
    })"""
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            for theme in ("dark", "light"):
                api.set_pref("theme", theme)
                tab = opened(browser, api, errors, viewport={"width": 900, "height": 640})
                open_bot(tab, "a-bot-with-quite-a-long-name")
                tab.wait_for_selector("#telemetry canvas[data-chart]")
                assert tab.evaluate(fits), "no sideways scrolling"
                for part in ("#bothead .actions .btn", "#bothead .figures > div", "#bothead .sector"):
                    assert tab.evaluate(inside, part), part
                assert tab.evaluate(inside, "#bothead .board, .chart, #ov-tools .tool")
                actions = [tab.locator("#bothead .actions .btn").nth(i).bounding_box() for i in range(3)]
                name = tab.locator("#bothead h1").bounding_box()
                for box in actions:
                    apart = box["y"] >= name["y"] + name["height"] or box["x"] >= name["x"] + name["width"]
                    assert apart, "the actions never lie over the name"
                tools_top = tab.locator("#ov-tools").bounding_box()["y"]
                charts = tab.locator(".charts").bounding_box()
                assert tools_top >= charts["y"] + charts["height"], "the tools come below the charts"
                for tab_name in ("tools", "console", "notes"):
                    tab.click(f"[data-tab={tab_name}]")
                    assert tab.evaluate(fits), tab_name
                tab.click("[data-tab=tools]")
                for tool in ("eval", "skills", "ab", "surgery", "settings", "export", "pack"):
                    tab.click(f"#tools-page [data-tool={tool}]")
                    tab.wait_for_selector("#tool-foot .btn.primary")
                    tab.wait_for_function("document.querySelector('#tool-command, #tool-cannot') !== null")
                    assert tab.evaluate(fits) and tab.evaluate(inside, ".modal"), tool
                    modal = tab.locator(".modal").bounding_box()
                    assert modal["y"] >= 0 and modal["y"] + modal["height"] <= 640, tool
                    tab.click("#tool-foot [data-close]")
                tab.click("#bothead [data-act=livery]")
                assert tab.evaluate(fits) and tab.evaluate(inside, ".modal, .swatch")
                tab.close()
        finally:
            browser.close()
    assert errors == []
