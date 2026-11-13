"""What a first-time user sees in the states a review found unclear: a bot asked to stop, waiting changes
thrown away, rewards being tried, a bot whose first round of training is not done yet, and a PC without a
graphics card."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro import reward_check
from nitro.dashboard import bots, garage, machine
from nitro.dashboard.api import Api
from nitro.dashboard.jobs import CONSOLE, Job
from nitro.runs import RunFolder
from test_dashboard import make_bot
from test_dashboard_bot import minutes_of_metrics, opened


def running(folder: RunFolder, started: float) -> None:
    """Marks the bot as training since `started`: this test's own process stands in for its trainer."""
    marker = {"pid": os.getpid(), "started": started}
    (folder.root / bots.RUNNING).write_text(json.dumps(marker), encoding="utf-8")


@pytest.fixture
def no_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """This PC as one without a graphics card training can use."""
    monkeypatch.setattr(machine, "graphics_card", lambda: False)


def test_the_graphics_card_is_read_without_loading_pytorch(tmp_path: Path) -> None:
    built = tmp_path / "version.py"
    built.write_text("__version__ = '2.5.1+cu124'\ncuda: Optional[str] = '12.4'\n", encoding="utf-8")
    assert machine.torch_cuda(built) is True
    built.write_text("__version__ = '2.5.1+cpu'\ncuda: Optional[str] = None\n", encoding="utf-8")
    assert machine.torch_cuda(built) is False
    assert machine.torch_cuda(tmp_path / "missing.py") is None
    facts = machine.facts()
    assert set(facts) == {"graphics_card", "windows"} and facts["windows"] == (sys.platform == "win32")


def test_stop_and_save_shows_it_is_stopping_until_the_trainer_ends(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer", saves=1)
    minutes_of_metrics(folder, 10)
    running(folder, time.time() - 100)
    api = Api(tmp_path / "project", runs)
    assert "stopping" not in api.bot("racer")
    assert api.stop("racer") == {"stopping": True}
    assert (folder.root / bots.STOP_REQUEST).read_text(encoding="utf-8").strip() == "stop"
    assert api.bot("racer")["stopping"] is True and api.home()["bots"][0]["stopping"] is True

    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            hero = tab.locator(".ontrack")
            hero.locator(".tag:has-text('Stopping')").wait_for()
            assert tab.locator("#gh-stop").is_disabled() and "Stopping…" in tab.text_content("#gh-stop")
            open_bot(tab, "racer")
            tab.wait_for_selector("#bothead h1:has-text('racer')")
            assert tab.text_content("#bothead .tag") == "Stopping"
            assert tab.text_content("#bothead .fact") == "Saving after this step, then it stops"
            stop = tab.locator("#bothead [data-act=stop]")
            assert stop.is_disabled() and stop.text_content() == "Stopping…"
            assert "then it stops" in tab.text_content("#bothead .board p")
        finally:
            browser.close()
    assert errors == []

    running(folder, time.time() + 1)  # trained again since (a new trainer): not the one asked to stop
    assert "stopping" not in api.bot("racer")
    (folder.root / bots.RUNNING).unlink()
    assert api.bot("racer")["status"] == "stopped" and "stopping" not in api.bot("racer")


def test_before_its_first_round_a_training_bot_says_it_waits_for_its_numbers(
    tmp_path: Path, no_card: None
) -> None:
    runs = tmp_path / "runs"
    fresh = make_bot(runs, "fresh", saves=1)
    minutes_of_metrics(fresh, 10)  # numbers of an earlier session
    running(fresh, time.time() + 2)
    api = Api(tmp_path / "project", runs)
    assert api.bot("fresh")["counted"] is False
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            hero = tab.locator(".ontrack")
            hero.locator(".tag:has-text('Warming up')").wait_for()
            assert "On track" not in hero.text_content()
            assert hero.locator("#gh-stats .g-fig").nth(1).text_content() == "\u2013steps a second"
            open_bot(tab, "fresh")
            tab.wait_for_selector("#bothead h1:has-text('fresh')")
            fact = tab.text_content("#bothead .fact")
            assert (
                fact
                == "Waiting for its first round of training (it can take several minutes on the processor)"
            )
            speed = tab.locator("#bothead .figures div:has(dt:text-is('Speed')) dd")
            assert speed.text_content() == "\u2013", "the speed of an earlier session is not this one's"
        finally:
            browser.close()
    assert errors == []
    fresh.metrics.write_text(fresh.metrics.read_text(encoding="utf-8"), encoding="utf-8")
    os.utime(fresh.metrics, (time.time() + 5, time.time() + 5))
    assert api.bot("fresh")["counted"] is True


def test_discard_asks_first_and_names_what_goes(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_bot(runs, "racer", saves=1)
    api = Api(tmp_path / "project", runs)
    assert api.stage("racer", "ppo.policy_lr", "0.0002")["pending"]
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "racer")
            tab.click("#pending [data-act=discard]")
            tab.wait_for_selector(".modal h2:has-text('Discard 1 change?')")
            assert "Learning rate" in tab.text_content(".modal .apply-list")
            assert "cannot be undone" in tab.text_content(".modal")
            tab.click(".modal [data-close]")
            assert api.bot("racer")["pending"] == {"ppo.policy_lr": 0.0002}, "Keep them keeps them"
            assert tab.locator("#pending").is_visible()
            tab.click("#pending [data-act=discard]")
            tab.click(".modal [data-act=discard-go]")
            tab.wait_for_selector("#pending", state="hidden")
            assert api.bot("racer")["pending"] == {}
        finally:
            browser.close()
    assert errors == []


def test_trying_rewards_can_be_stopped_and_says_how_long_it_takes(tmp_path: Path, no_card: None) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer", saves=1)
    minutes_of_metrics(folder, 10)  # 1,000 steps a second
    api = Api(tmp_path / "project", runs)
    checks = folder.root / reward_check.CHECKS
    checks.mkdir()
    done = {"finished": True, "ranked": [], "bot": "racer", "saved_steps": 1000, "agent_steps": 0}
    (checks / "20260101-000000.json").write_text(json.dumps(done), encoding="utf-8")
    (checks / "20260102-000000.json").write_text(json.dumps({"finished": False, "done": 1, "rounds": 4}))
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        job = Job("racer", "check", [], folder.root / CONSOLE, sleeper, started=time.time() - 30)
        api._jobs._jobs[("racer", "check")] = job
        playing = api.reward_check("racer")
        assert playing["playing"] and playing["progress"] == 0.25
        assert 85 <= playing["seconds_left"] <= 95, "three rounds more, at 30 s for the first"

        sync_api = playwright_or_skip()
        errors: list[str] = []
        with sync_api.sync_playwright() as p:
            browser = open_chromium(p)
            try:
                tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
                open_bot(tab, "racer")
                tab.click("[data-tab=rewards]")
                tab.wait_for_selector("#rw-try [data-act=rw-try-stop]")
                assert "Playing… 25%" in tab.text_content("#rw-try [data-act=rw-try]")
                assert "left" in tab.text_content("#rw-try-time")
                assert tab.locator("#rw-try-steps").is_disabled()
                tab.click("#rw-try [data-act=rw-try-stop]")
                sleeper.wait(10)
                tab.wait_for_selector("#rw-try [data-act=rw-try]:has-text('Try these rewards')")
                assert tab.locator("#rw-try [data-act=rw-try-stop]").count() == 0
                assert tab.locator("#rw-try .banner.bad").count() == 0, "stopped: not a failure"
                assert tab.input_value("#rw-try-steps") == "250000", "shorter without a graphics card"
                assert tab.text_content("#rw-try-time") == "Roughly 3 minutes at this bot's speed"
                tab.select_option("#rw-try-steps", "1000000")
                assert tab.text_content("#rw-try-time") == "Roughly 10 minutes at this bot's speed"
            finally:
                browser.close()
        assert errors == []
    finally:
        sleeper.kill()
        sleeper.wait()
    stopped = api.reward_check("racer")
    assert stopped["stopped"] and not stopped["error"] and not stopped["playing"]
    assert stopped["result"]["file"] == "20260101-000000.json", (
        "the check stopped part-way leaves the one before"
    )


def test_without_a_graphics_card_the_wizard_says_so_and_the_review_is_honest(no_card: None) -> None:
    assert garage.review_words("run.precision", "fp16", False).startswith("half (FP16) on a graphics card")
    assert "processor" in garage.review_words("run.precision", "fp16", False)
    assert garage.review_words("run.precision", "fp16", True) == "half (FP16)"
    heads = garage.review_words("model.value_heads", ["main", "goal", "vdag1", "vdag2"])
    assert heads == "4 (the main one, a goal critic, two twin critics)"


def test_the_wizard_marks_what_is_made_for_a_graphics_card(tmp_path: Path, no_card: None) -> None:
    from test_dashboard_home import ROOT, quiet

    api = quiet(Api(ROOT, tmp_path / "runs"))
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            tab.evaluate("newBotWizard()")
            tab.wait_for_selector("#wz-no-card")
            assert "no graphics card" in tab.text_content("#wz-no-card")
            gco = tab.locator("[data-kind='preset:gco_1v1_starter']")
            assert "Slow on this PC" in gco.text_content()
            assert "Slow without" not in tab.text_content("[data-kind='preset:1v1_starter']")
            tab.click("#wz-next")
            tab.click("#wz-next")
            tab.wait_for_selector("#wz-brain-large")
            assert "Slow on this PC" in tab.text_content("#wz-brain-large")
            assert "Slow without" not in tab.text_content("#wz-brain-small")
            assert tab.locator("#wz-no-card").count() == 1
        finally:
            browser.close()
    assert errors == []


def test_packing_leaves_the_game_files_out_and_points_to_the_garage(tmp_path: Path) -> None:
    from nitro import pack
    from test_dashboard_bot import rlbot_bot

    runs, project = tmp_path / "runs", tmp_path / "project"
    make_bot(runs, "racer", saves=1)
    rlbot_bot(project, "racer")
    api = Api(project, runs)
    assert "--no-meshes" in api.tool_command("racer", "pack", {})["command"]
    assert "--no-meshes" not in api.tool_command("racer", "pack", {"meshes": True})["command"]
    archive, _ = pack.pack(runs / "racer", tmp_path / "racer.nitropack.zip")
    asked = api.unpack_bot

    def unpack_bot(path: str = "", replace: bool = False, new_name: str = "") -> dict[str, Any]:
        return asked(path or str(archive), replace, new_name)  # the file the dialog would give

    api.unpack_bot = unpack_bot  # type: ignore[method-assign]
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "racer")
            tab.click("[data-tab=tools]")
            tab.click("#tools-page [data-tool=pack]")
            tab.wait_for_selector("#opt-meshes")
            assert not tab.is_checked("#opt-meshes"), "the arena files are game files: left out"
            about = tab.text_content("#tool-body")
            assert "Add a packed bot" in about and "nitro unpack" not in about
            assert "never share a pack that has them" in about
            tab.click("#tool-foot [data-close]")
            tab.click("#tools-page [data-tool=export]")
            tab.wait_for_selector("#tool-body [data-act=tool-copy-path]")
            assert "builds the bot program" not in tab.text_content("#tool-body")
            tab.click("#tool-body [data-act=tool-copy-path]")
            tab.wait_for_selector("#toasts .toast")
            tab.click("#tool-foot [data-close]")

            tab.click(".nav [data-go=home]")
            tab.click("#g-unpack")
            tab.wait_for_selector(".modal h2:has-text('There is already a bot called racer')")
            assert tab.input_value("#up-name") == "racer-2"
            tab.fill("#up-name", "racer!")
            tab.click(".modal [data-act=unpack-rename]")
            tab.wait_for_selector("#up-error .err")
            tab.fill("#up-name", "racer-2")
            tab.click(".modal [data-act=unpack-rename]")
            tab.wait_for_selector("#bothead h1:has-text('racer-2')")
        finally:
            browser.close()
    assert errors == []
    assert sorted(b["name"] for b in api.home()["bots"]) == ["racer", "racer-2"]


def test_the_bots_head_fits_short_windows_and_its_facts_never_leave_one_alone(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "racer", saves=2)
    minutes_of_metrics(folder, 30)
    running(folder, time.time() - 200)
    api = Api(tmp_path / "project", runs)
    assert api.stage("racer", "ppo.policy_lr", "0.0002")["pending"]
    rows = """() => {
        const cells = [...document.querySelectorAll('#bothead .figures > div')];
        const tops = cells.map(el => Math.round(el.getBoundingClientRect().top));
        const count = {}; tops.forEach(t => { count[t] = (count[t] || 0) + 1; });
        return Object.values(count);
    }"""
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            for width, height in ((900, 640), (1280, 720), (1366, 768), (1920, 1080)):
                tab = opened(browser, api, errors, viewport={"width": width, "height": height})
                open_bot(tab, "racer")
                tab.wait_for_selector("#bothead .figures")
                found = tab.evaluate(rows)
                assert len(found) == 1 or min(found) >= 2, (width, found)
                tab.click(".tabs [data-tab=settings]")
                tab.wait_for_selector("#tab .setting")
                if height < 900:
                    tabs_top = tab.locator(".tabs").bounding_box()["y"]
                    assert tabs_top < 100, "a tab opened in a short window shows its content"
                tab.close()
        finally:
            browser.close()
    assert errors == []


def test_settings_read_plainly_first_with_the_full_text_behind_nerd_explanation(tmp_path: Path) -> None:
    from nitro.config.schema import NitroConfig
    from nitro.dashboard import describe, guide, layout
    from test_dashboard import items

    found = items(describe.describe(NitroConfig(), "", locked=False))
    texts = " ".join(item["help"] for item in found.values())
    for internal in ("plan's rule", "nproven"):
        assert internal not in texts, internal
    assert "nproven" not in " ".join(guide.NERD.values()) + " ".join(guide.MORE.values())
    assert "policy_lr = 4.5e-4" in guide.NERD["ppo.policy_lr"], "the example shows the default"
    from nitro.dashboard.api import WIZARD

    for _, _, paths in WIZARD:
        for path in paths:
            assert found[path]["plain"] == layout.PLAIN[path], path
    assert found["env.obs_team_size"]["title"] == "Observation room (players per team)"
    from nitro.dashboard import starts

    kickoff = next(k for k in starts.catalog() if k["name"] == "kickoff")
    assert [p["label"] for p in kickoff["params"]] == ["Random shift (units)"]
    shot = next(k for k in starts.catalog() if k["name"] == "shot")
    assert "Min distance (units)" in [p["label"] for p in shot["params"]]

    runs = tmp_path / "runs"
    make_bot(runs, "racer", saves=1)
    from test_dashboard_home import ROOT, quiet

    api = quiet(Api(ROOT, runs))
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "racer")
            tab.click(".tabs [data-tab=settings]")
            row = tab.locator("[data-item='ppo.policy_lr']")
            row.wait_for()
            assert row.locator(".help").text_content() == layout.PLAIN["ppo.policy_lr"]
            row.locator("[data-nerd]").click()
            assert "Measured: 4.5e-4" in row.locator("[data-full-help]").text_content()

            tab.evaluate("newBotWizard()")
            tab.wait_for_selector("#wz-next")
            tab.click("#wz-next")
            tab.click("#wz-next")
            tab.click("#wz-brain-custom")
            tick = tab.locator("#wz-env-tick_skip")
            tick.wait_for()
            labels = tab.locator(".wiz-item label").all_text_contents()
            assert any(label.startswith("Observation room") for label in labels), labels
            minibatches = tab.locator(".wiz-item:has(#wz-ppo-minibatches)")
            assert "plan's rule" not in minibatches.text_content()
            minibatches.locator("[data-wizhelp]").click()
            assert "too little to count" in tab.text_content(
                ".wiz-item:has(#wz-ppo-minibatches) [data-nerd-help]"
            )
            tick.fill("abc")
            tick.dispatch_event("change")
            assert tab.text_content("#wz-rate") == "(a whole number, such as 4)"
            tab.wait_for_selector("#wz-count .err")
            assert "cannot be counted yet" in tab.text_content("#wz-count")
            tick.fill("8")
            tick.dispatch_event("change")
            assert tab.text_content("#wz-rate") == "(15.0 decisions a second)"
        finally:
            browser.close()
    assert errors == []


def test_check_this_pc_shows_code_as_code_and_addresses_as_links(tmp_path: Path) -> None:
    from test_dashboard_home import ROOT, quiet

    api = quiet(Api(ROOT, tmp_path / "runs"))
    fix = "Install its driver from https://www.nvidia.com/drivers, restart, then run `nitro doctor` again."
    api.check_pc = lambda: {  # type: ignore[method-assign]
        "checks": [
            {"name": "GPU", "state": "warning", "message": "PyTorch cannot see an NVIDIA GPU", "fix": fix},
            {"name": "Disk space", "state": "ok", "message": "Run `nitro fetch-meshes`", "fix": ""},
        ],
        "problems": 0,
        "warnings": 1,
        "verdict": "Ready to train.",
        "version": "1.0.0",
        "command": "nitro doctor",
    }
    opened_urls: list[str] = []
    api.open_url = lambda url: opened_urls.append(url) or {"opened": True}  # type: ignore[method-assign, func-returns-value]
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            tab.click(".nav [data-go=doctor]")
            row = tab.locator(".check-row[data-check=GPU]")
            row.wait_for()
            assert row.locator(".name").text_content() == "Graphics card"
            assert (
                "NVIDIA graphics card" in row.locator(".msg").text_content() and "`" not in row.text_content()
            )
            assert "press Check again" in row.locator(".fix").text_content()
            assert (
                tab.locator(".check-row[data-check='Disk space'] code").text_content() == "nitro fetch-meshes"
            )
            row.locator(".weblink").click()
            tab.wait_for_function("1")
            assert opened_urls == ["https://www.nvidia.com/drivers"]
        finally:
            browser.close()
    assert errors == []


def test_the_console_tab_words_terminal_lines_for_the_window() -> None:
    sync_api = playwright_or_skip()
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            from nitro.dashboard import app

            tab = browser.new_page()
            tab.set_content(app.page_source())
            lines = [
                "Training 'x'. Progress: x/report.html (open it in a browser). Ctrl+C stops.",
                "Stopping after this iteration and saving. Press Ctrl+C again to quit at once without "
                "saving.",
                "WARNING: Setting changed since the save: run.checkpoint_minutes: 15.0 -> 2.0",
                "Watching for new saves: the page shows the newest one from its next game on. Ctrl+C stops.",
            ]
            shown = tab.evaluate("lines => lines.map(consoleLine)", lines)
            assert shown == [
                "Training 'x'. Progress: x/report.html (open it in a browser). Stop and save stops it.",
                "Stopping after this round of training and saving.",
                "Note: changed since the last save (by Apply or in the settings file): "
                "run.checkpoint_minutes: "
                "15.0 -> 2.0",
                "Watching for new saves: the page shows the newest one from its next game on.",
            ]
        finally:
            browser.close()


def test_a_bot_can_be_renamed_and_deleted_without_touching_another(tmp_path: Path) -> None:
    from test_dashboard_bot import rlbot_bot

    runs, project = tmp_path / "runs", tmp_path / "project"
    racer = make_bot(runs, "racer", saves=1)
    other = make_bot(runs, "other", saves=1)
    rlbot_bot(project, "racer")
    api = Api(project, runs)
    assert api.stage("racer", "ppo.policy_lr", "0.0002")["pending"]
    running(racer, time.time())
    assert api.rename_bot("racer", "speedy")["error"] == "racer is training: stop it first"
    assert api.delete_bot("racer")["error"] == "racer is training: stop it first"
    (racer.root / bots.RUNNING).unlink()
    follower = make_bot(runs, "follower", text=_from("racer"))
    assert "follower will start from racer's networks" in api.delete_bot("racer")["error"]
    import shutil

    shutil.rmtree(follower.root)
    assert "at most 64 characters" in api.rename_bot("racer", "r" * 65)["error"]
    assert api.rename_bot("racer", "other")["error"] == "there is already a bot called other"
    info = json.loads(racer.info.read_text(encoding="utf-8"))
    info["settings_dir"] = str(racer.root / "settings-files")
    racer.info.write_text(json.dumps(info), encoding="utf-8")
    assert api.rename_bot("racer", "speedy") == {"renamed": "speedy"}
    moved = RunFolder(runs / "speedy")
    assert not racer.root.exists() and moved.complete_checkpoints()
    assert json.loads(moved.info.read_text(encoding="utf-8"))["settings_dir"] == str(
        moved.root / "settings-files"
    )
    assert api.bot("speedy")["pending"] == {"ppo.policy_lr": 0.0002}, "its waiting changes go with it"

    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "speedy")
            tab.click(".tabs [data-tab=tools]")
            tab.click("#tools-page [data-tool=rename]")
            tab.fill("#rn-name", "bad name")
            tab.click(".modal [data-act=rename-go]")
            tab.wait_for_selector("#rn-error .err")
            tab.fill("#rn-name", "zoomer")
            tab.click(".modal [data-act=rename-go]")
            tab.wait_for_selector("#bothead h1:has-text('zoomer')")
            tab.click("#tools-page [data-tool=delete]")
            sure = tab.text_content(".modal")
            assert (
                "Delete zoomer?" in sure and "No other bot is touched" in sure and "cannot be undone" in sure
            )
            assert tab.locator("#del-go").is_disabled(), "the name is typed first"
            tab.fill("#del-name", "zoome")
            assert tab.locator("#del-go").is_disabled()
            tab.fill("#del-name", "zoomer")
            tab.click("#del-go")
            tab.wait_for_selector("#main [data-bot=other]")
        finally:
            browser.close()
    assert errors == []
    assert not (runs / "zoomer").exists() and other.complete_checkpoints(), "the other bot is untouched"
    assert (project / "bots" / "racer").is_dir(), "its RLBot bot stays unless asked"
    assert "zoomer" not in json.dumps(api._pending)


def _from(source: str) -> str:
    from test_dashboard import TINY

    return TINY + f'\n[start_from]\nrun = "{source}"\n'


def test_notes_save_themselves_and_disabled_buttons_say_why(tmp_path: Path) -> None:
    from nitro.dashboard import resources

    assert resources.preset_title("gco_1v1_starter") == "GCO bot preset"
    assert resources.preset_title("1v1_starter") == "1v1 starter preset"
    runs = tmp_path / "runs"
    make_bot(runs, "fresh")
    api = Api(tmp_path / "project", runs)
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "fresh")
            tab.wait_for_selector("#bothead [data-why-off]")
            assert "come with its first save" in tab.text_content("#bothead [data-why-off]")
            assert tab.locator("#bothead [data-act=watch]").is_disabled()
            tab.click(".tabs [data-tab=notes]")
            tab.fill("#botnote", "try a bigger brain next")
            tab.wait_for_selector("[data-note-state=bot]:text-is('Saved')", timeout=5000)
            assert api.bot("fresh")["notes"] == {"bot": "try a bigger brain next"}, "saved without a click"
            tab.click(".tabs [data-tab=rewards]")
            tab.click("[data-act=rw-add]")
            tab.wait_for_selector(".catalog-pick .pick-word")
            words = set(tab.locator(".catalog-pick .pick-word").all_text_contents())
            assert words == {"+ Add to this bot", "Show it in the list"}
        finally:
            browser.close()
    assert errors == []


def test_a_gco_bot_is_named_by_its_design(tmp_path: Path) -> None:
    from test_dashboard import TINY

    runs = tmp_path / "runs"
    make_bot(runs, "gco", text=TINY.replace("value = [16]", 'value = [16]\nvalue_heads = ["main", "goal"]'))
    make_bot(runs, "plain")
    api = Api(tmp_path / "project", runs)
    assert api.bot("gco")["gco"] is True and api.bot("plain")["gco"] is False
    sync_api = playwright_or_skip()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            assert "GCO network" in tab.text_content(".botcard[data-bot=gco]")
            assert "Plain network" in tab.text_content(".botcard[data-bot=plain]")
        finally:
            browser.close()
    assert errors == []
