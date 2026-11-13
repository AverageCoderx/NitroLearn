"""The dashboard's first-run welcome (the Garage page before the first bot) and the App settings page's
Update NitroLearn and Editing NitroLearn sections, with the update's calls answered here."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import update
from nitro.dashboard import bots, updating
from nitro.dashboard.api import Api
from nitro.dashboard.jobs import CONSOLE, Job
from nitro.dashboard.updating import RESULT, UPDATE
from test_dashboard import ROOT, make_bot
from test_dashboard_home import opened, quiet, refresh, report

RELEASE = {
    "version": "1.1.0",
    "tag": "v1.1.0",
    "date": "1 December 2026",
    "notes": "Faster replays\n- Twice as fast",
    "page": "https://github.com/AverageCoderx/NitroLearn/releases/tag/v1.1.0",
}


def answer(state: str = "newer", **more: Any) -> dict[str, Any]:
    """What update_check answers, as nitro.update.Check.view and the mixin make it."""
    found = update.Check("1.0.0", update.Release(**RELEASE), state, git=False, blocked="")
    return {**found.view(), "finish_on_close": False, "busy": "", **more}


# ---------- the welcome ----------
def test_the_welcome_shows_its_three_steps_until_the_first_bot(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    looks: list[str] = []
    found = ["FAIL"]

    def check() -> dict[str, Any]:
        looks.append(found[0])
        return report(found[0])

    def choose() -> dict[str, Any]:  # in place of the window's file dialog and of starting the program
        program = tmp_path / "RLBot5" / "bin" / "launcher.exe"
        program.parent.mkdir(parents=True, exist_ok=True)
        program.touch()
        api.set_pref("rlbot_program", str(program))
        return {"opened": str(program)}

    api.check_pc = check  # type: ignore[method-assign]
    api.choose_rlbot = choose  # type: ignore[method-assign]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            welcome = tab.locator(".welcome")
            assert welcome.locator(".welcome-lockup svg").count() == 1, "the logo lockup"
            assert "trains Rocket League bots on this PC" in tab.text_content(".welcome-what")
            assert "Never use a bot online against real players" in tab.text_content(".welcome-fair")
            assert welcome.locator(".wtitle").all_text_contents() == [
                "Check this PC",
                "Make your first bot",
                "Tell NitroLearn where RLBot isLater",
            ], "the RLBot step is marked for later"
            assert [b.text_content() for b in welcome.locator(".wstep > .btn").all()] == [
                *("Check this PC", "Make your first bot", "Choose RLBot…"),
            ], "one button a step"
            assert tab.locator(".setup").count() == 0 and tab.locator(".firstrun").count() == 0, (
                "one onboarding"
            )
            tab.wait_for_timeout(1500)
            assert looks == [], "the checks run when asked, not by themselves"

            tab.click("#wel-pc")
            tab.wait_for_selector(".wstep[data-step=pc][data-state=problem]")
            assert looks == ["FAIL"] and "1 thing needs fixing" in tab.text_content("#wel-pc-line")
            problem = tab.locator(".wstep[data-step=pc] .wproblems li")
            assert problem.count() == 1 and "Collision meshes" in problem.text_content()
            problem.locator("[data-go=doctor]").click()
            tab.wait_for_selector(".check-row[data-state=problem] >> text=Collision meshes")
            assert looks == ["FAIL"], "the Check this PC page shows the answer the welcome had"

            found[0] = "OK"
            tab.click(".nav [data-go=home]")
            tab.click("#wel-pc")
            tab.wait_for_selector(".wstep[data-step=pc][data-state=done]")
            assert tab.text_content("#wel-pc-line").startswith("This PC is ready. 1 warning")

            tab.click("#wel-rlbot")
            tab.wait_for_selector(".wstep[data-step=rlbot][data-state=done]")
            assert tab.locator(".wstep[data-step=bot][data-state=todo]").count() == 1

            tab.click("#g-first")
            tab.wait_for_selector("#wizard [data-kind]")
            tab.keyboard.press("Escape")

            make_bot(runs, "first")
            tab.click(".nav [data-go=home]")
            refresh(tab)
            tab.wait_for_selector(".botcard[data-bot=first], .ontrack")
            assert tab.locator(".welcome").count() == 0, "gone once a bot exists"
            assert "All set" in tab.text_content(".setup"), "the short checklist carries on"
        finally:
            browser.close()
    assert errors == []


def test_the_welcome_can_be_hidden_for_good(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#welcome-dismiss")
            tab.wait_for_selector(".welcome", state="detached")
            assert "Your garage is empty" in tab.text_content(".firstrun")
            assert api.home()["prefs"]["setup_done"] is True, "kept with the other preferences"
            again = opened(browser, api, errors)
            assert again.locator(".welcome").count() == 0 and again.locator(".firstrun").count() == 1
        finally:
            browser.close()
    assert errors == []


def test_the_welcome_fits_the_smallest_window(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"), meshes="FAIL")
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            for theme in ("dark", "light"):
                api.set_pref("theme", theme)
                tab = opened(browser, api, errors, viewport={"width": 900, "height": 640})
                tab.click("#wel-pc")
                tab.wait_for_selector(".wstep[data-step=pc][data-state=problem]")
                assert tab.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), theme
                for step in tab.locator(".wstep").all():
                    button, body = (
                        step.locator("> .btn").bounding_box(),
                        step.locator(".wbody").bounding_box(),
                    )
                    assert button is not None and body is not None
                    assert (
                        button["x"] >= body["x"] + body["width"] - 1
                        or button["y"] >= body["y"] + body["height"] - 1
                    )
        finally:
            browser.close()
    assert errors == []


# ---------- Update NitroLearn ----------
class Updates:
    """Stands in for the update's calls: what the check answers, and how the update goes."""

    def __init__(self, api: Api) -> None:
        self.api = api
        self.check: dict[str, Any] = answer()
        self.states: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.restart: dict[str, Any] = {
            "manual": "Close NitroLearn and open it again to use the new version."
        }
        api.update_check = self.update_check  # type: ignore[method-assign]
        api.update_start = self.update_start  # type: ignore[method-assign]
        api.update_state = self.update_state  # type: ignore[method-assign]
        api.update_restart = self.update_restart  # type: ignore[method-assign]

    def update_check(self) -> dict[str, Any]:
        self.calls.append("check")
        return self.check

    def update_start(self) -> dict[str, Any]:
        self.calls.append("start")
        return {"started": True}

    def update_state(self, after: int = 0) -> dict[str, Any]:
        if not self.states:
            return {"state": "none", "text": "", "next": 0}
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def update_restart(self) -> dict[str, Any]:
        self.calls.append("restart")
        return self.restart


def done_state(
    installer: str = "done", replaced: list[str] | None = None, removed: list[str] | None = None
) -> dict[str, Any]:
    result = {"old": "1.0.0", "new": "1.1.0", "how": "files", "installer": installer}
    result |= {
        "replaced": replaced or [],
        "removed": removed or [],
        "backup": "/n/.tmp/update/replaced/1.0.0",
    }
    return {"state": "done", "text": "Copied 3 changed and 1 new files\n", "next": 40, "result": result}


def test_the_update_section_checks_only_when_asked_and_shows_each_answer(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    fake = Updates(api)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click(".nav [data-go=prefs]")
            tab.wait_for_selector("#upd-check")
            tab.wait_for_timeout(300)
            assert fake.calls == [], "never in the background"
            assert tab.text_content("#upd-box").startswith("You have NitroLearn")

            fake.check = {
                "error": "Could not check for updates: no release was found (NitroLearn's page ...)."
            }
            tab.click("#upd-check")
            tab.wait_for_selector("#upd-said.bad")
            assert tab.text_content("#upd-said").startswith("Could not check for updates: no release")

            fake.check = answer("same", line="You have 1.1.0, the newest version.")
            tab.click("#upd-check")
            tab.wait_for_selector("#upd-said >> text=the newest version")
            assert tab.locator("#upd-now").count() == 0

            fake.check = answer(busy="Stop training first: racer is training.")
            tab.click("#upd-check")
            tab.wait_for_selector("#upd-now")
            assert (
                tab.text_content("#upd-said") == "You have 1.0.0. NitroLearn 1.1.0 is out (1 December 2026)."
            )
            assert tab.text_content(".upd-notes") == "Faster replays\n- Twice as fast"
            assert tab.is_disabled("#upd-now") and "racer is training" in tab.text_content("#upd-blocked")

            fake.check = answer()
            tab.click("#upd-check")
            tab.wait_for_selector("#upd-now:not([disabled])")
            tab.click("#upd-now")
            sure = tab.text_content(".modal")
            assert "Update to NitroLearn 1.1.0?" in sure and "bots, their settings, replays and saves" in sure
            assert "Training must be stopped first" in sure and "copied to .tmp/update/replaced" in sure
            assert "nothing of yours is" in sure and "Nothing is deleted" not in sure
            tab.click("[data-close]")
            assert fake.calls.count("start") == 0, "Cancel changes nothing"

            fake.states = [
                {"state": "running", "text": "Downloading NitroLearn 1.1.0\n", "next": 30},
                done_state(replaced=["configs/presets/starter.toml"], removed=["docs/old_page.md"]),
            ]
            tab.click("#upd-now")
            tab.click("#upd-go")
            tab.wait_for_selector("#upd-said >> text=Updating to 1.1.0")
            tab.wait_for_selector("#upd-restart", timeout=10000)
            assert (
                tab.text_content("#upd-said")
                == "UpdatedUpdated from 1.0.0 to 1.1.0. Restart the dashboard to use it."
            )
            assert "configs/presets/starter.toml" in tab.text_content(".upd-replaced")
            removed = tab.text_content("#upd-removed")
            assert (
                "1 file the new version no longer has was removed" in removed
                and "docs/old_page.md" in removed
            )
            assert "Downloading NitroLearn 1.1.0" in tab.text_content("#updText")
            assert tab.text_content("#upd-restart") == "Restart"
            tab.click("#upd-restart")
            tab.wait_for_selector("#upd-manual >> text=Close NitroLearn and open it again")
        finally:
            browser.close()
    assert errors == []


def test_on_windows_the_update_closes_the_dashboard_to_finish(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    fake = Updates(api)
    fake.check = answer(finish_on_close=True)
    fake.states = [done_state(installer="later")]
    fake.restart = {"closing": True}
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click(".nav [data-go=prefs]")
            tab.click("#upd-check")
            tab.click("#upd-now")
            assert "To finish, NitroLearn closes and opens again." in tab.text_content(".modal")
            tab.click("#upd-go")
            tab.wait_for_selector("#upd-restart")
            assert "NitroLearn closes to finish the update" in tab.text_content("#upd-said")
            assert tab.text_content("#upd-restart") == "Close and finish"
            tab.click("#upd-restart")
            tab.wait_for_selector("#upd-restart[disabled] >> text=Closing…")
            assert fake.calls[-1] == "restart"

            # Closed without finishing: the next start offers it again.
            fake.states = [{**done_state(installer="later"), "text": "", "unfinished": True}]
            again = opened(browser, api, errors)
            again.wait_for_selector(".toast >> text=The update to 1.1.0 is not finished")
            again.click(".nav [data-go=prefs]")
            assert again.text_content("#upd-restart") == "Close and finish"
        finally:
            browser.close()
    assert errors == []


def test_the_search_and_the_editing_section(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    fake = Updates(api)
    opened_things: list[tuple[str, str]] = []
    api.open_folder = lambda name, which="bot": opened_things.append((name, which)) or {"opened": "x"}  # type: ignore[method-assign]
    api.build_rust = lambda: {"started": True, "files": []}  # type: ignore[method-assign]
    built = iter([{"state": "running", "text": "Compiling nitro_core\n", "next": 21}])
    api.build_state = lambda after=0: next(built, {"state": "done", "text": "", "next": 21})  # type: ignore[method-assign]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#finder-q")
            tab.keyboard.type("update")
            tab.wait_for_selector("#finder-list .finder-row >> text=Update NitroLearn")
            assert "nitro update" in tab.text_content("#finder-list")
            tab.click("#finder-list .finder-row >> text=Update NitroLearn")
            tab.wait_for_selector("#upd-now")
            assert fake.calls == ["check"], "the search's Update NitroLearn checks at once"

            editing = tab.locator("#editing")
            assert str(ROOT) in editing.text_content()
            assert "Rust" in editing.text_content() and "your_own_rewards.md" in editing.text_content()
            tab.click("#edit-open")
            tab.click("#edit-guide")
            tab.wait_for_function("document.querySelectorAll('.toast.bad').length === 0")
            assert opened_things[-2:] == [("", "code"), ("", "rewards-guide")]
            tab.click("#edit-rebuild")
            tab.wait_for_selector("#edit-build >> text=Built.", timeout=10000)
            assert "Compiling nitro_core" in tab.text_content("#editBuildText")
        finally:
            browser.close()
    assert errors == []


# ---------- the calls behind the section ----------
def test_the_update_calls_refuse_while_something_runs_and_report_what_the_update_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    racer = make_bot(runs, "racer")
    (racer.root / bots.RUNNING).write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    assert api.update_start() == {"error": "Stop training first: racer is training."}

    def failing(root: Path) -> update.Check:
        raise update.UpdateError("Could not check for updates: GitHub cannot be reached (offline).")

    monkeypatch.setattr(update, "check", failing)
    assert api.update_check() == {"error": "Could not check for updates: GitHub cannot be reached (offline)."}
    monkeypatch.setattr(
        update, "check", lambda root: update.Check("1.0.0", update.Release(**RELEASE), "newer", False, "")
    )
    assert api.update_check()["busy"] == "Stop training first: racer is training."
    (racer.root / bots.RUNNING).unlink()

    converting = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        api._jobs._jobs[(".convert", "convert")] = Job(
            ".convert", "convert", [], tmp_path / CONSOLE, converting
        )
        assert "the replay conversion" in api.update_start()["error"]
    finally:
        converting.kill()
        converting.wait()

    started: list[list[str]] = []

    def start(bot: str, kind: str, args: list[str], folder: Path) -> Job:
        started.append(args)
        result = Path(args[args.index("--result") + 1])
        data = {"old": "1.0.0", "new": "1.1.0", "how": "files", "installer": "done", "replaced": []}
        code = "import pathlib; print('Updated'); "
        code += f"pathlib.Path({str(result)!r}).write_text({json.dumps(data)!r})"
        with (folder / CONSOLE).open("ab") as out:
            process = subprocess.Popen([sys.executable, "-c", code], stdout=out, stderr=subprocess.STDOUT)
        job = Job(bot, kind, args, folder / CONSOLE, process)
        api._jobs._jobs[(bot, kind)] = job
        return job

    monkeypatch.setattr(api._jobs, "start", start)
    assert api.update_state() == {"state": "none", "text": "", "next": 0}
    folder = runs / ".dashboard" / "update"
    folder.mkdir(parents=True)
    (folder / RESULT).write_text(json.dumps({"old": "1.0.0", "new": "1.1.0", "installer": "later"}))
    monkeypatch.setattr(updating, "core_stale", lambda project: True)
    assert api.update_state()["unfinished"] is True, "closed before the installer ran: still to finish"
    monkeypatch.setattr(updating, "core_stale", lambda project: False)
    assert api.update_state()["state"] == "none", "finished by the installer: the core fits the code"
    assert updating.core_stale(ROOT) is False, "this folder's core was built from its code"
    assert api.update_start() == {"started": True}
    assert started[0][:3] == ["update", "--yes", "--runs"] and started[0][4:6] == [
        "--installer",
        "later" if update.WINDOWS else "now",
    ]
    for _ in range(200):
        state = api.update_state()
        if state["state"] != "running":
            break
        time.sleep(0.05)
    assert state["state"] == "done" and state["result"]["new"] == "1.1.0" and "Updated" in state["text"]
    assert (runs / ".dashboard" / "update" / RESULT).is_file()
    assert api.update_restart() == {"manual": "Close NitroLearn and open it again to use the new version."}, (
        "no window here: it only says what to do"
    )
    assert api._jobs.get(UPDATE, "update") is not None
