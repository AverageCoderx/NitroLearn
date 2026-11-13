"""The dashboard's Install RLBot v5 button: in the welcome's RLBot step and in the Open RLBot dialog while
RLBot is not found, its progress and console text, and what it says when it cannot install. The two calls
behind it (install_rlbot, install_rlbot_state) are answered here."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import rlbot_install
from nitro.dashboard import machine
from nitro.dashboard.api import Api
from test_dashboard import ROOT, make_bot
from test_dashboard_home import opened, quiet

NOT_FOUND = "RLBot was not found after its installer. Get it from https://rlbot.org if it did not install."
NOTE = (
    "RLBot is a separate program with its own licence. It downloads from RLBot's official releases. "
    "Remove it later from Windows' Apps list."
)


@pytest.fixture(autouse=True)
def on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """RLBot's installer is offered on Windows only: these tests see the page as on Windows."""
    monkeypatch.setattr(machine, "facts", lambda: {"graphics_card": None, "windows": True})


class Installer:
    """Stands in for RLBot's install: `state` is what install_rlbot_state says, `printed` its console."""

    def __init__(self, api: Api, program: Path) -> None:
        self.api = api
        self.program = program
        self.state = "none"
        self.printed = ""
        self.start: dict[str, Any] = {"started": True, "from": rlbot_install.OFFICIAL + "installer/"}
        self.calls: list[str] = []
        api.install_rlbot = self.install_rlbot  # type: ignore[method-assign]
        api.install_rlbot_state = self.install_rlbot_state  # type: ignore[method-assign]

    def install_rlbot(self) -> dict[str, Any]:
        self.calls.append("install")
        if "started" in self.start:
            self.state = "running"
        return self.start

    def install_rlbot_state(self, after: int = 0) -> dict[str, Any]:
        found = self.state == "done"
        if found:
            self.program.write_bytes(b"")
        return {
            "state": self.state,
            "text": self.printed[int(after) :],
            "next": len(self.printed),
            "program": str(self.program) if found else "",
        }


def test_the_welcome_installs_rlbot_and_then_shows_it_found(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    fake = Installer(api, tmp_path / "RLBot5" / "bin" / "launcher.exe")
    fake.program.parent.mkdir(parents=True)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            step = tab.locator(".wstep[data-step=rlbot]")
            assert step.get_attribute("data-state") == "todo"
            assert step.locator(".rli-go").text_content() == "Install RLBot v5"
            assert step.locator(".rli-note").text_content() == NOTE
            assert step.locator("> .btn").text_content() == "Choose RLBot…", (
                "choosing it stays the step's button"
            )
            assert tab.evaluate("COMMANDS.find(c => c.id === 'install-rlbot').when()") is True

            fake.printed = "Downloading RLBot's official installer from https://github.com/RLBot/launcher\n"
            step.locator(".rli-go").click()
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-said >> text=Installing RLBot…")
            assert "Click through RLBot's installer window." in step.locator(".rli-said").text_content()
            assert (
                step.locator(".rli-go").is_disabled()
                and step.locator(".rli-go").text_content() == "Installing…"
            )
            assert step.locator(".rli-text").count() == 0, "the console text waits for Details"
            step.locator(".rli-toggle").click()
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-text")
            assert "Downloading RLBot's official installer" in step.locator(".rli-text").text_content()
            assert tab.locator(".modal").count() == 0

            fake.printed += "Opening RLBot's installer: follow its windows.\n"
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-text >> text=follow its windows")
            fake.printed += f"RLBot is installed: {fake.program}\n"
            fake.state = "done"
            tab.wait_for_selector(".wstep[data-step=rlbot][data-state=done]")
            tab.wait_for_selector(".toast >> text=RLBot is installed")
            assert tab.locator(".rli-go").count() == 0, "found: the button is gone"
            assert api.home()["prefs"]["rlbot_program"] == str(fake.program), "Open RLBot now opens it"
            assert tab.evaluate("COMMANDS.find(c => c.id === 'install-rlbot').when()") is False
            assert fake.calls == ["install"]
        finally:
            browser.close()
    assert errors == []


def test_rlbot_v5_installed_already_shows_as_found_without_choosing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RLBot v5 where its installer puts it (as Check this PC finds it): the welcome's RLBot step is done,
    with no Install or Choose button, and Open RLBot starts it, though nothing was chosen."""
    sync_api = playwright_or_skip()
    local = tmp_path / "Local"
    launcher = local / "RLBot5" / "bin" / "launcher.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"")
    monkeypatch.setattr(rlbot_install, "local_app_data", lambda: local)
    starts: list[Path] = []
    monkeypatch.setattr("nitro.dashboard.api._open", starts.append)
    api = quiet(Api(ROOT, tmp_path / "runs"))
    assert api.home()["rlbot_ready"] is True and "rlbot_program" not in api.home()["prefs"]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            step = tab.locator(".wstep[data-step=rlbot]")
            assert step.get_attribute("data-state") == "done"
            assert step.locator(".wtitle").text_content() == "RLBot v5 is foundLater"
            assert step.locator(".rli-go, [data-act=rlbot-choose]").count() == 0
            step.locator("> .btn").click()
            tab.wait_for_selector(".toast >> text=Opening RLBot")
            assert starts == [launcher]
        finally:
            browser.close()
    assert errors == []
    older = tmp_path / "RLBotGUIX" / "RLBot.exe"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"")
    api.set_pref("rlbot_program", str(older))
    assert api.open_rlbot() == {"opened": str(launcher)}, "an older RLBot chosen: RLBot v5 is used"
    assert api.home()["prefs"]["rlbot_program"] == "", "and the older one forgotten"
    launcher.unlink()
    assert api.home()["rlbot_ready"] is False and api.open_rlbot() == {"need_program": True}


def test_the_open_rlbot_dialog_installs_rlbot_next_to_choose(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown")
    api = quiet(Api(ROOT, runs))
    fake = Installer(api, tmp_path / "RLBot5" / "bin" / "launcher.exe")
    fake.program.parent.mkdir(parents=True)
    opened_programs: list[str] = []

    def start() -> dict[str, Any]:  # in place of starting RLBot's program
        chosen = api.home()["prefs"].get("rlbot_program", "")
        if not chosen:
            return {"need_program": True}
        opened_programs.append(chosen)
        return {"opened": chosen}

    api.open_rlbot = start  # type: ignore[method-assign]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#band [data-go=rlbot]")
            tab.wait_for_selector(".modal >> text=Open RLBot")
            assert tab.locator(".modal .foot .btn").all_text_contents() == [
                *("Not now", "Install RLBot v5", "Choose RLBot…"),
            ]
            assert NOTE in tab.text_content(".modal")
            fake.printed = "Opening RLBot's installer: follow its windows.\n"
            tab.click(".modal .rli-go")
            tab.wait_for_selector(".modal .rli-said >> text=Click through RLBot's installer window")
            fake.state = "done"
            tab.wait_for_selector(".modal .rli-said >> text=RLBot is installed")
            assert tab.locator(".modal .rli-go").count() == 0
            tab.click(".modal [data-act=rlbot-open-close]")
            tab.wait_for_selector(".toast >> text=Opening RLBot")
            assert opened_programs == [str(fake.program)] and not tab.is_visible(".modal")
        finally:
            browser.close()
    assert errors == []


def test_rlbot_install_problems_are_said_plainly(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    real = api.install_rlbot
    fake = Installer(api, tmp_path / "RLBot5" / "bin" / "launcher.exe")
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            box = tab.locator(".wstep[data-step=rlbot] .rli")

            if sys.platform != "win32":  # off Windows the real call says so itself (on Windows it installs)
                assert real() == {"error": rlbot_install.NOT_WINDOWS}
            fake.start = {"error": rlbot_install.NOT_WINDOWS}
            box.locator(".rli-go").click()
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-said.warn")
            assert box.locator(".rli-said").text_content() == rlbot_install.NOT_WINDOWS
            assert not box.locator(".rli-go").is_disabled(), "it can be tried again"

            fake.start = {"error": "RLBot's installer is already open"}
            box.locator(".rli-go").click()
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-said >> text=already open")

            fake.start = {"started": True, "from": rlbot_install.OFFICIAL + "installer/"}
            fake.printed = (
                "Opening RLBot's installer: follow its windows.\nRLBot's installer was cancelled.\n"
                f"{NOT_FOUND}\n"
            )
            fake.state = "running"
            box.locator(".rli-go").click()
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-said >> text=Click through")
            fake.state = "failed"
            tab.wait_for_selector(".wstep[data-step=rlbot] .rli-said.bad")
            assert box.locator(".rli-said").text_content() == NOT_FOUND, "its last line, as it said it"
            assert tab.locator(".wstep[data-step=rlbot][data-state=todo]").count() == 1
            assert box.locator(".rli-go").text_content() == "Install RLBot v5"
            box.locator(".rli-toggle").click()
            assert "was cancelled" in box.locator(".rli-text").text_content()
        finally:
            browser.close()
    assert errors == []


def test_no_install_button_once_rlbot_is_chosen(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    program = tmp_path / "RLBot5" / "bin" / "launcher.exe"  # an older RLBot would not count
    program.parent.mkdir(parents=True)
    program.write_bytes(b"")
    api.set_pref("rlbot_program", str(program))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            assert tab.locator(".wstep[data-step=rlbot][data-state=done]").count() == 1
            assert tab.locator(".rli-go").count() == 0 and tab.locator(".rli-note").count() == 0
            assert tab.evaluate("COMMANDS.find(c => c.id === 'install-rlbot').when()") is False
        finally:
            browser.close()
    assert errors == []


def test_elsewhere_than_windows_only_choosing_rlbot_is_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(machine, "facts", lambda: {"graphics_card": None, "windows": False})
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            step = tab.locator(".wstep[data-step=rlbot]")
            step.wait_for()
            assert step.locator(".rli-go").count() == 0 and "Apps list" not in step.text_content()
            assert "installer is for Windows" in step.text_content()
            assert tab.evaluate("COMMANDS.find(c => c.id === 'install-rlbot').when()") is False
            tab.evaluate("rlbotModal()")
            assert tab.locator(".modal .rli-go").count() == 0
            assert tab.locator(".modal [data-act=rlbot-choose-close]").count() == 1
            assert "Install RLBot v5 from here" not in tab.text_content(".modal")
        finally:
            browser.close()
    assert errors == []
