"""The dashboard's notice of a new version: asked when the page opens, GitHub at most once a day (its answer
kept in runs/.dashboard), silent when GitHub cannot be reached, Not now remembered for one version, App
settings' switch turning it off; and the Garage's line and App settings' dot that show it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import update
from nitro.dashboard import updating
from nitro.dashboard.api import Api
from test_dashboard import ROOT, make_bot
from test_dashboard_home import mark_running, opened, quiet
from test_dashboard_welcome import Updates, answer

INSTALLED = "1.0.1"


def release(version: str, **more: str) -> update.Release:
    fields = {
        "version": version,
        "tag": f"v{version}",
        "date": "20 November 2026",
        "notes": "Faster start\n- The Garage opens at once",
        "page": f"https://github.com/AverageCoderx/NitroLearn/releases/tag/v{version}",
    }
    return update.Release(**{**fields, **more})


class GitHub:
    """Stands in for GitHub: the newest release it gives (or the failure it raises), and how often asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, version: str = "1.0.2") -> None:
        self.asked = 0
        self.answer: update.Release | Exception = release(version)
        monkeypatch.setattr(updating, "ask_github", self.ask)
        monkeypatch.setattr(update, "installed_version", lambda root: INSTALLED)

    def ask(self) -> update.Release:
        self.asked += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def kept(runs: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((runs / ".dashboard" / updating.NOTICE).read_text(encoding="utf-8"))
    return data


def age(runs: Path, seconds: float) -> None:
    """Makes the kept answer `seconds` older."""
    data = kept(runs)
    data["checked"] -= seconds
    (runs / ".dashboard" / updating.NOTICE).write_text(json.dumps(data), encoding="utf-8")


# ---------- the calls ----------
def test_github_is_asked_once_an_interval_and_the_answer_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    github = GitHub(monkeypatch)
    api = Api(ROOT, runs)
    first = api.update_notice()
    assert github.asked == 1 and first["state"] == "newer" and first["not_now"] is False
    assert first["installed"] == INSTALLED and first["release"]["version"] == "1.0.2"
    assert first["git"] == update.is_git(ROOT)
    assert kept(runs)["release"]["notes"] == "Faster start\n- The Garage opens at once"

    for _ in range(3):  # opened again, and the hourly asks of a window left open
        assert api.update_notice() == first
    assert Api(ROOT, runs).update_notice() == first, "a dashboard opened again reads the kept answer"
    assert github.asked == 1

    assert 12 * 3600 <= updating.EVERY <= 24 * 3600
    age(runs, updating.EVERY - 60)
    api.update_notice()
    assert github.asked == 1, "not before the interval"
    age(runs, 120)
    github.answer = release("1.0.3")
    assert api.update_notice()["release"]["version"] == "1.0.3" and github.asked == 2

    age(runs, -3 * updating.EVERY)  # the clock was set back: the kept time is in the future
    api.update_notice()
    assert github.asked == 3


def test_a_failed_check_is_silent_and_keeps_what_was_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    github = GitHub(monkeypatch)
    github.answer = update.UpdateError("Could not check for updates: GitHub cannot be reached (offline).")
    api = Api(ROOT, runs)
    assert api.update_notice() == {"state": "unknown", "installed": INSTALLED}
    assert github.asked == 1 and "release" not in kept(runs)
    assert api.update_notice() == {"state": "unknown", "installed": INSTALLED}
    assert github.asked == 1, "a failed check waits for the interval too: never a request at every open"

    github.answer = release("1.0.2")
    age(runs, updating.EVERY)
    assert api.update_notice()["state"] == "newer"
    for failure in (OSError("reset"), ValueError("bad answer"), update.UpdateError("GitHub answered 503")):
        github.answer = failure
        age(runs, updating.EVERY)
        out = api.update_notice()
        assert out["state"] == "newer" and out["release"]["version"] == "1.0.2", "the kept answer stands"
        assert "error" not in out

    (runs / ".dashboard" / updating.NOTICE).write_text("{not json", encoding="utf-8")
    github.answer = update.UpdateError("offline")
    assert api.update_notice() == {"state": "unknown", "installed": INSTALLED}, "a broken file: asked again"


@pytest.mark.parametrize(("version", "state"), [("1.0.2", "newer"), ("1.0.1", "same"), ("1.0.0", "older")])
def test_the_notice_compares_the_newest_release_with_this_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, state: str
) -> None:
    GitHub(monkeypatch, version)
    assert Api(ROOT, tmp_path / "runs").update_notice()["state"] == state


def test_not_now_hides_one_version_until_the_next(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    github = GitHub(monkeypatch)
    api = Api(ROOT, runs)
    assert api.update_notice()["not_now"] is False
    assert api.update_not_now("1.0.2") == {"not_now": "1.0.2"}
    assert api.update_notice()["not_now"] is True
    assert Api(ROOT, runs).update_notice()["not_now"] is True, "remembered when the dashboard opens again"
    github.answer = release("1.0.3")
    age(runs, updating.EVERY)
    out = api.update_notice()
    assert out["state"] == "newer" and out["release"]["version"] == "1.0.3" and out["not_now"] is False


def test_the_setting_turns_the_check_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    github = GitHub(monkeypatch)
    api = Api(ROOT, runs)
    assert api.home()["prefs"].get(updating.CHECK_PREF, True) is True, "on unless turned off"
    api.set_pref(updating.CHECK_PREF, False)
    assert api.update_notice() == {"state": "off"}
    assert github.asked == 0 and not (runs / ".dashboard" / updating.NOTICE).exists()
    api.set_pref(updating.CHECK_PREF, True)
    assert api.update_notice()["state"] == "newer" and github.asked == 1


def test_the_buttons_check_keeps_its_answer_for_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    github = GitHub(monkeypatch)
    monkeypatch.setattr(
        update, "check", lambda root: update.Check(INSTALLED, release("1.0.4"), "newer", False, "")
    )
    api = Api(ROOT, runs)
    assert api.update_check()["release"]["version"] == "1.0.4"
    assert api.update_notice()["release"]["version"] == "1.0.4" and github.asked == 0, "fresh: not asked"


def test_a_kept_answer_is_read_with_care() -> None:
    whole = {**release("1.0.2").__dict__, "page": "https://example.com/elsewhere"}
    found = updating.kept_release(whole)
    assert found is not None and found.version == "1.0.2" and found.page == "", "only GitHub's pages open"
    assert updating.kept_release({**whole, "notes": 5}) is None
    assert updating.kept_release({"version": "1.0.2"}) is None
    assert updating.kept_release(["1.0.2"]) is None


def test_the_tests_never_reach_github() -> None:
    with pytest.raises(update.UpdateError, match="offline in the tests"):
        updating.ask_github()


# ---------- the page ----------
class Notice:
    """Stands in for update_notice and update_not_now: the answer the page gets, and what it asked."""

    def __init__(self, api: Api, **more: Any) -> None:
        self.answer: dict[str, Any] = {
            "state": "newer",
            "installed": INSTALLED,
            "release": release("1.0.2").__dict__,
            "git": False,
            "not_now": False,
            **more,
        }
        self.calls: list[str] = []
        api.update_notice = self.update_notice  # type: ignore[method-assign]
        api.update_not_now = self.update_not_now  # type: ignore[method-assign]

    def update_notice(self) -> dict[str, Any]:
        self.calls.append("notice")
        return self.answer

    def update_not_now(self, version: str) -> dict[str, Any]:
        self.calls.append(f"not now {version}")
        self.answer = {**self.answer, "not_now": True}
        return {"not_now": version}


DOT = "document.querySelector('.nav [data-go=prefs]').classList.contains('has-news')"


def test_the_garage_shows_a_new_version_and_not_now_hides_it(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "striker", saves=1)
    api = quiet(Api(ROOT, runs))
    notice = Notice(api)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.wait_for_selector("#news")
            assert notice.calls == ["notice"], "asked once when the page opens"
            line = tab.text_content("#news")
            assert "NitroLearn 1.0.2 is out." in line and "You have 1.0.1." in line and "git" not in line
            assert tab.evaluate(DOT)
            assert "1.0.2 is out" in tab.get_attribute(".nav [data-go=prefs]", "title")
            news, hero = tab.locator("#news").bounding_box(), tab.locator("#g-list").bounding_box()
            assert news is not None and hero is not None and news["y"] < hero["y"], "on top of the Garage"

            tab.click("#news-notes")
            assert "What's new in NitroLearn 1.0.2" in tab.text_content(".modal")
            assert tab.text_content("#news-text") == "Faster start\n- The Garage opens at once"
            opened_urls: list[str] = []
            api.open_url = lambda url: opened_urls.append(url) or {"opened": True}  # type: ignore[method-assign]
            tab.click("#news-page")
            tab.wait_for_timeout(200)
            assert opened_urls == [notice.answer["release"]["page"]]
            tab.click("[data-close]")

            tab.click("#news-later")
            tab.wait_for_selector("#news", state="detached")
            assert not tab.evaluate(DOT)
            assert notice.calls[-1] == "not now 1.0.2"
            again = opened(browser, api, errors)
            again.wait_for_timeout(400)
            assert again.locator("#news").count() == 0 and not again.evaluate(DOT), "remembered"
        finally:
            browser.close()
    assert errors == []


def test_while_a_bot_trains_only_the_dot_shows(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    mark_running(make_bot(runs, "racer", saves=1))
    api = quiet(Api(ROOT, runs))
    Notice(api, git=True)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.wait_for_function(DOT)
            assert tab.locator("#news").count() == 0, "no line over the training bot"
            tab.click(".nav [data-go=prefs]")
            tab.wait_for_selector("#upd-check")
            assert "NitroLearn 1.0.2 is out." in tab.text_content("#upd-box")
            assert tab.text_content("#upd-check") == "See the update"
        finally:
            browser.close()
    assert errors == []


def test_update_now_takes_app_settings_steps(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "striker", saves=1)
    api = quiet(Api(ROOT, runs))
    Notice(api, git=True)
    updates = Updates(api)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.wait_for_selector("#news")
            assert "This folder is a git checkout: updating moves it to v1.0.2 with git." in tab.text_content(
                "#news"
            )
            tab.click("#news-update")
            tab.wait_for_selector(".modal >> text=Update to NitroLearn 1.1.0?")
            assert updates.calls == ["check"], "a fresh check first, as App settings' button does"
            assert tab.evaluate("S.view") == "prefs" and tab.is_visible("#upd-now")
            tab.click("[data-close]")

            # Something stands in the way (a bot trains): App settings says why, and no dialog opens.
            updates.check = answer(busy="Stop training first: racer is training.")
            tab.click(".nav [data-go=home]")
            tab.click("#news-notes")
            tab.click(".modal [data-act=news-update]")
            tab.wait_for_selector("#upd-blocked >> text=racer is training")
            assert tab.is_hidden("#modal") and updates.calls == ["check", "check"]
        finally:
            browser.close()
    assert errors == []


def test_the_switch_in_app_settings_turns_the_check_off_and_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "striker", saves=1)
    github = GitHub(monkeypatch)
    api = quiet(Api(ROOT, runs))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.wait_for_selector("#news")
            assert github.asked == 1
            tab.click(".nav [data-go=prefs]")
            switch = "#pref-check-updates"
            assert tab.is_checked(switch)
            assert "at most once a day" in tab.text_content("#upd-sub")
            assert "Nothing about you or this PC is sent" in tab.text_content("#update")
            tab.click(switch)
            tab.wait_for_function(f"!{DOT}")
            assert api.home()["prefs"][updating.CHECK_PREF] is False
            assert tab.text_content("#upd-sub") == "Checked only when you press the button."

            again = opened(browser, api, errors)
            again.wait_for_timeout(400)
            assert again.locator("#news").count() == 0 and github.asked == 1
            again.click(".nav [data-go=prefs]")
            assert not again.is_checked(switch)
            again.click(switch)
            again.wait_for_function(DOT)
            assert api.home()["prefs"][updating.CHECK_PREF] is True and github.asked == 1, "kept answer"
        finally:
            browser.close()
    assert errors == []
