"""The dashboard's Starts tab: a bot's game starts ([[state_setters]]) and how often each is picked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.config import load_config
from nitro.dashboard import starts
from nitro.dashboard.api import Api
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]
PRESETS = ROOT / "configs" / "presets"


def _bot(tmp_path: Path) -> tuple[Api, RunFolder]:
    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "b", PRESETS / "2v2_starter.toml")
    return Api(ROOT, runs), folder


def _entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"from": e["from"], "view": e["view"]} for e in state["starts"]]


def test_random_70_kickoff_30_waits_for_apply_and_keeps_the_file(tmp_path: Path) -> None:
    api, folder = _bot(tmp_path)
    before = folder.settings.read_text(encoding="utf-8")
    state = api.starts("b")
    assert [(e["view"]["name"], e["share"]) for e in state["starts"]] == [("kickoff", 1.0)]
    assert {k["name"] for k in state["catalog"]} >= {"kickoff", "random", "replay", "shot"}
    entries = _entries(state)
    entries[0]["view"]["weight"] = "30"
    entries.append({"from": None, "view": {"name": "random", "weight": 70, "params": {}}})
    staged = api.stage_starts("b", entries)
    assert [e["share"] for e in staged["starts"]] == [0.3, 0.7]
    assert staged["changes"] == ["kickoff: weight 1 -> 30", "random: added, weight 70"]
    assert api.bot("b")["start_changes"] == staged["changes"]
    assert folder.settings.read_text(encoding="utf-8") == before, "nothing written before Apply"
    assert "apply or discard" in api.start("b")["error"]
    assert api.apply("b")["applied"] == 2
    after = folder.settings.read_text(encoding="utf-8")
    assert [(s.name, s.weight) for s in load_config(folder.settings).state_setters] == [
        ("kickoff", 30.0),
        ("random", 70.0),
    ]
    kept = [line for line in before.splitlines() if line.startswith("#")]
    assert all(line in after for line in kept), "every comment of the settings file is kept"
    assert api.bot("b")["start_changes"] == [] and api.starts("b")["pending"] is False


def test_problems_are_shown_and_discard_forgets_the_starts(tmp_path: Path) -> None:
    api, folder = _bot(tmp_path)
    entries = _entries(api.starts("b"))
    entries[0]["view"]["weight"] = "lots"
    assert "must be a number" in api.stage_starts("b", entries)["starts"][0]["problems"][0]
    entries[0]["view"]["weight"] = 0
    assert "weight above 0" in api.stage_starts("b", entries)["general"][0]
    assert "fix these first" in api.apply("b")["error"]
    api.discard("b")
    assert api.bot("b")["start_changes"] == []
    assert Api(ROOT, folder.root.parent).starts("b")["pending"] is False, "kept across windows, and cleared"


def test_settings_of_a_start_and_a_list_typed_as_text() -> None:
    text = '[[state_setters]]\nname = "replay"\nfiles = ["a.rlpr"]\n# about the next section\n'
    views, listed = starts.read(text)
    assert listed and views[0]["params"]["files"] == ["a.rlpr"] and views[0]["params"]["bundled"] is False
    views[0]["params"]["files"] = "a.rlpr, b/*.rlpr"
    views[0]["params"]["air_bias"] = "2"
    out = starts.written(text, [{"from": 0, "view": views[0]}])
    assert 'files = ["a.rlpr", "b/*.rlpr"]' in out and "air_bias = 2" in out and "bundled" not in out
    assert out.rstrip().endswith("# about the next section")
    assert starts.changes(starts.read(text)[0], [{"from": 0, "view": views[0]}]) == [
        "replay: files a.rlpr -> a.rlpr, b/*.rlpr",
        "replay: air bias 0 -> 2",
    ]
    assert starts.shares([{"weight": 1}, {"weight": "x"}, {"weight": 3}]) == [0.25, None, 0.75]


def test_the_starts_tab_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    api, folder = _bot(tmp_path)
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
            open_bot(tab)
            tab.click("[data-tab=starts]")
            tab.wait_for_selector("[data-start='0']")
            tab.select_option("#st-add", "random")
            tab.wait_for_selector("[data-start='1'] .badge.new")
            tab.fill("#st-1-weight", "70")
            tab.press("#st-1-weight", "Enter")
            tab.fill("#st-0-weight", "30")
            tab.press("#st-0-weight", "Enter")
            tab.wait_for_function("document.querySelector('#st-mix').textContent.includes('70%')")
            assert "30%" in tab.inner_text("[data-start='0'] .sharechip")
            tab.wait_for_selector("#pending:not([hidden])")
            assert "2" in tab.inner_text("[data-tab=starts] .count")
            tab.click("[data-act=apply]")
            popup = tab.inner_text(".modal")
            assert "Start · kickoff: weight 1 -> 30" in popup and "Reward ·" not in popup, "named as starts"
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
            # The window may turn text selection off for the page (pywebview without text_select): the
            # Console and error text stay selectable.
            tab.add_style_tag(content="body { -webkit-user-select: none; user-select: none; }")
            tab.click("[data-tab=console]")
            assert tab.eval_on_selector("#consoleText", "e => getComputedStyle(e).userSelect") == "text"
        finally:
            browser.close()
    assert errors == []
    assert [(s.name, s.weight) for s in load_config(folder.settings).state_setters] == [
        ("kickoff", 30.0),
        ("random", 70.0),
    ]
