"""The dashboard's Replays in training tab (`[replay_mix]`): every
setting of copying replays while the bot trains in a tab of its own, how many converted replays of the bot's
size there are to copy (a warning when none), and the newest copy steps from the run's metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import replays
from nitro.dashboard.api import Api
from nitro.runs import RunFolder
from test_dashboard import make_bot

ROOT = Path(__file__).parents[1]
HEADER = (
    "agent_steps,iteration,replay_mix_steps,replay_mix_loss,replay_mix_agreement,replay_mix_wait_seconds\n"
)


def _converted(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(count):
        (folder / f"r{k}{replays.STORED}").write_bytes(b"")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(replays, "OUT", tmp_path / "project-replays")
    return replays.store_folder()


def test_the_tab_counts_the_replays_to_copy_and_reads_the_newest_copy_steps(
    tmp_path: Path, store: Path
) -> None:
    folder = make_bot(tmp_path / "runs", "mixer", saves=1)
    api = Api(ROOT, tmp_path / "runs")
    _converted(store / "2v2", 2)
    out = api.replay_mix("mixer")
    src = out["sources"]
    assert src["sizes"] == ["1v1"] and not src["named"] and src["total"] == 0, "a 1v1 bot: none of its size"
    assert src["folders"][0]["missing"] and src["store_counts"] == {"2v2": 2}
    assert out["progress"] is None and set(out["guide"]) == {
        f"replay_mix.{k}" for k in ("enabled", "share", "lr", "minibatch", "kickoff_only_seconds", "replays")
    }
    _converted(store / "1v1" / "inner", 3)
    assert api.replay_mix("mixer")["sources"]["total"] == 3, "subfolders count, as the copy steps take them"
    mine = tmp_path / "mine"
    _converted(mine, 4)
    api.stage("mixer", "replay_mix.replays", [str(mine), str(mine / "gone")])
    src = api.replay_mix("mixer")["sources"]
    assert src["named"] and src["total"] == 4, "the waiting change is counted"
    assert [f["replays"] for f in src["folders"]] == [4, 0] and src["folders"][1]["missing"]
    api.stage("mixer", "replay_mix.enabled", True)
    api.stage("mixer", "replay_mix.lr", 0.001)
    assert any("replay_mix.lr" in note for note in api.replay_mix("mixer")["advice"]), "not below policy_lr"
    folder.metrics.write_text(HEADER + "100,1,2,1.5,0.25,0.0\n200,2,0,,,\n", encoding="utf-8")
    progress = api.replay_mix_progress("mixer")["progress"]
    assert progress["newest"] == {
        "agent_steps": 200,
        "steps": 0,
        "loss": None,
        "agreement": None,
        "wait_seconds": None,
    }
    assert progress["last"]["agent_steps"] == 100 and progress["last"]["loss"] == 1.5


def test_the_tab_in_the_page(tmp_path: Path, store: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    folder = make_bot(runs, "mixer", saves=2)
    _converted(store / "2v2", 2)
    mine = tmp_path / "mine"
    _converted(mine, 3)
    api = Api(ROOT, runs)
    api.choose_replay_folder = lambda: {"folder": str(mine)}  # type: ignore[method-assign]  # no window
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
            tab.click("text=mixer")
            tab.click("[data-tab=settings]")
            tab.click("[data-setgroup=replay_mix]")
            tab.wait_for_selector("#set-main >> text=Open the Replays in training tab")
            assert tab.locator("#set-main [data-item^='replay_mix.']").count() == 2, "its two basic settings"
            tab.fill("#settings-search", "copy step")
            tab.wait_for_selector("text=Copying replays while training has its own tab.")
            tabs = tab.locator(".tabs button").all_text_contents()
            assert tabs.index("Replays in training") == tabs.index("Starts") + 1, "beside Starts"
            tab.click("[data-tab=replays]")
            for key in ("enabled", "share", "lr", "minibatch", "kickoff_only_seconds", "replays"):
                tab.wait_for_selector(f"[data-item='replay_mix.{key}']")
            assert not tab.locator("#rm-settings [disabled]").count(), "none is fixed on a bot with saves"
            # No 1v1 replays in the store: said plainly, with the sizes it does hold.
            warning = tab.inner_text("#rm-store")
            assert "no converted replays of its size (1v1)" in warning and "2 2v2 replays" in warning
            tab.click("[data-item='replay_mix.share'] [data-help]")
            assert "What it does:" in tab.inner_text("[data-item='replay_mix.share'] .guide")
            tab.click("[data-act=rm-guide]")
            assert "Decisions per copy step" in tab.inner_text(".modal")
            tab.click("[data-close]")
            tab.click("[data-item='replay_mix.enabled'] input")
            tab.fill("[data-item='replay_mix.share'] input", "0.3")
            tab.press("[data-item='replay_mix.share'] input", "Enter")
            tab.wait_for_selector("[data-item='replay_mix.share'].changed")
            tab.wait_for_selector("#rm-head >> text=On, 30%")
            tab.click("[data-act=rm-add]")
            tab.wait_for_selector("#rm-store >> text=Its folders hold 3 converted replays.")
            tab.wait_for_selector("[data-item='replay_mix.replays'] >> text=3 replays")
            assert tab.inner_text("[data-tab=replays] .count") == "3"
            assert not tab.locator("[data-tab=settings] .count").count()
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
            chosen = RunFolder(runs / "mixer").load_settings().replay_mix
            assert chosen.enabled and chosen.share == 0.3 and chosen.replays == (mine,)
            tab.wait_for_selector("#rm-progress", state="detached")
            assert "Once the bot trains with this on" in tab.inner_text("#rm-aside")
            folder.metrics.write_text(HEADER + "50000,1,2,1.234,0.5,0.25\n", encoding="utf-8")
            tab.evaluate("refreshBot()")
            tab.wait_for_selector("#rm-progress >> text=1.234")
            aside = tab.text_content("#rm-progress")
            assert "Copy steps per update" in aside and "50.0%" in aside and "0.3 s" in aside
            assert "50.0K agent steps" in aside
        finally:
            browser.close()
    assert errors == []
