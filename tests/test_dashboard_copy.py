"""The dashboard's Learning from replays card and the copy's progress: the card stages the copy's
settings, the page shows how far a copy is from the file it writes, and the replay folders are counted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import pretrain
from nitro.dashboard import bots
from nitro.dashboard.api import Api
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]
PRESETS = ROOT / "configs" / "presets"


def bot(runs: Path, name: str = "copier") -> RunFolder:
    return RunFolder.create(runs, name, PRESETS / "2v2_starter.toml")


def test_the_page_reads_a_copys_progress_or_what_a_finished_copy_kept(tmp_path: Path) -> None:
    folder = bot(tmp_path / "runs")
    assert bots.copy_view(folder) is None
    work = folder.root / pretrain.COPY
    work.mkdir()
    pretrain._write_live(work, {"learned": 2, "limit": 4, "finished": False, "seconds_left": float("nan")})
    view = bots.copy_view(folder)
    assert view == {"learned": 2, "limit": 4, "finished": False, "seconds_left": None}, "no NaN in JSON"
    (work / pretrain.LIVE).unlink()
    details = {
        "replays": 4,
        "held_out_replays": 2,
        "epochs": [{"epoch": 1, "held_loss": 2.0, "held_agreement": 0.4, "replays": 2}],
        "large": {"batch_replays": 2, "by_size": {"2": 6}},
    }
    (work / "pretrain.json").write_text(json.dumps(details), encoding="utf-8")
    view = bots.copy_view(folder)
    assert view is not None and view["finished"] and view["learned"] == 4
    assert view["rounds"][0]["held_agreement"] == 0.4 and view["skill"] is None
    assert Api(ROOT, tmp_path / "runs").bot("copier")["copy"]["learned"] == 4


def test_replay_folders_are_counted_as_the_copy_finds_them(tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    (one / "inner").mkdir(parents=True)
    two.mkdir()
    for path in (
        one / "a.replay",
        one / "b.replay",
        one / "inner" / "c.replay",
        two / "a.replay",
        two / "d.txt",
    ):
        path.write_bytes(b"")
    out = Api(ROOT, tmp_path / "runs").count_replays([str(one), str(two), str(tmp_path / "gone")])
    assert [f["replays"] for f in out["folders"]] == [2, 1, 0] and out["folders"][2]["missing"]
    assert out["total"] == 2 and not out["default"], "a.replay in both folders counts once"


def test_the_card_switches_the_copy_on_and_picks_folders_sizes_and_skill_checks(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    bot(runs)
    replays = tmp_path / "replays"
    replays.mkdir()
    (replays / "x.replay").write_bytes(b"")
    api = Api(ROOT, runs)
    api.choose_replay_folder = lambda: {"folder": str(replays)}  # type: ignore[method-assign]  # no window
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page()
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click("text=copier")
            tab.click("[data-tab=overview]")
            tab.click("[data-tab=settings]")
            tab.click("[data-setgroup=replays_first]")
            tab.wait_for_selector("#copysetup >> text=Off")
            assert tab.locator("#set-main .setting").count() == 1, "the card, not every setting in rows"
            tab.click("[data-cp=enabled]")
            tab.click("[data-act=cp-add]")
            tab.wait_for_selector("#copysetup >> text=1 replays")
            tab.click("[data-cp=skill]")
            tab.wait_for_selector("[data-item='pretrain.skill_points']")
            assert tab.locator("[data-cpsize='3']").is_disabled(), "no room for 3v3 in this observation"
            tab.click("[data-act=cp-more]")  # what converts the replays, before the copy is made
            tab.select_option("[data-item='pretrain.converter'] select", "replicar")
            tab.wait_for_selector("[data-item='pretrain.converter'].changed")
            pending = api.bot("copier")["pending"]
            assert pending == {
                "pretrain.enabled": True,
                "pretrain.batch_replays": 100,
                "pretrain.replays": [str(replays)],
                "pretrain.skill_every": 10,
                "pretrain.converter": "replicar",
            }
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
            chosen = RunFolder(runs / "copier").load_settings().pretrain
            assert chosen.enabled and chosen.replays == (replays,) and chosen.skill_every == 10
            assert chosen.converter == "replicar"
            tab.click("[data-tab=overview]")
            tab.wait_for_selector("text=Not started yet")
        finally:
            browser.close()
    assert errors == []


def test_the_resources_page_shows_the_kept_replays_and_starts_converting_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Converted replays card on the Resources page: what the store holds, the folders to convert
    (kept for next time), how many, and Convert, which starts `nitro convert-replays`."""
    sync_api = playwright_or_skip()
    from nitro import replays as replays_module
    from nitro.dashboard.app import page

    monkeypatch.setattr(replays_module, "OUT", tmp_path / "project-replays")
    kept = replays_module.stored("old", 2)
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"\0" * 2048)
    runs = tmp_path / "runs"
    folder = tmp_path / "mine"
    folder.mkdir()
    for name in ("a", "b"):
        (folder / f"{name}.replay").write_bytes(b"")
    api = Api(ROOT, runs)
    api.choose_replay_folder = lambda: {"folder": str(folder)}  # type: ignore[method-assign]  # no window
    started: list[tuple[list[str], int]] = []

    def start(folders: list[str], limit: int = 0) -> dict[str, Any]:
        started.append((folders, limit))
        return {"started": True}

    api.convert_start = start  # type: ignore[method-assign]  # not the real converter here
    view = api.converting()
    assert view["store"]["counts"] == {"2v2": 1} and view["state"] == "none" and view["folders"] == []
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page()
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click("[data-go=resources]")
            tab.wait_for_selector("#convert >> text=2v2: 1")
            tab.wait_for_selector("#convert >> text=None chosen")
            tab.click("[data-act=cv-add]")
            tab.wait_for_selector("#convert >> text=2 replays")
            assert api.converting()["folders"] == [str(folder)], "kept for next time"
            tab.fill("#cv-limit", "1,000")
            tab.click("[data-act=cv-start]")
            tab.wait_for_function("() => document.querySelector('#convert') !== null")
            assert started == [([str(folder)], 1000)]
            tab.fill("#cv-limit", "lots")
            tab.click("[data-act=cv-start]")
            tab.wait_for_selector("text=a whole number")
        finally:
            browser.close()
    assert errors == [] and len(started) == 1


def test_converting_refuses_a_missing_folder_and_reads_its_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nitro import replays as replays_module

    monkeypatch.setattr(replays_module, "OUT", tmp_path / "project-replays")
    api = Api(ROOT, tmp_path / "runs")
    assert "not a folder" in api.convert_start([str(tmp_path / "gone")])["error"]
    live = replays_module.converting_file()
    live.parent.mkdir(parents=True)
    live.write_text(json.dumps({"state": "converting", "done": 3, "todo": 10}), encoding="utf-8")
    assert api.converting()["progress"]["done"] == 3
