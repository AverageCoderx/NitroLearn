"""Transfer learning in the dashboard: a switch in the Settings tab, read-only after the
first save, a start button that says what will happen, transfers kept in run.json by --from, new rewards
switching a transfer not started yet to the policy only, and real starts that take the other bot's
networks only while the switch is on."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.cli import main
from nitro.dashboard import bots
from nitro.dashboard.api import Api
from nitro.runs import RunFolder
from test_dashboard import NEEDS_MESHES, ROOT, TINY, items, make_bot

INHERITED = (
    "# first started from elder, its policy only\n"
    '[start_from]\nrun = "elder"  # how first itself started\n'
    'parts = "policy"\n'
)


def save(folder: RunFolder, steps: int, started_from: str | None = None, **kept: Any) -> None:
    """A save with what the dashboard reads of one (a trainer's holds much more)."""
    path = folder.checkpoints / f"{steps:012d}"
    path.mkdir(parents=True)
    state = {
        "agent_steps": steps,
        "started_from": started_from,
        "started_from_replays": bool(kept.get("from_replays")),
        "learner": {"model": {"policy.0.weight": torch.zeros(4, 4), "value.0.weight": torch.zeros(4, 4)}},
    }
    torch.save(state, path / "state.pt")


def start_section(folder: RunFolder) -> dict[str, Any]:
    table: dict[str, Any] = tomllib.loads(folder.settings.read_text(encoding="utf-8")).get("start_from", {})
    return table


def change_a_reward(api: Api, name: str) -> None:
    shown = api.rewards(name)["rewards"]
    entries = [{"from": e["from"], "view": dict(e["view"])} for e in shown]
    entries[0]["view"]["weight"] = float(entries[0]["view"]["weight"]) + 1.0
    assert api.stage_rewards(name, entries)["changes"]


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    make_bot(runs, "first", TINY + INHERITED, saves=2)  # saves at 1,000 and 2,000 agent steps
    return runs


def test_a_copy_writes_its_transfer_into_its_settings_and_the_switch_turns_it_on_and_off(runs: Path) -> None:
    api = Api(ROOT, runs)
    assert api.copy_bot("first", "second", True) == {"made": "second", "notes": []}
    second = RunFolder(runs / "second")
    assert "start_from" not in json.loads(second.info.read_text(encoding="utf-8"))
    # first's own start (from elder, policy only) is not what the copy does.
    assert start_section(second) == {"enabled": True, "run": "first", "at": 2000, "parts": "all"}
    text = second.settings.read_text(encoding="utf-8")
    assert "# a tiny bot for the dashboard's tests" in text and "elder" not in text, "first's own start went"
    shown = api.bot("second")
    start = shown["start"]
    assert (start["kind"], start["source"], start["steps"], start["done"]) == ("from", "first", 2000, False)
    assert (
        shown["start_word"] == "Start from first (save at 2.0K steps)"
        and start["chip"] == "Starts from first"
    )
    assert "takes everything first learned" in start["line"] and not start["problem"]
    found = items(shown["settings"])
    switch = found["start_from.enabled"]
    assert switch["label"] == "Transfer learning" and switch["value"] is True and not switch["fixed"]
    assert [c["value"] for c in shown["transfer"]["choices"]] == ["first"]
    assert shown["transfer"]["choices"][0]["saves"] == [2000, 1000], "newest first"

    assert api.stage("second", "start_from.enabled", False)["pending"] == {"start_from.enabled": False}
    assert api.apply("second") == {"applied": 1, "restarting": False}
    off = api.bot("second")
    assert off["start_word"] == "Start training from scratch" and off["start"]["kind"] == "scratch"
    assert "Switch it on to start from first" in off["start"]["line"]
    assert second.load_settings().start_from.source == "" and start_section(second)["run"] == "first"
    api.stage("second", "start_from.enabled", True)
    api.stage("second", "start_from.at", "1000")
    api.apply("second")
    assert api.bot("second")["start_word"] == "Start from first (save at 1.0K steps)"

    assert api.copy_bot("first", "plain", False) == {"made": "plain", "notes": []}
    plain = RunFolder(runs / "plain")
    assert start_section(plain)["enabled"] is False and start_section(plain)["run"] == "first"
    assert api.bot("plain")["start_word"] == "Start training from scratch"
    make_bot(runs, "fresh")
    assert "no save yet" in api.copy_bot("fresh", "fresh-2", True)["error"]
    assert api.bot("fresh")["start_word"] == "Start training from scratch"


def test_a_copy_with_transfer_does_not_also_copy_replays_first(runs: Path) -> None:
    make_bot(runs, "copier", TINY + "[pretrain]\nenabled = true\n", saves=1)
    api = Api(ROOT, runs)
    made = api.copy_bot("copier", "copier-2", True)
    assert made["notes"] == ["It starts from copier, so it does not copy replays first ([pretrain] is off)."]
    copy = RunFolder(runs / "copier-2").load_settings()
    assert not copy.pretrain.enabled and copy.start_from.source == "copier" and not copy.problems()
    assert api.copy_bot("copier", "copier-3", False)["notes"] == []
    assert api.bot("copier-3")["start_word"] == "Copy replays", "the same settings: it copies replays first"


def test_after_the_first_save_where_it_started_is_read_only(runs: Path) -> None:
    api = Api(ROOT, runs)
    api.copy_bot("first", "second", True)
    second = RunFolder(runs / "second")
    save(second, 5000, "first at 2,000 agent steps")
    shown = api.bot("second")
    start = shown["start"]
    assert (start["kind"], start["source"], start["steps"], start["done"]) == ("from", "first", 2000, True)
    assert shown["start_word"] == "Carry on training" and start["chip"] == "Started from first"
    assert (
        "Started from first's save at 2,000 steps." in start["line"]
        and "can no longer change" in start["line"]
    )
    assert shown["transfer"]["choices"] == []
    found = items(shown["settings"])
    for name in ("enabled", "run", "at", "parts"):
        assert found[f"start_from.{name}"]["fixed"], name
        assert "own trained networks" in found[f"start_from.{name}"]["why_fixed"]
    assert "own trained networks" in api.stage("second", "start_from.enabled", False)["error"]

    solo = make_bot(runs, "solo")
    save(solo, 3000)
    alone = api.bot("solo")
    assert alone["start"]["chip"] == "Started from scratch" and alone["start_word"] == "Carry on training"
    copied = make_bot(runs, "copied", TINY + "[pretrain]\nenabled = true\n")
    save(copied, 0, "a copy of the players in your replays ([pretrain])", from_replays=True)
    assert api.bot("copied")["start_word"] == "Start training from the copy"
    assert api.bot("copied")["start"]["chip"] == "Started from your replays"


def test_a_transfer_in_run_json_still_counts_off_in_the_settings_wins_and_an_edit_moves_it(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = runs.parent / "older.toml"
    settings.write_text(TINY, encoding="utf-8")
    older = RunFolder.create(runs, "older", settings, {"run": "first", "at": 0})  # as --from writes it
    api = Api(ROOT, runs)
    shown = api.bot("older")
    assert older.load_settings().start_from.source == "first"
    assert shown["start_word"] == "Start from first (save at 2.0K steps)", "its newest save, shown"
    assert items(shown["settings"])["start_from.run"]["value"] == "first", "shown as the file's would be"

    settings.write_text(TINY + '[start_from]\nrun = "other"\n', encoding="utf-8")
    named = RunFolder.create(runs, "named", settings, {"run": "first", "at": 0})
    assert named.load_settings().start_from.source == "first", "run.json's run is laid over the file's"
    settings.write_text(TINY + "[start_from]\nenabled = false\n", encoding="utf-8")
    switched_off = RunFolder.create(runs, "switched-off", settings, {"run": "first", "at": 0})
    config = switched_off.load_settings()
    assert config.start_from.run == "first" and config.start_from.source == "", "off wins"
    assert api.bot("switched-off")["start_word"] == "Start training from scratch"

    assert api.stage("older", "start_from.parts", "policy")["pending"]
    api.apply("older")
    assert "start_from" not in json.loads(older.info.read_text(encoding="utf-8")), "moved into the file"
    assert start_section(older) == {"run": "first", "at": 0, "parts": "policy"}
    assert older.load_settings().start_from.source == "first"

    assert main(["train", str(settings), "--from", "first", "--runs", str(runs), "--name", "cli"]) == 1
    assert "enabled = false" in capsys.readouterr().err
    assert not (runs / "cli").exists(), "--from with the file's transfer switched off is refused"


def test_new_rewards_switch_a_transfer_not_started_yet_to_the_policy_only(runs: Path) -> None:
    api = Api(ROOT, runs)
    api.copy_bot("first", "second", True)
    change_a_reward(api, "second")
    out = api.apply("second")
    assert "will take only first's policy" in out["switched"]
    second = RunFolder(runs / "second")
    assert second.load_settings().start_from.parts == "policy"
    shown = api.bot("second")
    assert shown["transfer"]["note"] == out["switched"], "the Settings tab says so too"
    assert "only first's policy" in shown["start"]["line"]
    change_a_reward(api, "second")
    assert "switched" not in api.apply("second"), "already the policy only"
    api.stage("second", "start_from.parts", "all")
    api.apply("second")
    assert api.bot("second")["transfer"]["note"] == "", "changed back: the note goes"

    api.copy_bot("first", "chosen", True)
    api.stage("chosen", "start_from.parts", "matching")
    change_a_reward(api, "chosen")
    assert "switched" not in api.apply("chosen"), "the user's own choice in the same Apply wins"
    assert RunFolder(runs / "chosen").load_settings().start_from.parts == "matching"

    for name, prepare in (("off", "off"), ("saved", "saved")):
        api.copy_bot("first", name, True)
        if prepare == "off":
            api.stage(name, "start_from.enabled", False)
            api.apply(name)
        else:
            save(RunFolder(runs / name), 3000, "first at 2,000 agent steps")
        change_a_reward(api, name)
        assert "switched" not in api.apply(name), name
        assert RunFolder(runs / name).load_settings().start_from.parts == "all", name


@NEEDS_MESHES
def test_a_real_start_takes_the_other_bots_networks_only_while_the_switch_is_on(tmp_path: Path) -> None:
    from nitro.training import TrainingRun

    runs = tmp_path / "runs"
    first = make_bot(runs, "first")
    run = TrainingRun(first.load_settings(), first)
    run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    trained = {k: v.cpu().clone() for k, v in run.model.state_dict().items()}
    steps = run.agent_steps

    api = Api(ROOT, runs)
    api.copy_bot("first", "on", False)  # made from scratch, switched on in its Settings tab
    api.stage("on", "start_from.enabled", True)
    api.apply("on")
    api.copy_bot("first", "off", True)  # made with transfer, switched off
    api.stage("off", "start_from.enabled", False)
    api.apply("off")
    settings = runs.parent / "older.toml"
    settings.write_text(TINY + "[start_from]\nenabled = false\n", encoding="utf-8")
    RunFolder.create(runs, "older-off", settings, {"run": "first", "at": 0})

    on = RunFolder(runs / "on")
    started = TrainingRun(on.load_settings(), on)
    assert started.started_from == f"first at {steps:,} agent steps"
    mine = started.model.state_dict()
    assert all(torch.equal(mine[k].cpu(), v) for k, v in trained.items()), "everything taken (parts = all)"
    started.iteration()
    started.checkpoint(continue_after=False)
    started.metrics.close()
    shown = api.bot("on")
    assert shown["start"]["done"] and shown["start"]["source"] == "first" and shown["start"]["steps"] == steps
    assert shown["start_word"] == "Carry on training"

    for name in ("off", "older-off"):
        folder = RunFolder(runs / name)
        scratch = TrainingRun(folder.load_settings(), folder)
        assert scratch.started_from is None, name
        policy = {k: v for k, v in scratch.model.state_dict().items() if k.startswith("policy.")}
        assert not all(torch.equal(v.cpu(), trained[k]) for k, v in policy.items()), f"{name}: its own start"
        scratch.metrics.close()


def test_the_transfer_switch_and_the_start_button_on_the_page(runs: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    api = Api(ROOT, runs)
    api.copy_bot("first", "second", True)
    errors: list[str] = []
    start = "#bothead [data-act=start]"
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1300, "height": 900})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            open_bot(tab, "second")
            tab.wait_for_selector(f"{start}:has-text('Start from first (save at 2.0K steps)')")
            assert tab.text_content("[data-start-chip]") == "Starts from first"
            assert "takes everything first learned" in tab.inner_text("[data-start-line]")
            tab.click("[data-tab=settings]")
            tab.click("[data-setgroup=start_from]")
            assert tab.text_content("#transfer [data-tf-state]") == "On"
            assert tab.is_checked("#f-start_from-enabled") and tab.is_visible("#f-start_from-run")

            tab.click("#f-start_from-enabled")  # off
            tab.wait_for_selector("#pending:not([hidden])")
            assert tab.text_content("#transfer [data-tf-state]") == "Off" and not tab.is_visible(
                "#f-start_from-run"
            )
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector(f"{start}:has-text('Start training from scratch')")
            assert "it starts from scratch" in tab.inner_text("#transfer [data-tf-line]")

            tab.click("#f-start_from-enabled")  # on again, from an older save
            tab.select_option("#f-start_from-at", "1000")
            tab.wait_for_selector("#transfer .setting.changed")
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector(f"{start}:has-text('Start from first (save at 1.0K steps)')")

            change_a_reward(api, "second")  # as the Rewards tab stages it
            tab.evaluate("refreshBot()")
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector(".toast >> text=will take only first's policy")
            tab.click("[data-tab=settings]")
            assert "only first's policy" in tab.inner_text("#transfer [data-tf-note]")
            assert tab.input_value("#f-start_from-parts") == "policy"

            save(RunFolder(runs / "second"), 5000, "first at 1,000 agent steps")
            tab.evaluate("refreshBot()")
            tab.wait_for_selector(f"{start}:has-text('Carry on training')")
            tab.click("[data-tab=settings]")
            assert tab.text_content("#transfer [data-tf-state]") == "Started from first"
            assert "can no longer change" in tab.inner_text("#transfer [data-tf-line]")
            assert tab.is_disabled("#f-start_from-enabled") and tab.is_disabled("#f-start_from-run")
            assert tab.text_content("[data-start-chip]") == "Started from first"
        finally:
            browser.close()
    assert errors == []
    assert bots.summary(RunFolder(runs / "second"))["start_word"] == "Carry on training"
