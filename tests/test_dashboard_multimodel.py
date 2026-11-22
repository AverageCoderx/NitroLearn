"""The dashboard's multi-model bots: the form lists the user's bots with their kind, size
and any problem in plain words, checks the shares, makes a bot with no settings file edited by hand, and a
multi-model bot's Overview shows each model's share of the driving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from browser import open_chromium, playwright_or_skip
from nitro.dashboard import multimodel
from nitro.dashboard.api import Api
from nitro.runs import RunFolder
from test_dashboard import ROOT, TINY, make_bot

ALONE = TINY.replace("[env]\n", "[env]\nopponents = false\n")
BIG = TINY.replace("[env]\n", "[env]\nteam_size = 2\n")
SLOW = TINY.replace("[env]\n", "[env]\ntick_skip = 8\n")


def bots(runs: Path) -> None:
    make_bot(runs, "full", TINY, saves=1)
    make_bot(runs, "aerial", ALONE, saves=1)
    make_bot(runs, "duo", BIG, saves=1)
    make_bot(runs, "slow", SLOW, saves=1)
    make_bot(runs, "new", TINY)


def test_the_form_lists_each_bot_with_its_kind_size_and_problems_in_plain_words(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    bots(runs)
    found = {b["name"]: b for b in multimodel.candidates(runs, "full")["bots"]}
    assert found["full"]["problems"] == [] and found["aerial"]["problems"] == []
    assert found["aerial"]["game"] == "1 car alone" and found["full"]["kind"] == "plain network"
    assert found["full"]["parameters"] > 0
    assert "2v2" in found["duo"]["problems"][0]
    assert "tick_skip must match" in found["slow"]["problems"][0]
    assert found["new"]["problems"] == ["it has no save yet: train it first"]
    unchosen = {b["name"]: b for b in multimodel.candidates(runs)["bots"]}
    assert unchosen["duo"]["problems"] == [], "no game chosen yet: only a bot's own problems"
    labels = {f["path"]: f for f in multimodel.fields()}
    assert labels["selector.min_share"]["more"].startswith("What it does:")


def test_a_multi_model_bot_is_made_from_the_form_with_its_base_bots_game(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    bots(runs)
    form: dict[str, Any] = {
        "models": ["full", "aerial"],
        "base": "full",
        "selector_size": "medium",
        "min_share": [0, 20],
        "min_hold_seconds": 0.3,
        "switch_cost": 0.01,
        "train_models": True,
        "model_pull": 0.2,
        "models_start_after": 5_000_000,
        "goal_heads": True,
        "attention": False,
    }
    out = multimodel.create(runs, "combo", form)
    assert out["made"] == "combo"
    config = RunFolder(runs / "combo").load_settings()
    s = config.selector
    assert s.specialists == ("full", "aerial") and s.min_share == (0.0, 20.0)
    assert (s.min_hold_seconds, s.switch_cost, s.train_models, s.model_pull, s.goal_heads) == (
        0.3,
        0.01,
        True,
        0.2,
        True,
    )
    assert s.models_start_after == 5_000_000, "the selector's warm-up"
    plain = {k: v for k, v in form.items() if k != "models_start_after"}
    assert multimodel.create(runs, "suggested", plain)["made"] == "suggested"
    suggested = RunFolder(runs / "suggested").load_settings().selector.models_start_after
    assert suggested == multimodel.SELECTOR_WARMUP, "the page's suggestion when none is given"
    assert config.model.policy == (256, 256)
    assert config.ppo.policy_lr == 3e-4, "the base bot's other settings"
    bad = dict(form, min_share=[60, 40])
    assert "errors" in multimodel.create(runs, "too-much", bad) and not (runs / "too-much").exists()


def test_the_page_makes_a_multi_model_bot_and_its_overview_shows_the_shares(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    bots(runs)
    api = Api(ROOT, runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            from nitro.dashboard.app import page

            tab = browser.new_page(viewport={"width": 1440, "height": 920})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.wait_for_selector(".nav [data-go=home]")
            tab.wait_for_function("typeof homeBots === 'function' && homeBots().length >= 2")
            tab.evaluate("newBotWizard()")
            tab.click("#wz-kind-multi")  # the new-bot page's way in
            tab.wait_for_selector("[data-mmpick=full]")
            assert tab.is_disabled("[data-act=mm-make]"), "nothing picked yet"
            tab.check("[data-mmpick=full]")
            tab.check("[data-mmpick=aerial]")
            tab.wait_for_selector("[data-mmshare=aerial]")
            tab.wait_for_function(
                "document.querySelector('label[for=mm-pick-duo]').textContent.includes('2v2')"
            )
            assert "no save yet" in tab.text_content("label[for=mm-pick-new]")
            tab.fill("[data-mmshare=aerial]", "120")
            tab.wait_for_function(
                "document.querySelector('.mm-page').textContent.includes('keep them below 100%')"
            )
            tab.fill("[data-mmshare=aerial]", "15")
            tab.fill("#mm-name", "combo")
            tab.wait_for_function("!document.querySelector('[data-act=mm-make]').disabled")
            tab.click("[data-act=mm-make]")
            tab.wait_for_selector("[role=tab][data-tab=overview]")
            config = RunFolder(runs / "combo").load_settings()
            assert config.selector.specialists == ("full", "aerial") and config.selector.min_share == (
                0.0,
                15.0,
            )
            metrics = RunFolder(runs / "combo").metrics
            metrics.write_text(
                "agent_steps,steps_per_second,selector_switches_per_minute,selector_share_full,selector_share_aerial,"
                "selector_bonus_full,selector_bonus_aerial\n1000,10,12.5,0.95,0.05,0,0.02\n",
                encoding="utf-8",
            )
            tab.evaluate("openBot('combo', 'overview')")
            tab.wait_for_selector(".mm-live")
            text = tab.text_content(".mm-live")
            assert (
                "aerial" in text
                and "aim: at least 15%" in text
                and "below its aim" in text
                and "12.5" in text
            )
        finally:
            browser.close()
    assert errors == []


def test_the_overview_names_models_given_by_path_as_training_does(tmp_path: Path) -> None:
    """A model written as a path is named after its run folder in the metrics."""
    import dataclasses

    runs = tmp_path / "runs"
    bots(runs)
    multimodel.create(runs, "combo", {"models": ["full", "aerial"], "base": "full"})
    folder = RunFolder(runs / "combo")
    config = folder.load_settings()
    paths = (str(runs / "full"), str(runs / "aerial"))
    config = dataclasses.replace(config, selector=dataclasses.replace(config.selector, specialists=paths))
    folder.metrics.write_text(
        "agent_steps,selector_share_full,selector_share_aerial\n1000,0.75,0.25\n", encoding="utf-8"
    )
    shown = multimodel.live(folder, config)
    assert shown is not None
    assert [m["share"] for m in shown["models"]] == [0.75, 0.25]
    assert [m["name"] for m in shown["models"]] == list(paths)


def test_what_a_multi_model_bot_cannot_do_yet_is_switched_off_from_its_base_bots_settings(
    tmp_path: Path,
) -> None:
    """A base bot with replays copied while it trains still makes a multi-model bot from the form: they
    are switched off in the new bot only."""
    runs = tmp_path / "runs"
    bots(runs)
    text = TINY + "\n[replay_mix]\nenabled = true\n"
    make_bot(runs, "busy", text, saves=1)
    out = multimodel.create(runs, "combo", {"models": ["busy", "full"], "train_models": True})
    assert out.get("made") == "combo", out
    config = RunFolder(runs / "combo").load_settings()
    assert not config.replay_mix.enabled
    base = RunFolder(runs / "busy").load_settings()
    assert base.replay_mix.enabled, "the base is unchanged"
