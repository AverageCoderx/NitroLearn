"""The Settings tab's groups and levels (`nitro.dashboard.layout`), its plain titles, its search, the "Nerd
explanation" of a setting and of a reward, the Self-play card, and staging and Apply."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro import __version__, brand
from nitro.config.schema import NitroConfig
from nitro.dashboard import describe, guide, layout, reward_guide, rewards
from nitro.dashboard.api import Api
from nitro.runs import RunFolder
from test_dashboard import ROOT, TINY, items, make_bot

WRITTEN = (
    "Why it matters: the step size of the optimizer.\n\n```\n[ppo]\npolicy_lr = 1e-4\n```\n\nKeep it small."
)
REWARDS = """
[[rewards]]
name = "touch_ball"
weight = 1

[[rewards]]
name = "strong_touch"
weight = 60
min_speed_kph = 30
zero_sum = { team_spirit = 0.5 }
"""


def shown() -> set[str]:
    """Every setting the editor shows."""
    return set(items(describe.describe(NitroConfig(), "", locked=False)))


def drift() -> tuple[list[str], list[str]]:
    """The settings the editor shows that the layout leaves out, and the ones the layout names that do not
    exist."""
    return sorted(shown() - set(layout.SETTINGS)), sorted(set(layout.SETTINGS) - shown())


def opened(browser: Any, api: Api, errors: list[str], **page_options: Any) -> Any:
    """The dashboard's page in a new tab, talking to `api`."""
    from nitro.dashboard.app import page

    tab = browser.new_page(**page_options)
    tab.on("pageerror", lambda e: errors.append(str(e)))
    tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
    tab.add_init_script("window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});")
    tab.goto("about:blank")
    tab.set_content(page(api))
    return tab


def test_the_layout_places_every_setting_the_editor_shows_and_no_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert drift() == ([], []), "add the missing settings to layout.SETTINGS, take the stale ones out"
    assert not set(layout.NOT_SHOWN) & (shown() | set(layout.SETTINGS)), "a setting left out stays out"
    assert all(layout.NOT_SHOWN.values()), "each says why"
    groups = [group.id for group in layout.GROUPS]
    assert groups[0] == "basics" and len(set(groups)) == len(groups)
    assert all(group.title and group.about for group in layout.GROUPS)
    for path, place in layout.SETTINGS.items():
        assert place.group in groups[1:], f"{path} is in a group the tab does not list: {place.group}"
        assert place.level in ("basic", "advanced"), path
    assert {place.group for place in layout.SETTINGS.values()} == set(groups[1:]), "no group is empty"
    assert 8 <= len(layout.BASICS) <= 12 and len(set(layout.BASICS)) == len(layout.BASICS)
    for path in layout.BASICS:
        assert layout.SETTINGS[path].level == "basic", f"{path} is in Basics but advanced"
    found = items(describe.describe(NitroConfig(), "", locked=False))
    titles = [item["title"] or item["label"] for item in found.values()]
    assert len(set(titles)) == len(titles), "two settings must not share a title"

    # The check itself notices a setting left out and a line that names none.
    changed = {path: place for path, place in layout.SETTINGS.items() if path != "ppo.policy_lr"}
    monkeypatch.setattr(layout, "SETTINGS", {**changed, "ppo.no_such_setting": layout.advanced("learning")})
    assert drift() == (["ppo.policy_lr"], ["ppo.no_such_setting"])
    left_out = items(describe.describe(NitroConfig(), "", locked=False))["ppo.policy_lr"]
    assert (left_out["group"], left_out["level"], left_out["title"]) == ("learning", "advanced", ""), (
        "a setting not placed yet still shows, as advanced, in its section's group"
    )


def test_group_level_title_and_nerd_text_reach_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = items(describe.describe(NitroConfig(), "", locked=False))
    lr = found["ppo.policy_lr"]
    assert (lr["group"], lr["level"], lr["title"], lr["label"]) == (
        *("learning", "basic", "Learning rate", "Policy learning rate"),
    )
    assert found["env.num_arenas"]["title"] == "Games played at once"
    assert found["env.tick_skip"]["title"] == "Physics ticks per decision"
    assert found["ppo.epochs"]["level"] == "advanced" and found["ppo.optimizer"]["title"] == ""
    assert sorted(found, key=lambda path: found[path]["order"]) == [p for p in layout.SETTINGS if p in found]
    assert (lr["applies"], found["model.policy"]["applies"], found["start_from.run"]["applies"]) == (
        *("next_start", "saves", "first_start"),
    )
    assert all(item["nerd"] == guide.NERD[path] for path, item in found.items()), "every one written"
    assert all(kind["nerd"] == reward_guide.NERD[kind["name"]] for kind in rewards.catalog())

    monkeypatch.setattr(guide, "NERD", {"ppo.policy_lr": WRITTEN})
    monkeypatch.setattr(reward_guide, "NERD", {"touch_ball": "Pays on contact."})
    found = items(describe.describe(NitroConfig(), "", locked=False))
    assert found["ppo.policy_lr"]["nerd"] == WRITTEN and found["ppo.value_lr"]["nerd"] == ""
    kinds = {kind["name"]: kind for kind in rewards.catalog()}
    assert kinds["touch_ball"]["nerd"] == "Pays on contact." and kinds["goal"]["nerd"] == ""

    api = Api(ROOT, tmp_path / "runs")
    sent = api.settings_layout()
    assert [group["id"] for group in sent["groups"]] == [group.id for group in layout.GROUPS]
    assert sent["groups"][1] == {"id": "learning", "title": "Learning", "about": layout.GROUPS[1].about}
    assert sent["basics"] == list(layout.BASICS)
    about = api.about()
    assert (about["name"], about["credit"], about["version"]) == (brand.NAME, brand.CREDIT, __version__)
    assert "non-commercial" in about["use"] and "Never use a bot online" in about["fair_play"]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert about["licence"].replace(" License", "").replace(" ", "-") == project["license"], (
        "the same licence"
    )
    assert [piece["name"].split(" (")[0] for piece in about["bundled"]] == [
        *("Archivo and JetBrains Mono", "pywebview", "tomlkit", "three.js"),
    ]
    assert all(piece["licence"] for piece in about["bundled"])


def test_the_settings_tab_in_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sync_api = playwright_or_skip()
    monkeypatch.setattr(guide, "NERD", {"ppo.value_lr": WRITTEN})
    monkeypatch.setattr(reward_guide, "NERD", {"touch_ball": "Pays 1 on a touch."})
    runs = tmp_path / "runs"
    saved = make_bot(runs, "shown", TINY.replace("num_arenas = 2", "num_arenas = 64") + REWARDS, saves=1)
    make_bot(runs, "fresh")
    api = Api(ROOT, runs)
    advanced = sum(place.level == "advanced" for place in layout.SETTINGS.values())
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "shown")
            tab.click("[data-tab=settings]")

            # Basics first, by plain titles, the name in the file under each.
            tab.wait_for_selector("[data-setgroup=basics][aria-current=true]")
            assert tab.locator("#set-groups .set-group .t").all_text_contents() == [
                group.title for group in layout.GROUPS
            ]
            assert tab.locator("#set-main .setting").count() == len(layout.BASICS)
            assert tab.text_content("[data-setgroup=basics] .n") == str(len(layout.BASICS))
            row = "[data-item='env.num_arenas']"
            assert tab.text_content(f"{row} .name").startswith("Games played at once")
            assert tab.text_content(f"{row} .path") == "env.num_arenas"
            assert tab.locator("#f-env-num_arenas").input_value() == "64"

            # Basic hides the advanced rows; the switch shows them and is remembered.
            tab.click("[data-setgroup=learning]")
            tab.wait_for_selector("[data-item='ppo.entropy']")
            assert tab.locator("[data-item='ppo.epochs']").count() == 0 and not tab.is_checked(
                "#set-advanced"
            )
            assert f"{advanced} more" in tab.text_content("#set-adv-count")
            assert "advanced settings in this group are hidden" in tab.text_content("#set-main .set-hint")
            assert tab.locator("#set-main .tag >> text=Advanced").count() == 0
            tab.click("#set-advanced")
            tab.wait_for_selector("[data-item='ppo.epochs'] .tag >> text=Advanced")
            assert tab.locator("#set-main .set-hint").count() == 0
            assert api.home()["prefs"]["settings_advanced"] is True, "remembered"
            again = opened(browser, api, errors)
            open_bot(again, "shown")
            again.click("[data-tab=settings]")
            again.click("[data-setgroup=learning]")
            again.wait_for_selector("[data-item='ppo.epochs']")
            assert again.is_checked("#set-advanced"), "the stored choice opens"
            again.close()
            tab.click("#set-advanced")
            tab.wait_for_selector("[data-item='ppo.epochs']", state="detached")
            assert api.home()["prefs"]["settings_advanced"] is False

            # A search finds settings in every group, advanced ones too, by title, name in the file and help.
            tab.fill("#settings-search", "learning rate")
            tab.wait_for_selector("[data-item='replay_mix.lr']")
            heads = tab.locator("#set-main .set-found > .heading h3").all_text_contents()
            assert heads[0] == "Learning" and "Replays in training" in heads and "Games" not in heads
            assert tab.locator("[data-item='ppo.policy_lr']").count() == 1
            tab.fill("#settings-search", "selfplay.pool")
            tab.wait_for_selector("[data-item='selfplay.pool_size']")
            assert tab.locator("#set-main .setting").count() == 2 and tab.locator(".set-found").count() == 1
            tab.fill("#settings-search", "for keeping options open")
            tab.wait_for_selector("[data-item='ppo.entropy']")
            tab.fill("#settings-search", "no such setting anywhere")
            tab.wait_for_selector("#set-main >> text=No setting matches that search.")
            tab.click("[data-setgroup=games]")
            assert tab.input_value("#settings-search") == "", "picking a group ends the search"
            tab.wait_for_selector("[data-item='env.team_size']")

            # Another part's search names one setting: its group opens, advanced with it, and it shows.
            tab.click("[data-tab=overview]")
            tab.evaluate("S.searchTarget = 'ppo.gae_lambda'")
            tab.click("[data-tab=settings]")
            tab.wait_for_selector("[data-item='ppo.gae_lambda'].set-flash")
            assert tab.get_attribute("[data-setgroup=learning]", "aria-current") == "true"
            assert tab.is_checked("#set-advanced") and tab.evaluate("S.searchTarget") is None
            assert tab.evaluate(
                "(() => { const r = document.querySelector(\"[data-item='ppo.gae_lambda']\")"
                ".getBoundingClientRect(); return r.top >= 0 && r.bottom <= window.innerHeight; })()"
            ), "scrolled into view"
            assert tab.locator("#f-ppo-gae_lambda").count() == 1, "its field keeps its id"

            # The automatic nerd explanation of a number, with its line in the file and the two commands.
            row = "[data-item='ppo.policy_lr']"
            tab.click(f"{row} [data-nerd]")
            facts = tab.text_content(f"{row} .nerd .facts")
            for said in (
                "policy_lr in [ppo]",
                "A number",
                "0.00045",
                "From 0 to 1",
                "From the bot's next start",
            ):
                assert said in facts, said
            assert tab.locator(f"{row} .nerd pre").all_text_contents() == [
                "[ppo]\npolicy_lr = 0.0003",
                f"nitro check-settings {saved.settings}\nnitro resume shown",
            ]
            tab.click(f"{row} .nerd [data-copy] >> nth=0")
            tab.wait_for_selector(".toast")
            tab.click(f"{row} [data-nerd]")
            assert tab.locator(f"{row} .nerd").count() == 0, "it closes again"

            # A written text (guide.NERD) shows above the facts, its fenced lines as a block to copy.
            row = "[data-item='ppo.value_lr']"
            tab.click(f"{row} [data-nerd]")
            assert "Why it matters: the step size of the optimizer." in tab.text_content(f"{row} .nerd-text")
            assert "Keep it small." in tab.text_content(f"{row} .nerd-text")
            assert tab.locator(f"{row} .nerd pre").all_text_contents()[:2] == [
                "[ppo]\npolicy_lr = 1e-4",
                "[ppo]\nvalue_lr = 0.00015",
            ]
            assert "Name in the file" in tab.text_content(f"{row} .nerd .facts")

            # A choice, and a setting that is fixed once the bot has a save.
            tab.click("[data-setgroup=network]")
            row = "[data-item='model.init']"
            tab.click(f"{row} [data-nerd]")
            assert "One of its choices" in tab.text_content(f"{row} .facts")
            assert "orthogonal, pytorch" in tab.text_content(f"{row} .facts")
            assert tab.text_content(f"{row} .nerd pre >> nth=0") == '[model]\ninit = "orthogonal"'
            row = "[data-item='model.policy']"
            assert tab.text_content(f"{row} .tag").strip() == "Fixed" and tab.is_disabled("#f-model-policy")
            tab.click(f"{row} [data-nerd]")
            assert "Fixed: this bot has a save" in tab.text_content(f"{row} .facts")
            assert "A list of numbers" in tab.text_content(f"{row} .facts")
            assert tab.text_content(f"{row} .nerd pre >> nth=0") == "[model]\npolicy = [16]"
            open_bot(tab, "fresh")
            tab.click("[data-tab=settings]")
            tab.click("[data-setgroup=network]")
            assert tab.text_content(f"{row} .tag").strip() == "Fixed once it has a save"
            assert not tab.is_disabled("#f-model-policy")
            tab.click(f"{row} [data-nerd]")
            assert "Until the bot's first save" in tab.text_content(f"{row} .facts")
            # Self-play does not fit this bot's two games: switching it on is refused, saying what to change.
            tab.click("[data-setgroup=selfplay]")
            tab.click("#f-selfplay-enabled")
            tab.wait_for_selector("#selfplay-card .err >> text=raise past_share or env.num_arenas")
            assert not tab.is_checked("#f-selfplay-enabled") and api.bot("fresh")["pending"] == {}

            # Self-play: a card with the switch; the settings that matter show once it is on.
            open_bot(tab, "shown")
            tab.click("[data-tab=settings]")
            tab.click("#set-advanced")
            tab.click("[data-setgroup=selfplay]")
            card = "#selfplay-card"
            assert "against saved past versions" in tab.text_content(f"{card} .help")
            assert not tab.is_checked("#f-selfplay-enabled")
            assert tab.locator("[data-item='selfplay.past_share']").count() == 0
            assert tab.locator("#set-main .setting").count() == 1
            tab.click("#f-selfplay-enabled")
            tab.wait_for_selector(f"{card} [data-item='selfplay.past_share']")
            assert tab.locator(f"{card} .setting").count() == 2 and tab.is_checked("#f-selfplay-enabled")
            assert tab.locator(f"{card} [data-item='selfplay.enabled'].changed .tag.hot").count() == 1
            assert tab.locator("[data-setgroup=selfplay] .mark").count() == 1, (
                "the group shows it holds a change"
            )
            assert tab.locator("[data-setgroup=basics] .mark").count() == 1, "and Basics, which lists it too"
            assert api.bot("shown")["pending"] == {"selfplay.enabled": True}

            # A bad value is still refused in plain words, and nothing is kept of it.
            tab.fill("#f-selfplay-past_share", "5")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("[data-item='selfplay.past_share'] .err >> text=the largest allowed is 0.9")
            assert tab.input_value("#f-selfplay-past_share") == "5", "what was typed stays beside the reason"
            assert api.bot("shown")["pending"] == {"selfplay.enabled": True}
            tab.fill("#f-selfplay-past_share", "0.3")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("[data-item='selfplay.past_share'].changed")

            # Back to default, Undo, a note; then the bar names the changes and Apply writes the file.
            tab.click("[data-setgroup=basics]")
            row = "[data-item='ppo.policy_lr']"
            assert "Default 0.00045" in tab.text_content(f"{row} .control")
            tab.click(f"{row} [data-default]")
            tab.wait_for_selector(f"{row}.changed")
            assert api.bot("shown")["pending"]["ppo.policy_lr"] == 0.00045
            assert "Was 0.0003" in tab.text_content(f"{row} .control")
            tab.click(f"{row} [data-undo]")
            tab.wait_for_selector(f"{row}:not(.changed)")
            assert "ppo.policy_lr" not in api.bot("shown")["pending"]
            tab.click(f"{row} [data-noteopen]")
            tab.fill(f"{row} textarea", "halve it later")
            tab.click(f"{row} [data-savenote]")
            tab.wait_for_selector(f"{row} .notetext >> text=halve it later")
            assert api.bot("shown")["notes"]["setting:ppo.policy_lr"] == "halve it later"
            tab.fill("#f-env-num_arenas", "128")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("[data-item='env.num_arenas'].changed")
            bar = tab.text_content("#pending")
            assert "3 changes waiting" in bar and "Games played at once" in bar
            assert "Play against its past versions" in bar and "Share of games against past versions" in bar
            before = saved.settings.read_text(encoding="utf-8")
            tab.click("[data-act=apply]")
            popup = tab.text_content(".modal")
            assert "Games played at once" in popup and "env.num_arenas" in popup
            assert saved.settings.read_text(encoding="utf-8") == before, "nothing written before Apply"
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
            written = RunFolder(runs / "shown").load_settings()
            assert written.env.num_arenas == 128 and written.selfplay.enabled
            assert written.selfplay.past_share == 0.3 and written.ppo.policy_lr == 0.0003
            assert "# a tiny bot for the dashboard's tests" in saved.settings.read_text(encoding="utf-8")
            assert tab.locator("#set-groups .mark").count() == 0

            # A reward's nerd explanation: its written text, its settings and its block for the file.
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("[data-reward='1']")
            card = "[data-reward='0']"
            tab.click(f"{card} [data-rwnerd]")
            assert "Pays 1 on a touch." in tab.text_content(f"{card} .nerd-text")
            assert "touch_ball, in a [[rewards]] block" in tab.text_content(f"{card} .facts")
            assert (
                tab.text_content(f"{card} .nerd pre >> nth=0")
                == '[[rewards]]\nname = "touch_ball"\nweight = 1.0'
            )
            assert "nitro resume shown" in tab.text_content(f"{card} .nerd pre >> nth=1")
            card = "[data-reward='1']"
            tab.click(f"{card} [data-rwnerd]")
            assert tab.locator(f"{card} .nerd-text").count() == 0, "no written text: the facts alone"
            facts = tab.text_content(f"{card} .facts")
            assert "min_speed_kph" in facts and "default 20.0" in facts and "0 or more" in facts
            assert tab.text_content(f"{card} .nerd pre >> nth=0") == (
                '[[rewards]]\nname = "strong_touch"\nweight = 60.0\nmin_speed_kph = 30.0\n'
                "zero_sum = { team_spirit = 0.5, opponent_scale = 1 }"
            )
        finally:
            browser.close()
    assert errors == []


# Plain words a newcomer types, and the setting both searches put first.
PLAIN_WORDS = {
    "learning rate": "ppo.policy_lr",
    "learning rate replays": "pretrain.lr",
    "speed": "env.num_arenas",
    "how often it saves": "run.checkpoint_minutes",
    "save every": "run.checkpoint_minutes",
    "games at once": "env.num_arenas",
    "arenas": "env.num_arenas",
    "decisions per second": "env.tick_skip",
    "self play": "selfplay.enabled",
    "randomness": "ppo.entropy",
    "network size": "model.policy",
    "replays": "pretrain.enabled",
    "Games played at once": "env.num_arenas",
    "gae lambda": "ppo.gae_lambda",
}


def test_both_searches_find_settings_by_plain_words(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown", saves=1)
    api = Api(ROOT, runs)
    errors: list[str] = []
    rows = "#finder-list [role=option]"
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            open_bot(tab, "shown")
            tab.click("[data-tab=settings]")
            tab.wait_for_selector("[data-setgroup=basics][aria-current=true]")
            for query, first in PLAIN_WORDS.items():
                tab.fill("#settings-search", query)
                assert tab.evaluate("firstMatch()") == first, f"the tab's box: {query}"
                tab.fill("#finder-q", query)
                found = tab.locator(f"{rows}:has(.hint.mono)").first.locator(".hint").text_content()
                assert found == first, f"the top band's search: {query}"
                tab.keyboard.press("Escape")
            # The main training's learning rate leads; the copy's follow, and only the settings that fit.
            tab.fill("#settings-search", "learning rate")
            ranked = tab.evaluate("rankSettings(settingItems(), 'learning rate').map(x => x.item.path)")
            assert ranked[:4] == ["ppo.policy_lr", "ppo.value_lr", "pretrain.lr", "replay_mix.lr"]
            assert tab.get_attribute("#set-main [data-item] >> nth=0", "data-item") == "ppo.policy_lr"
            tab.fill("#settings-search", "speed")
            ranked = tab.evaluate("rankSettings(settingItems(), 'speed').map(x => x.item.path)")
            assert ranked[:3] == ["env.num_arenas", "env.threads", "run.precision"]
            assert "run.console" not in ranked, "a word in passing in a help text does not count"

            # Enter opens the best match in its group and puts the keyboard in its field.
            tab.fill("#settings-search", "games at once")
            tab.press("#settings-search", "Enter")
            tab.wait_for_selector("[data-item='env.num_arenas'].set-flash")
            assert tab.evaluate("document.activeElement.id") == "f-env-num_arenas"
            assert tab.get_attribute("[data-setgroup=games]", "aria-current") == "true"
            assert tab.input_value("#settings-search") == ""
            # An advanced one switches Show advanced on.
            assert not tab.is_checked("#set-advanced")
            tab.fill("#settings-search", "gae lambda")
            tab.press("#settings-search", "Enter")
            tab.wait_for_selector("[data-item='ppo.gae_lambda'].set-flash")
            assert tab.evaluate("document.activeElement.id") == "f-ppo-gae_lambda"
            assert tab.is_checked("#set-advanced")
            # A click on a found setting's name does the same.
            tab.fill("#settings-search", "how often it saves")
            tab.click("#set-main [data-item='run.checkpoint_minutes'] .what .name")
            tab.wait_for_selector("[data-item='run.checkpoint_minutes'].set-flash")
            assert tab.evaluate("document.activeElement.id") == "f-run-checkpoint_minutes"
            assert tab.get_attribute("[data-setgroup=saving]", "aria-current") == "true"
            # The top band's search, the Settings tab open or not.
            for query, field in (
                ("decisions per second", "f-env-tick_skip"),
                ("randomness", "f-ppo-entropy"),
            ):
                tab.click("#finder-q")
                tab.keyboard.type(query)
                tab.keyboard.press("Enter")
                tab.wait_for_function(f"document.activeElement && document.activeElement.id === '{field}'")
            tab.click("[data-tab=overview]")
            tab.click("#finder-q")
            tab.keyboard.type("how often it saves")
            tab.click(f"{rows}:has(.hint.mono) >> nth=0")
            tab.wait_for_function(
                "document.activeElement && document.activeElement.id === 'f-run-checkpoint_minutes'"
            )
        finally:
            browser.close()
    assert errors == []


def test_the_about_block_and_the_resources_sections_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown")
    api = Api(ROOT, runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 900, "height": 640})
            tab.click("#band .nav [data-go=prefs]")
            tab.wait_for_selector("[data-about-version]")
            assert tab.text_content("[data-about-version]") == __version__
            about = tab.text_content("#main")
            assert tab.text_content("[data-about-licence]") == "PolyForm Noncommercial License 1.0.0"
            for said in ("Free for non-commercial use.", "Never use a bot online against real players."):
                assert said in about, said
            for piece in ("Archivo and JetBrains Mono", "pywebview", "tomlkit", "three.js", "MIT License"):
                assert piece in about, piece
            assert tab.locator("#main .about-lockup svg").count() == 1 and tab.locator("#main a").count() == 0
            assert tab.locator("[data-seg=theme] button").count() == 2
            assert tab.locator("[data-act=rlbot-choose]").count() == 1 and str(runs) in about
            assert tab.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (
                "nothing runs off"
            )

            tab.click("#band .nav [data-go=resources]")
            tab.wait_for_selector("#res-rewards")
            titles = tab.locator("#main .set-nav .set-group .t").all_text_contents()
            assert titles == [
                *("Converted replays", "Settings files", "Reward sets", "Your Rust files"),
                *("Built in: rewards", "Built in: observations"),
            ]
            tab.click("[data-resjump=rewards]")
            tab.wait_for_function(
                "document.querySelector('#res-rewards').getBoundingClientRect().top < window.innerHeight / 2"
            )
            assert tab.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        finally:
            browser.close()
    assert errors == []
