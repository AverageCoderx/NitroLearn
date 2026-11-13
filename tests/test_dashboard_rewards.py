"""The dashboard's Rewards tab and new-bot form: rewards read from and written back
into a settings file block by block, changes held until Apply, each reward's share of all reward paid with the
farming alerts, a new bot made from the form, and the page itself."""

from __future__ import annotations

import copy
import csv
import difflib
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.config.loader import load_toml, parse_toml
from nitro.config.schema import NitroConfig, large_networks
from nitro.dashboard import rewards
from nitro.dashboard.api import Api
from nitro.dashboard.edits import problems
from nitro.model import observation_room
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]
PRESET = ROOT / "configs" / "presets" / "2v2_starter.toml"


def entries_of(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"from": i, "view": copy.deepcopy(view)} for i, view in enumerate(views)]


def changed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def test_rewards_written_back_change_only_their_lines_and_keep_every_comment() -> None:
    text = PRESET.read_text(encoding="utf-8")
    views, in_file = rewards.read(text)
    names = [view["name"] for view in views]
    assert in_file and names[:4] == ["air", "face_ball", "velocity_player_to_ball", "strong_touch"]
    assert rewards.written(text, entries_of(views)) == text, "nothing changed, nothing written"

    entries = entries_of(views)
    entries[names.index("strong_touch")]["view"]["weight"] = "80"
    entries[names.index("demo")]["view"]["enabled"] = False
    del entries[names.index("bump")]
    entries.append({"from": None, "view": rewards.view({"name": "touch_ball", "weight": 5})})
    after = rewards.written(text, entries)
    assert problems(after, PRESET, PRESET.parent) == []
    assert sorted(changed_lines(text, after)) == sorted(
        [
            "-weight = 60",
            "+weight = 80",
            "+enabled = false",
            "-[[rewards]]",
            '-name = "bump"',
            "-weight = 20",
            "-zero_sum = { team_spirit = 0.5 }",
            "-",
            "+",
            "+[[rewards]]",
            '+name = "touch_ball"',
            "+weight = 5",
        ]
    ), "only the changed lines changed; the comment about start states still follows the rewards"
    touch = after.index('name = "touch_ball"')
    assert touch < after.index("# How games start", touch) < after.index("\n[[state_setters]]", touch)
    config = parse_toml(after, NitroConfig, file=PRESET, base_dir=PRESET.parent)
    assert [r.name for r in config.active_rewards][-1] == "touch_ball"
    assert "demo" in [r.name for r in config.rewards] and "demo" not in [
        r.name for r in config.active_rewards
    ]
    assert rewards.labels([e["view"] for e in entries]).count(None) == 1
    assert rewards.changes(views, entries) == [
        "strong_touch: weight 60 -> 80",
        "demo: switched off",
        "touch_ball: added, weight 5",
        "bump: removed",
    ]


def test_a_bot_without_rewards_written_shows_the_example_rewards_and_writes_them_all_on_a_change() -> None:
    bare = "[env]\nteam_size = 2\n"
    views, in_file = rewards.read(bare)
    defaults = parse_toml(bare, NitroConfig, file=Path("bare.toml"), base_dir=ROOT).rewards
    assert not in_file and [v["name"] for v in views] == [r.name for r in defaults]
    entries = entries_of(views)
    entries[0]["view"]["weight"] = 1
    after = rewards.written(bare, entries)
    config = parse_toml(after, NitroConfig, file=Path("bare.toml"), base_dir=ROOT)
    assert config.rewards[0].weight == 1.0
    assert config.rewards[1:] == defaults[1:], "the rest written exactly as the defaults"


def write_metrics(folder: RunFolder, shares_by_row: list[dict[str, float]], speed: float = 1000.0) -> None:
    """Rows of 1,000 agent steps at `speed` steps a second (one second each), paying `shares_by_row`."""
    labels = list(shares_by_row[0])
    header = ["agent_steps", "iteration", "steps_per_second", "reward_per_step"]
    header += [f"raw_{x}" for x in labels] + [f"paid_{x}" for x in labels] + [f"paid_abs_{x}" for x in labels]
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(header)
        for i, row in enumerate(shares_by_row, 1):
            pay = [row[x] for x in labels]
            out.writerow([i * 1000, i, speed, sum(pay), *pay, *pay, *[abs(p) for p in pay]])


def test_shares_cover_the_window_by_training_time_and_raise_the_farming_alerts(tmp_path: Path) -> None:
    folder = RunFolder.create(tmp_path, "b", PRESET)
    calm = {"touch_ball": 0.5, "goal": 0.3, "air": -0.2}
    write_metrics(folder, [calm] * 300)
    out = rewards.shares(folder, 60)
    assert 60 <= out["seconds"] <= 61 and out["agent_steps"] == pytest.approx(out["seconds"] * 1000)
    assert [r["label"] for r in out["rewards"]] == ["touch_ball", "goal", "air"]
    assert [r["share"] for r in out["rewards"]] == pytest.approx([0.5, 0.3, 0.2])
    assert out["rewards"][2]["paid_per_step"] == pytest.approx(-0.2), "a penalty counts by its size"
    assert out["rewards"][0]["share_before"] == pytest.approx(0.5) and out["alerts"] == []

    farmed = {"touch_ball": 9.5, "goal": 0.3, "air": -0.2}
    write_metrics(folder, [calm] * 240 + [farmed] * 60)
    out = rewards.shares(folder, 60)
    assert out["rewards"][0]["share"] == pytest.approx(0.95)
    assert [a["level"] for a in out["alerts"]] == ["warning"] and "may be farming" in out["alerts"][0]["text"]
    quiet, rising = (
        {"touch_ball": 0.2, "goal": 0.5, "air": -0.3},
        {"touch_ball": 0.8, "goal": 0.5, "air": -0.3},
    )
    write_metrics(folder, [quiet] * 240 + [rising] * 60)
    out = rewards.shares(folder, 60)
    assert [a["level"] for a in out["alerts"]] == ["watch"] and "went from 20% to 50%" in out["alerts"][0][
        "text"
    ]
    assert rewards.shares(RunFolder.create(tmp_path, "new", PRESET), 60)["rewards"] == []


def test_a_zero_sum_reward_shows_its_share_before_zero_sum_too(tmp_path: Path) -> None:
    """Zero-sum can cancel a reward both sides earn; the sidebar shows both sizes."""
    folder = RunFolder.create(tmp_path, "z", PRESET)
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["agent_steps", "iteration", "steps_per_second", "paid_abs_speed", "paid_abs_goal",
                      "own_abs_speed", "own_abs_goal"])  # fmt: skip
        for i in range(1, 100):
            out.writerow([i * 1000, i, 1000.0, 0.1, 0.9, 3.0, 1.0])
    shown = {r["label"]: r for r in rewards.shares(folder, 60)["rewards"]}
    assert shown["speed"]["share"] == pytest.approx(0.1) and shown["speed"]["own_share"] == pytest.approx(
        0.75
    )
    assert shown["speed"]["own_per_step"] == pytest.approx(3.0)


def test_how_often_each_reward_fires_whatever_its_weight(tmp_path: Path) -> None:
    """A flick at weight 10 pays a smaller share than a reset at 100 but fires more often; a minute of one
    car's play is 1,800 decisions at the preset's tick skip 4."""
    folder = RunFolder.create(tmp_path, "f", PRESET)
    labels = ("flick", "reset", "speed")
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["agent_steps", "iteration", "steps_per_second",
                      *(f"paid_abs_{label}" for label in labels),
                      *(f"fired_{label}" for label in labels)])  # fmt: skip
        for i in range(1, 100):
            out.writerow([i * 1000, i, 1000.0, 0.05, 0.1, 0.85, 0.01, 0.001, 1.0])
    shown = rewards.shares(folder, 60)
    ranked = shown["rewards"]
    assert [r["label"] for r in ranked] == ["speed", "reset", "flick"] and shown["fires_note"] == ""
    by = {r["label"]: r for r in ranked}
    assert by["flick"]["fire_rate"] == pytest.approx(0.01) and by["flick"][
        "fires_per_minute"
    ] == pytest.approx(18)
    assert by["reset"]["fires_per_minute"] == pytest.approx(1.8) and by["speed"][
        "fire_rate"
    ] == pytest.approx(1)

    old = RunFolder.create(tmp_path, "old", PRESET)  # metrics written before the count existed
    with old.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["agent_steps", "iteration", "steps_per_second", "paid_abs_flick"])
        for i in range(1, 100):
            out.writerow([i * 1000, i, 1000.0, 0.05])
    shown = rewards.shares(old, 60)
    assert shown["rewards"][0]["fire_rate"] is None and shown["rewards"][0]["fires_per_minute"] is None
    assert "appear once the bot trains with this version" in shown["fires_note"]


def test_reward_changes_wait_for_apply_which_refuses_a_problem(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    RunFolder.create(runs, "r", PRESET)
    api = Api(ROOT, runs)
    state = api.rewards("r")
    built_in = [kind for kind in state["catalog"] if not kind.get("own")]
    assert not state["pending"] and len(built_in) == 13 and "zero_sum.team_spirit" in state["help"]
    entries = [{"from": e["from"], "view": e["view"]} for e in state["rewards"]]
    entries[0]["view"]["weight"] = "lots"
    entries[3]["view"]["weight"] = 75
    staged = api.stage_rewards("r", entries)
    assert staged["pending"] and staged["rewards"][0]["problems"] and not staged["rewards"][3]["problems"]
    assert api.bot("r")["reward_changes"] == staged["changes"] and api.home()["bots"][0]["pending_count"] == 2
    assert "error" in api.start("r") and "fix these first" in api.apply("r")["error"]
    assert "weight = 75" not in (runs / "r" / "settings.toml").read_text(encoding="utf-8")
    entries[0]["view"]["weight"] = 0.5
    assert not any(e["problems"] for e in api.stage_rewards("r", entries)["rewards"])
    assert api.apply("r") == {"applied": 2, "restarting": False}
    config = RunFolder(runs / "r").load_settings()
    assert config.rewards[3].weight == 75 and config.rewards[0].weight == 0.5
    assert not api.rewards("r")["pending"]
    entries = [{"from": e["from"], "view": e["view"]} for e in api.rewards("r")["rewards"]]
    entries.pop()
    assert api.stage_rewards("r", entries)["changes"] == ["goal: removed"]
    api.discard("r")
    assert not api.rewards("r")["pending"] and api.bot("r")["reward_changes"] == []
    assert "error" in api.stage_rewards("r", [{"view": {"name": 3}}])


def test_the_new_bot_form_makes_a_bot_not_started_with_the_presets_comments(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    form = api.wizard("2v2_starter")
    asked = [item["path"] for step in form["steps"] for item in step["items"]]
    assert form["team_size"] == 2 and {"env.tick_skip", "model.encoder", "ppo.policy_lr"} <= set(asked)
    assert "error" in api.wizard("no-such-preset") and "error" in api.create_bot(
        "bad name", "2v2_starter", {}
    )
    assert api.create_bot("w", "2v2_starter", {"env.tick_skip": "x"})["errors"] == {
        "env.tick_skip": "Tick skip: 'x' is not a whole number"
    }
    made = api.create_bot(
        "w", "2v2_starter", {"env.tick_skip": "8", "model.encoder": "attention", "model.policy": "512, 512"}
    )
    assert made["made"] == "w" and len(made["adjusted"]) == 1, made  # no room is written for attention
    folder = RunFolder(runs / "w")
    config = folder.load_settings()
    assert config.model.attention_fast, "the same maths, 10% faster"
    assert (config.env.tick_skip, config.env.action_delay, config.env.obs_team_size) == (8, 1, 0), (
        "tick skip 8 and action delay 1"
    )
    assert observation_room(config) == 2, "an attention network reads its team size's room"
    assert config.model.encoder == "attention" and config.model.policy == (512, 512)
    text = folder.settings.read_text(encoding="utf-8")
    assert "# 2v2: 4 cars per game" in text and text.count("[[rewards]]") == PRESET.read_text().count(
        "[[rewards]]"
    )
    assert not folder.complete_checkpoints() and api.home()["bots"][0]["status"] == "stopped"
    assert load_toml(PRESET, NitroConfig).env.tick_skip == 4, "the preset itself is untouched"
    assert "already" in api.create_bot("w", "2v2_starter", {})["error"]


def test_a_new_bot_with_big_networks_gets_fp16_and_small_ones_keep_fp32(tmp_path: Path) -> None:
    """FP16 was about twice as fast for networks with a 1024-by-1024 layer and learned as well, and
    no faster with the presets' sizes; the form picks it for big networks unless it was asked."""
    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    big = api.create_bot("big", "2v2_starter", {"model.policy": "1024, 1024, 1024"})
    assert any("fp16" in note for note in big["adjusted"]), big
    config = RunFolder(runs / "big").load_settings()
    assert (config.run.precision, config.model.fused_norm) == ("fp16", True)
    assert not any("fp16" in note for note in config.advice())
    small = api.create_bot("small", "2v2_starter", {"model.policy": "512, 512"})
    assert not any("fp16" in note for note in small["adjusted"])
    assert RunFolder(runs / "small").load_settings().run.precision == "fp32"
    asked = api.create_bot("asked", "2v2_starter", {"model.value": "1024, 1024", "run.precision": "fp32"})
    assert not any("fp16" in note for note in asked["adjusted"])
    config = RunFolder(runs / "asked").load_settings()
    assert config.run.precision == "fp32" and any("fp16" in note for note in config.advice())
    assert large_networks((1024,), (1024, 256), (256,)), "the shared trunk feeds each head"
    assert not large_networks((), (1024,), (2048, 512))


def test_the_rewards_tab_and_the_new_bot_form_work_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "shown", PRESET)
    write_metrics(folder, [{"air": 0.1, "face_ball": 0.2, "strong_touch": 0.7}] * 100)
    api = Api(ROOT, runs)
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
            tab.click(".botcard")
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("text=What each reward pays")
            tab.wait_for_selector(".share >> text=Strong touch")
            tab.fill("#rw-3-weight", "70")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("#pending:not([hidden])")
            tab.uncheck("#rw-6-enabled")
            tab.fill("#rw-search", "boost")
            tab.wait_for_function("document.querySelectorAll('[data-reward]').length === 2")
            tab.fill("#rw-search", "")
            tab.click("[data-act=rw-add]")
            tab.fill("#rw-cat-q", "touch ball")
            tab.click("[data-rwadd=touch_ball]")
            tab.wait_for_selector(".badge.new")
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
            config = folder.load_settings()
            assert config.rewards[3].weight == 70 and not config.rewards[6].enabled
            assert config.rewards[-1].name == "touch_ball"

            tab.click(".nav [data-go=home]")
            tab.click("[data-act=newbot]")
            tab.click("[data-kind='preset:2v2_starter']")
            tab.click("[data-act=wiz-next]")
            tab.fill("#wz-name", "from-form")
            tab.click("[data-act=wiz-next]")
            tab.click("[data-brain=custom]")
            tab.fill("#wz-env-tick_skip", "6")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("#wizard >> text=Nothing is made until you press Make the bot")
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("h1:has-text('from-form')")
            made = RunFolder(runs / "from-form").load_settings()
            assert (made.env.team_size, made.env.tick_skip, made.env.action_delay) == (2, 6, 1)
        finally:
            browser.close()
    assert errors == []


def test_the_before_zero_sum_line_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    settings = tmp_path / "z.toml"
    settings.write_text(
        '[[rewards]]\nname = "velocity_player_to_ball"\nzero_sum = { team_spirit = 0 }\n'
        '[[rewards]]\nname = "goal"\nweight = 10\n',
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "z", settings)
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["agent_steps", "iteration", "steps_per_second", "paid_abs_velocity_player_to_ball",
                      "paid_abs_goal", "own_abs_velocity_player_to_ball", "own_abs_goal"])  # fmt: skip
        for i in range(1, 100):
            out.writerow([i * 1000, i, 1000.0, 0.1, 0.9, 3.0, 1.0])
    api = Api(ROOT, tmp_path / "runs")
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1400, "height": 900})
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            open_bot(tab)
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("#rw-aside >> text=Before zero-sum: 75%")
            assert tab.inner_text("#rw-aside").count("Before zero-sum") == 1, "only the zero-sum reward"
        finally:
            browser.close()


def test_the_reward_lists_sort_by_how_often_rewards_fire_and_hide_some(tmp_path: Path) -> None:
    """Sort by how often each reward triggers; hide continuous rewards or chosen ones, with the shares over
    the rewards shown. View only: settings.toml stays as it was."""
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    settings = tmp_path / "t.toml"
    settings.write_text(
        "[env]\ntick_skip = 8\n"
        '[[rewards]]\nname = "goal"\nweight = 100\n[[rewards]]\nname = "touch_ball"\nweight = 10\n'
        '[[rewards]]\nname = "velocity_player_to_ball"\nweight = 1\n',
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "t", settings)
    written = folder.settings.read_text(encoding="utf-8")
    labels = ("goal", "touch_ball", "velocity_player_to_ball")
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["agent_steps", "iteration", "steps_per_second",
                      *(f"paid_abs_{label}" for label in labels),
                      *(f"fired_{label}" for label in labels)])  # fmt: skip
        for i in range(1, 100):
            out.writerow([i * 1000, i, 1000.0, 0.5, 0.3, 0.2, 0.0005, 0.01, 1.0])
    api = Api(ROOT, tmp_path / "runs")
    shown = "Array.from(document.querySelectorAll('#rw-aside .share .n')).map(e => e.textContent).join('|')"
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
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("#rw-aside >> text=fires 9.0 a minute")  # tick skip 8: 900 a minute
            assert tab.evaluate(shown) == "Goal|Touch ball|Velocity player to ball"
            tab.select_option("#rw-sh-sort", "fires")
            tab.wait_for_function(f"{shown} === 'Velocity player to ball|Touch ball|Goal'")
            assert "fires 900 a minute (100% of steps)" in tab.inner_text("#rw-aside")
            tab.check("#rw-sh-cont")
            tab.wait_for_function(f"{shown} === 'Touch ball|Goal'")
            assert "share of shown rewards" in tab.inner_text("#rw-aside")
            assert tab.inner_text("#rw-aside .share .pct >> nth=1").strip() == "63%", "0.5 of the 0.8 shown"
            tab.click("[data-rwhide=goal]")
            tab.wait_for_function(f"{shown} === 'Touch ball'")
            assert tab.inner_text("#rw-aside .share .pct").strip() == "100%"
            assert api._prefs()["reward_view"] == {
                "sort": "fires",
                "bots": {"t": {"hide_continuous": True, "hidden": ["goal"]}},
            }
            tab.click("#rw-aside >> text=Show 1 hidden")
            tab.click("#rw-aside [data-rwunhide=goal]")
            tab.wait_for_function(f"{shown} === 'Touch ball|Goal'")
            assert folder.settings.read_text(encoding="utf-8") == written, "view only"
        finally:
            browser.close()
    assert errors == []
